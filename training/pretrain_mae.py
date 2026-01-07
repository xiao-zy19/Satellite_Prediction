"""
MAE Pretrain Trainer for 64-channel satellite embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import os
import re
import logging
from tqdm import tqdm
from pathlib import Path

from .training_utils import (
    get_optimizer,
    get_scheduler,
    AverageMeter,
    set_seed,
    clip_grad_norm,
)

logger = logging.getLogger(__name__)


class MAETrainer:
    """Trainer for MAE pretraining."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 1,
        num_epochs: int = 10,
        max_grad_norm: float = 1.0,
        save_dir: str = "checkpoints/pretrain",
        log_interval: int = 100,
        save_interval: int = 1000,
        save_total_limit: int = 5,
        use_wandb: bool = False,
        wandb_project: str = "mllm-pretrain",
        device: str = "cuda",
        precision: str = "bf16",
    ):
        """
        Initialize MAE Trainer.

        Args:
            model: MAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_epochs: Number of warmup epochs
            num_epochs: Total epochs
            max_grad_norm: Maximum gradient norm
            save_dir: Directory to save checkpoints
            log_interval: Steps between logging
            save_interval: Steps between saving
            save_total_limit: Max number of step checkpoints to keep (epoch/best/final always kept)
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
            device: Device to train on
            precision: Training precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_total_limit = save_total_limit
        self.device = device
        self.use_wandb = use_wandb

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = get_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = len(train_loader) * warmup_epochs
        self.scheduler = get_scheduler(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        # Mixed precision
        self.use_amp = precision in ["bf16", "fp16"]
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

        # Metrics
        self.global_step = 0
        self.best_loss = float('inf')

        # Wandb
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, config={
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'num_epochs': num_epochs,
                'batch_size': train_loader.batch_size,
            })

        logger.info(f"MAE Trainer initialized")
        logger.info(f"Training for {num_epochs} epochs, {num_training_steps} steps")

    def train(self) -> Dict:
        """
        Run training loop.

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self._train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['learning_rate'].append(self.scheduler.get_last_lr()[0])

            # Validate
            if self.val_loader is not None:
                val_loss = self._validate()
                history['val_loss'].append(val_loss)

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint('best', epoch, val_loss)

            # Save epoch checkpoint
            self._save_checkpoint(f'epoch_{epoch}', epoch, train_loss)

        # Save final checkpoint
        self._save_checkpoint('final', self.num_epochs - 1, train_loss)

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            ids_keep = batch['ids_keep'].to(self.device)
            ids_restore = batch['ids_restore'].to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                pred, loss = self.model(
                    images,
                    mask=masks,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm(self.model, self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm(self.model, self.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()

            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                })

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': loss_meter.avg,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step,
                    })

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint(f'step_{self.global_step}', epoch, loss_meter.avg)

        return loss_meter.avg

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        loss_meter = AverageMeter()

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            ids_keep = batch['ids_keep'].to(self.device)
            ids_restore = batch['ids_restore'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                pred, loss = self.model(
                    images,
                    mask=masks,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )

            loss_meter.update(loss.item(), images.size(0))

        logger.info(f"Validation loss: {loss_meter.avg:.4f}")

        if self.use_wandb:
            import wandb
            wandb.log({
                'val/loss': loss_meter.avg,
                'val/step': self.global_step,
            })

        return loss_meter.avg

    def _save_checkpoint(self, name: str, epoch: int, loss: float) -> None:
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            'best_loss': self.best_loss,
        }

        save_path = self.save_dir / f'{name}.pt'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

        # Clean up old step checkpoints (keep epoch/best/final)
        if name.startswith('step_') and self.save_total_limit > 0:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """
        Clean up old step checkpoints, keeping only the most recent ones.

        保留策略:
        - epoch_*.pt: 全部保留 (每个epoch的checkpoint)
        - best.pt: 保留 (最佳模型)
        - final.pt: 保留 (最终模型)
        - step_*.pt: 只保留最近的 save_total_limit 个
        """
        # Find all step checkpoints
        step_pattern = re.compile(r'^step_(\d+)\.pt$')
        step_checkpoints: List[tuple] = []  # [(step_num, filepath), ...]

        for f in self.save_dir.iterdir():
            if f.is_file():
                match = step_pattern.match(f.name)
                if match:
                    step_num = int(match.group(1))
                    step_checkpoints.append((step_num, f))

        # Sort by step number (oldest first)
        step_checkpoints.sort(key=lambda x: x[0])

        # Delete old checkpoints if exceeding limit
        num_to_delete = len(step_checkpoints) - self.save_total_limit
        if num_to_delete > 0:
            for step_num, filepath in step_checkpoints[:num_to_delete]:
                try:
                    filepath.unlink()
                    logger.info(f"Deleted old checkpoint: {filepath.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {filepath}: {e}")


def pretrain_mae(
    model: nn.Module,
    data_dir: str,
    config: Dict,
    output_dir: str = "checkpoints/pretrain",
) -> Dict:
    """
    Main function to run MAE pretraining.

    Args:
        model: MAE model
        data_dir: Directory containing training data
        config: Training configuration
        output_dir: Output directory

    Returns:
        Training history
    """
    from data.pretrain_dataset import get_pretrain_dataloader

    # Set seed
    set_seed(config.get('seed', 42))

    # Create data loaders
    train_loader = get_pretrain_dataloader(
        data_dir=data_dir,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 8),
        mask_ratio=config.get('mask_ratio', 0.75),
    )

    # Create trainer
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.05),
        warmup_epochs=config.get('warmup_epochs', 1),
        num_epochs=config.get('num_epochs', 10),
        save_dir=output_dir,
        use_wandb=config.get('use_wandb', False),
    )

    # Train
    history = trainer.train()

    return history
