"""
SFT Trainer for supervised fine-tuning on population prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import os
import logging
from tqdm import tqdm
from pathlib import Path

from .training_utils import (
    get_optimizer,
    get_scheduler,
    AverageMeter,
    EarlyStopping,
    compute_metrics,
    set_seed,
    clip_grad_norm,
)

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_epochs: int = 20,
        max_grad_norm: float = 1.0,
        freeze_encoder_epochs: int = 5,
        unfreeze_lr: float = 1e-6,
        save_dir: str = "checkpoints/sft",
        log_interval: int = 50,
        use_wandb: bool = False,
        wandb_project: str = "mllm-sft",
        device: str = "cuda",
        precision: str = "bf16",
        early_stopping_patience: int = 5,
    ):
        """
        Initialize SFT Trainer.

        Args:
            model: Model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            num_epochs: Total epochs
            max_grad_norm: Max gradient norm
            freeze_encoder_epochs: Epochs to freeze encoder
            unfreeze_lr: LR after unfreezing
            save_dir: Save directory
            log_interval: Log interval
            use_wandb: Use wandb
            wandb_project: Wandb project
            device: Device
            precision: Precision
            early_stopping_patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.unfreeze_lr = unfreeze_lr
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.device = device
        self.use_wandb = use_wandb

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Freeze encoder initially
        if freeze_encoder_epochs > 0:
            self._freeze_encoder()

        # Optimizer
        self.optimizer = get_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        self.scheduler = get_scheduler(
            self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        # Mixed precision
        self.use_amp = precision in ["bf16", "fp16"]
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode="max",
        )

        # Metrics
        self.global_step = 0
        self.best_r2 = -float('inf')

        # Wandb
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project)

        logger.info(f"SFT Trainer initialized")

    def _freeze_encoder(self):
        """Freeze vision encoder."""
        if hasattr(self.model, 'freeze_vision_encoder'):
            self.model.freeze_vision_encoder()
        else:
            for name, param in self.model.named_parameters():
                if 'visual' in name or 'vision' in name:
                    param.requires_grad = False
        logger.info("Vision encoder frozen")

    def _unfreeze_encoder(self):
        """Unfreeze vision encoder with lower LR."""
        if hasattr(self.model, 'unfreeze_vision_encoder'):
            self.model.unfreeze_vision_encoder()
        else:
            for name, param in self.model.named_parameters():
                if 'visual' in name or 'vision' in name:
                    param.requires_grad = True

        # Update optimizer with new LR for encoder
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.unfreeze_lr

        logger.info(f"Vision encoder unfrozen, LR set to {self.unfreeze_lr}")

    def train(self) -> Dict:
        """Run training loop."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_r2': [],
            'learning_rate': [],
        }

        for epoch in range(self.num_epochs):
            # Unfreeze encoder after specified epochs
            if epoch == self.freeze_encoder_epochs:
                self._unfreeze_encoder()

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self._train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['learning_rate'].append(self.scheduler.get_last_lr()[0])

            # Validate
            val_metrics = self._validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_r2'].append(val_metrics['r2'])

            # Save best model
            if val_metrics['r2'] > self.best_r2:
                self.best_r2 = val_metrics['r2']
                self._save_checkpoint('best', epoch, val_metrics)

            # Early stopping
            if self.early_stopping(val_metrics['r2']):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Test evaluation
        if self.test_loader is not None:
            test_metrics = self._evaluate(self.test_loader, "Test")
            history['test_metrics'] = test_metrics

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in pbar:
            # Move to device
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images)

                # Compute loss (depends on model output format)
                if isinstance(outputs, dict):
                    logits = outputs.get('regression', outputs.get('logits'))
                else:
                    logits = outputs

                loss = F.mse_loss(logits.squeeze(), labels.squeeze())

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

        return loss_meter.avg

    @torch.no_grad()
    def _validate(self) -> Dict:
        """Run validation."""
        return self._evaluate(self.val_loader, "Validation")

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, name: str) -> Dict:
        """Evaluate on a data loader."""
        self.model.eval()
        loss_meter = AverageMeter()

        all_preds = []
        all_labels = []

        for batch in tqdm(loader, desc=f"Evaluating ({name})"):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images)

                if isinstance(outputs, dict):
                    logits = outputs.get('regression', outputs.get('logits'))
                else:
                    logits = outputs

                loss = F.mse_loss(logits.squeeze(), labels.squeeze())

            loss_meter.update(loss.item(), images.size(0))

            all_preds.extend(logits.squeeze().cpu().tolist())
            all_labels.extend(labels.squeeze().cpu().tolist())

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = loss_meter.avg

        logger.info(f"{name} - Loss: {metrics['loss']:.4f}, R²: {metrics['r2']:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

        if self.use_wandb:
            import wandb
            wandb.log({
                f'{name.lower()}/loss': metrics['loss'],
                f'{name.lower()}/r2': metrics['r2'],
                f'{name.lower()}/mae': metrics['mae'],
                f'{name.lower()}/rmse': metrics['rmse'],
            })

        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict) -> None:
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'metrics': metrics,
            'best_r2': self.best_r2,
        }

        save_path = self.save_dir / f'{name}.pt'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


def train_sft(
    model: nn.Module,
    data_dir: str,
    labels_file: str,
    config: Dict,
    output_dir: str = "checkpoints/sft",
) -> Dict:
    """
    Main function to run SFT training.

    Args:
        model: Model to fine-tune
        data_dir: Data directory
        labels_file: Labels file path
        config: Training configuration
        output_dir: Output directory

    Returns:
        Training history
    """
    from data.sft_dataset import get_sft_dataloader

    # Set seed
    set_seed(config.get('seed', 42))

    # Create data loaders
    train_loader = get_sft_dataloader(
        data_dir=data_dir,
        labels_file=labels_file,
        split="train",
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
    )

    val_loader = get_sft_dataloader(
        data_dir=data_dir,
        labels_file=labels_file,
        split="val",
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
    )

    test_loader = get_sft_dataloader(
        data_dir=data_dir,
        labels_file=labels_file,
        split="test",
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config.get('learning_rate', 1e-5),
        num_epochs=config.get('num_epochs', 20),
        save_dir=output_dir,
        use_wandb=config.get('use_wandb', False),
    )

    # Train
    history = trainer.train()

    return history
