"""
DPO Trainer for Direct Preference Optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import logging
from tqdm import tqdm
from pathlib import Path
import copy

from .training_utils import (
    get_optimizer,
    get_scheduler,
    AverageMeter,
    compute_metrics,
    set_seed,
    clip_grad_norm,
)

logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization.

    DPO optimizes a policy to prefer 'chosen' responses over 'rejected' ones
    without needing a separate reward model.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        train_loader: DataLoader = None,
        val_loader: Optional[DataLoader] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        max_grad_norm: float = 0.5,
        save_dir: str = "checkpoints/rl",
        log_interval: int = 20,
        use_wandb: bool = False,
        wandb_project: str = "mllm-rl",
        device: str = "cuda",
        precision: str = "bf16",
    ):
        """
        Initialize DPO Trainer.

        Args:
            model: Policy model to optimize
            ref_model: Reference model (frozen)
            train_loader: Training data loader
            val_loader: Validation data loader
            beta: KL penalty coefficient
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            num_epochs: Number of epochs
            max_grad_norm: Max gradient norm
            save_dir: Save directory
            log_interval: Log interval
            use_wandb: Use wandb
            wandb_project: Wandb project
            device: Device
            precision: Precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.device = device
        self.use_wandb = use_wandb

        # Reference model
        if ref_model is None:
            # Create a copy as reference
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model
        self.ref_model = self.ref_model.to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = get_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        if train_loader is not None:
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

        # Metrics
        self.global_step = 0
        self.best_reward = -float('inf')

        logger.info(f"DPO Trainer initialized with beta={beta}")

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen under policy
            policy_rejected_logps: Log probs of rejected under policy
            reference_chosen_logps: Log probs of chosen under reference
            reference_rejected_logps: Log probs of rejected under reference

        Returns:
            Tuple of (loss, metrics dict)
        """
        # Compute log ratios
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # DPO loss
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits).mean()

        # Metrics
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        metrics = {
            'loss': loss.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'accuracy': (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return loss, metrics

    def get_logprobs(
        self,
        model: nn.Module,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get log probabilities for given inputs.

        Args:
            model: Model to use
            images: Image tensor
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Label token IDs

        Returns:
            Log probabilities tensor
        """
        outputs = model(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log probs for labels
        labels = labels.clone()
        labels[labels == -100] = 0  # Replace ignore index

        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding
        mask = (attention_mask == 1) & (labels != -100)
        per_sequence_logps = (per_token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return per_sequence_logps

    def train(self) -> Dict:
        """Run training loop."""
        history = {
            'loss': [],
            'reward_margin': [],
            'accuracy': [],
        }

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Train
            epoch_metrics = self._train_epoch(epoch)
            history['loss'].append(epoch_metrics['loss'])
            history['reward_margin'].append(epoch_metrics['reward_margin'])
            history['accuracy'].append(epoch_metrics['accuracy'])

            # Validate
            if self.val_loader is not None:
                val_metrics = self._validate()

                # Save best model
                if val_metrics['reward_margin'] > self.best_reward:
                    self.best_reward = val_metrics['reward_margin']
                    self._save_checkpoint('best', epoch, val_metrics)

            # Save epoch checkpoint
            self._save_checkpoint(f'epoch_{epoch}', epoch, epoch_metrics)

        return history

    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        loss_meter = AverageMeter()
        margin_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"DPO Training Epoch {epoch + 1}")

        for batch in pbar:
            # Move to device
            images = batch['images'].to(self.device)
            chosen_ids = batch['chosen_ids']['input_ids'].to(self.device)
            chosen_mask = batch['chosen_ids']['attention_mask'].to(self.device)
            rejected_ids = batch['rejected_ids']['input_ids'].to(self.device)
            rejected_mask = batch['rejected_ids']['attention_mask'].to(self.device)

            # Get policy log probs
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                policy_chosen_logps = self.get_logprobs(
                    self.model, images, chosen_ids, chosen_mask, chosen_ids
                )
                policy_rejected_logps = self.get_logprobs(
                    self.model, images, rejected_ids, rejected_mask, rejected_ids
                )

            # Get reference log probs
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                    ref_chosen_logps = self.get_logprobs(
                        self.ref_model, images, chosen_ids, chosen_mask, chosen_ids
                    )
                    ref_rejected_logps = self.get_logprobs(
                        self.ref_model, images, rejected_ids, rejected_mask, rejected_ids
                    )

            # Compute loss
            loss, metrics = self.compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(self.model, self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            loss_meter.update(metrics['loss'])
            margin_meter.update(metrics['reward_margin'])
            acc_meter.update(metrics['accuracy'])
            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'margin': f'{margin_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.4f}',
                })

        return {
            'loss': loss_meter.avg,
            'reward_margin': margin_meter.avg,
            'accuracy': acc_meter.avg,
        }

    @torch.no_grad()
    def _validate(self) -> Dict:
        """Run validation."""
        self.model.eval()

        loss_meter = AverageMeter()
        margin_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['images'].to(self.device)
            chosen_ids = batch['chosen_ids']['input_ids'].to(self.device)
            chosen_mask = batch['chosen_ids']['attention_mask'].to(self.device)
            rejected_ids = batch['rejected_ids']['input_ids'].to(self.device)
            rejected_mask = batch['rejected_ids']['attention_mask'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.dtype):
                policy_chosen_logps = self.get_logprobs(
                    self.model, images, chosen_ids, chosen_mask, chosen_ids
                )
                policy_rejected_logps = self.get_logprobs(
                    self.model, images, rejected_ids, rejected_mask, rejected_ids
                )
                ref_chosen_logps = self.get_logprobs(
                    self.ref_model, images, chosen_ids, chosen_mask, chosen_ids
                )
                ref_rejected_logps = self.get_logprobs(
                    self.ref_model, images, rejected_ids, rejected_mask, rejected_ids
                )

            _, metrics = self.compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            loss_meter.update(metrics['loss'])
            margin_meter.update(metrics['reward_margin'])
            acc_meter.update(metrics['accuracy'])

        metrics = {
            'loss': loss_meter.avg,
            'reward_margin': margin_meter.avg,
            'accuracy': acc_meter.avg,
        }

        logger.info(f"Validation - Loss: {metrics['loss']:.4f}, "
                    f"Margin: {metrics['reward_margin']:.4f}, "
                    f"Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict) -> None:
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'metrics': metrics,
            'best_reward': self.best_reward,
        }

        save_path = self.save_dir / f'{name}.pt'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


def train_dpo(
    model: nn.Module,
    preference_file: str,
    data_dir: str,
    config: Dict,
    output_dir: str = "checkpoints/rl",
) -> Dict:
    """
    Main function to run DPO training.

    Args:
        model: Model to optimize
        preference_file: Path to preference pairs JSON
        data_dir: Data directory
        config: Training configuration
        output_dir: Output directory

    Returns:
        Training history
    """
    from data.dpo_dataset import get_dpo_dataloader

    # Set seed
    set_seed(config.get('seed', 42))

    # Create data loader
    train_loader = get_dpo_dataloader(
        data_dir=data_dir,
        preference_file=preference_file,
        batch_size=config.get('batch_size', 2),
        num_workers=config.get('num_workers', 4),
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        train_loader=train_loader,
        beta=config.get('beta', 0.1),
        learning_rate=config.get('learning_rate', 1e-6),
        num_epochs=config.get('num_epochs', 5),
        save_dir=output_dir,
        use_wandb=config.get('use_wandb', False),
    )

    # Train
    history = trainer.train()

    return history
