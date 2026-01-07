"""
Training utility functions.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Adam betas
        eps: Adam epsilon

    Returns:
        Optimizer
    """
    # Separate parameters with and without weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    optimizer_groups = [
        {'params': params_with_decay, 'weight_decay': weight_decay},
        {'params': params_without_decay, 'weight_decay': 0.0},
    ]

    optimizer = AdamW(
        optimizer_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    scheduler_type: str = "cosine",
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        Scheduler
    """
    if scheduler_type == "cosine":
        # Warmup + Cosine decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=optimizer.defaults['lr'] * min_lr_ratio,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
    elif scheduler_type == "linear":
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "max" or "min"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # R² score
    r2 = r2_score(y_true, y_pred)

    # Pearson correlation
    pearson_r, p_value = pearsonr(y_true, y_pred)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        'r2': r2,
        'pearson_r': pearson_r,
        'p_value': p_value,
        'mae': mae,
        'rmse': rmse,
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_grad_norm(model: nn.Module) -> float:
    """Get gradient norm of model parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    """Clip gradient norm."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class GradientScaler:
    """Gradient scaler for mixed precision training."""

    def __init__(self, enabled: bool = True):
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def scale(self, loss):
        return self.scaler.scale(loss)

    def step(self, optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    def unscale_(self, optimizer):
        self.scaler.unscale_(optimizer)
