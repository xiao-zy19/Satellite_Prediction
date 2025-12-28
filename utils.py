"""
Utility functions for training and evaluation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Optional
import logging
from datetime import datetime

import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Get computation device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(exp_name: str, log_dir: str = config.LOG_DIR) -> logging.Logger:
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


def get_scheduler(optimizer: torch.optim.Optimizer, train_config) -> Optional[object]:
    """Create learning rate scheduler."""
    if train_config.scheduler == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_config.t_max,
            T_mult=train_config.t_mult,
            eta_min=train_config.eta_min
        )
    elif train_config.scheduler == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif train_config.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {train_config.scheduler}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Handle edge cases
    if len(y_true) < 2:
        return {'pearson_r': 0, 'p_value': 1, 'r2': 0, 'mae': 0, 'rmse': 0}

    pearson_r_val, p_value = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        'pearson_r': pearson_r_val,
        'p_value': p_value,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


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


def save_checkpoint(state: Dict, exp_name: str, filename: str = 'best_model.pth'):
    """Save model checkpoint."""
    exp_dir = os.path.join(config.CHECKPOINT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    filepath = os.path.join(exp_dir, filename)
    torch.save(state, filepath)
    return filepath


def load_checkpoint(filepath: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a formatted way."""
    print(f"{prefix}Pearson r: {metrics['pearson_r']:.4f}")
    print(f"{prefix}R2 Score:  {metrics['r2']:.4f}")
    print(f"{prefix}MAE:       {metrics['mae']:.4f}")
    print(f"{prefix}RMSE:      {metrics['rmse']:.4f}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GradualWarmupScheduler:
    """Gradually warmup learning rate for first few epochs."""

    def __init__(self, optimizer, warmup_epochs: int, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * warmup_factor
        elif self.after_scheduler:
            self.after_scheduler.step()

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
