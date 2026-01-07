"""
Training module for MLLM training pipeline.
Provides trainers for pretrain, SFT, and RL stages.
"""

from .pretrain_mae import MAETrainer, pretrain_mae
from .sft_trainer import SFTTrainer, train_sft
from .dpo_trainer import DPOTrainer, train_dpo
from .training_utils import (
    get_optimizer,
    get_scheduler,
    EarlyStopping,
    AverageMeter,
    compute_metrics,
)

__all__ = [
    "MAETrainer",
    "pretrain_mae",
    "SFTTrainer",
    "train_sft",
    "DPOTrainer",
    "train_dpo",
    "get_optimizer",
    "get_scheduler",
    "EarlyStopping",
    "AverageMeter",
    "compute_metrics",
]
