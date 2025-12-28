"""
Self-supervised pretraining methods
"""

from .simclr import SimCLR, SimCLRTrainer
from .mae import MAE, MAETrainer

__all__ = ['SimCLR', 'SimCLRTrainer', 'MAE', 'MAETrainer']
