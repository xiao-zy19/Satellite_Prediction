"""
General utility functions for MLLM Training.
"""

import os
import sys
import json
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def format_memory(bytes: int) -> str:
    """Format bytes into human-readable memory."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"


def get_gpu_memory_info() -> Dict[str, str]:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {"status": "No GPU available"}

    info = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        allocated = torch.cuda.memory_allocated(i)
        cached = torch.cuda.memory_reserved(i)
        free = total - allocated

        info[f"GPU {i}"] = {
            "total": format_memory(total),
            "allocated": format_memory(allocated),
            "cached": format_memory(cached),
            "free": format_memory(free),
        }

    return info


def print_gpu_memory():
    """Print GPU memory usage."""
    info = get_gpu_memory_info()
    print("\nGPU Memory Usage:")
    print("-" * 40)
    for gpu, stats in info.items():
        if isinstance(stats, dict):
            print(f"{gpu}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"{gpu}: {stats}")
    print("-" * 40)
