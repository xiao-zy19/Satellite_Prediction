"""
Model utility functions for MLLM training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module) -> None:
    """
    Freeze all parameters in a model.

    Args:
        model: Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    logger.info(f"Froze {count_parameters(model)} parameters")


def unfreeze_model(model: nn.Module) -> None:
    """
    Unfreeze all parameters in a model.

    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info(f"Unfroze {count_parameters(model)} parameters")


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers by name.

    Args:
        model: Model
        layer_names: List of layer name patterns to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in layer_names):
            param.requires_grad = False
            frozen_count += param.numel()

    logger.info(f"Froze {frozen_count} parameters in layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Unfreeze specific layers by name.

    Args:
        model: Model
        layer_names: List of layer name patterns to unfreeze
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in layer_names):
            param.requires_grad = True
            unfrozen_count += param.numel()

    logger.info(f"Unfroze {unfrozen_count} parameters in layers: {layer_names}")


def get_trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """
    Get iterator over trainable parameters.

    Args:
        model: Model

    Yields:
        Trainable parameters
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    lr_multipliers: Optional[Dict[str, float]] = None,
    weight_decay: float = 0.01,
    no_decay_patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Get parameter groups with different learning rates and weight decay.

    Args:
        model: Model
        base_lr: Base learning rate
        lr_multipliers: Dict mapping layer patterns to LR multipliers
        weight_decay: Weight decay
        no_decay_patterns: Patterns for parameters without weight decay

    Returns:
        List of parameter groups for optimizer
    """
    lr_multipliers = lr_multipliers or {}
    no_decay_patterns = no_decay_patterns or ['bias', 'LayerNorm', 'layer_norm']

    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine learning rate
        lr = base_lr
        for pattern, multiplier in lr_multipliers.items():
            if pattern in name:
                lr = base_lr * multiplier
                break

        # Determine weight decay
        wd = weight_decay
        if any(nd in name for nd in no_decay_patterns):
            wd = 0.0

        # Group key
        key = (lr, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # Convert to list
    groups = [
        {'params': params, 'lr': lr, 'weight_decay': wd}
        for (lr, wd), params in param_groups.items()
    ]

    return groups


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    map_location: str = 'cpu',
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        map_location: Device to map checkpoint to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    save_path: str,
    metrics: Optional[Dict] = None,
    scheduler=None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        save_path: Path to save checkpoint
        metrics: Optional metrics dictionary
        scheduler: Optional scheduler state
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {},
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in MB.

    Args:
        model: Model

    Returns:
        Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: Model to summarize
    """
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {frozen_params:,}")
    print(f"Model Size:           {get_model_size_mb(model):.2f} MB")
    print("=" * 60)

    # Print layer-wise summary
    print("\nLayer-wise Parameter Count:")
    print("-" * 60)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:30s}: {params:>12,} ({trainable:,} trainable)")
    print("-" * 60)


def gradient_checkpointing_enable(model: nn.Module) -> None:
    """
    Enable gradient checkpointing for memory efficiency.

    Args:
        model: Model
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    else:
        logger.warning("Model does not support gradient checkpointing")


def prepare_model_for_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
    use_bf16: bool = True,
) -> nn.Module:
    """
    Prepare model for training with various optimizations.

    Args:
        model: Model
        use_gradient_checkpointing: Enable gradient checkpointing
        use_bf16: Use bfloat16 precision

    Returns:
        Prepared model
    """
    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        gradient_checkpointing_enable(model)

    # Set to training mode
    model.train()

    # Convert to bf16 if needed
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)

    return model
