"""
Models module for MLLM Training.
Provides modified MLLM models for 64-channel satellite embeddings.

支持:
- Qwen2-VL (Conv2d patch embedding)
- Qwen2.5-VL (Conv3d patch embedding)

Note: Some models require additional dependencies (transformers).
Use try/except for optional imports.
"""

# MAE components (supports both Conv2d and Conv3d)
from .mae_decoder import (
    MAEDecoder,
    TransformerBlock,
    MAEForPretrainingV2,
    SimpleMAEForPretraining,
    create_mae_for_qwen25_vl,
    create_mae_for_qwen2_vl,
)

# Backward compatibility aliases
MAEForPretraining = MAEForPretrainingV2
SimpleMAEForPretrainingV2 = SimpleMAEForPretraining

from .regression_head import RegressionHead
from .model_utils import (
    count_parameters,
    freeze_model,
    unfreeze_model,
    get_trainable_parameters,
)

# Optional: Requires transformers library
_HAS_TRANSFORMERS = False
Qwen2VL64Ch = None
Qwen25VL64Ch = None
modify_vision_encoder_for_64ch = None
modify_qwen25_vision_encoder_for_64ch = None
load_pretrained_64ch_model = None

try:
    from .qwen_vl_64ch import (
        Qwen2VL64Ch,
        modify_vision_encoder_for_64ch,
        load_pretrained_64ch_model,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    from .qwen25_vl_64ch import (
        Qwen25VL64Ch,
        modify_qwen25_vision_encoder_for_64ch,
        load_pretrained_64ch_model as load_pretrained_64ch_model_v2,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    load_pretrained_64ch_model_v2 = None


def check_transformers():
    """Check if transformers library is available."""
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers library is required for Qwen VL models. "
            "Install with: pip install transformers>=4.45.0"
        )
    return True


def get_model_class(model_type: str):
    """Get the appropriate model class for the given type."""
    check_transformers()
    if model_type == "qwen2.5-vl":
        if Qwen25VL64Ch is None:
            raise ImportError("Qwen25VL64Ch not available. Check transformers version.")
        return Qwen25VL64Ch
    elif model_type == "qwen2-vl":
        if Qwen2VL64Ch is None:
            raise ImportError("Qwen2VL64Ch not available. Check transformers version.")
        return Qwen2VL64Ch
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    # MAE components (from mae_decoder)
    "MAEDecoder",
    "TransformerBlock",
    "MAEForPretraining",       # Alias for MAEForPretrainingV2
    "MAEForPretrainingV2",
    "SimpleMAEForPretraining",
    "SimpleMAEForPretrainingV2",  # Alias for SimpleMAEForPretraining
    "create_mae_for_qwen25_vl",
    "create_mae_for_qwen2_vl",
    # Other utilities
    "RegressionHead",
    "count_parameters",
    "freeze_model",
    "unfreeze_model",
    "get_trainable_parameters",
    # Optional (requires transformers)
    "Qwen2VL64Ch",
    "Qwen25VL64Ch",
    "modify_vision_encoder_for_64ch",
    "modify_qwen25_vision_encoder_for_64ch",
    "load_pretrained_64ch_model",
    "check_transformers",
    "get_model_class",
]
