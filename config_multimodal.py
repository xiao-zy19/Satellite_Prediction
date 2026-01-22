"""
Configuration for multimodal experiments (Image + Policy features)

This extends the base config with multimodal-specific configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import config


@dataclass
class MultiModalConfig:
    """Configuration for multimodal model."""
    name: str = "multimodal"

    # Image encoder type
    image_encoder_type: str = "light_cnn"  # mlp, light_cnn, resnet
    image_feature_dim: int = 64  # Project image features to this dim (default 64)

    # LightCNN specific (when image_encoder_type == "light_cnn")
    light_cnn_channels: list = None  # e.g., [32, 64, 128], None = use default
    light_cnn_kernels: list = None   # e.g., [3, 3, 3], None = use default
    use_batch_norm: bool = True

    # ResNet specific (when image_encoder_type == "resnet")
    resnet_model_name: str = "resnet18"  # resnet10, resnet18, resnet34, resnet50, resnet101
    use_pretrained_resnet: bool = False

    # Policy features
    policy_feature_dim: int = 12  # Raw policy feature dim
    policy_hidden_dim: int = 64   # Hidden dim if using policy encoder
    use_policy_encoder: bool = False  # Default: use raw 12-dim features

    # Fusion strategy
    fusion_type: str = "concat"  # concat, gated, attention, film

    # Aggregation (for city-level)
    aggregation: str = "mean"

    # Common
    dropout_rate: float = 0.3


@dataclass
class MultiModalTrainConfig:
    """Training configuration for multimodal experiments."""
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 60

    # Scheduler
    scheduler: str = "cosine_warm_restarts"
    t_max: int = 10
    t_mult: int = 2
    eta_min: float = 1e-7

    # Training mode
    training_mode: str = "city_level"  # city_level or patch_level

    # Patch-level specific
    patch_level_aggregation: str = "trimmed_mean"
    patch_level_trim_ratio: float = 0.1


@dataclass
class MultiModalExperimentConfig:
    """Full experiment configuration for multimodal training."""
    exp_name: str = "multimodal_default"
    model_config: MultiModalConfig = field(default_factory=MultiModalConfig)
    train_config: MultiModalTrainConfig = field(default_factory=MultiModalTrainConfig)
    device: str = "cuda"
    num_workers: int = 8
    wandb_enabled: bool = True
    wandb_project: str = "population-multimodal"


# =============================================================================
# Multimodal Experiment Presets
# =============================================================================

MULTIMODAL_EXPERIMENTS = {
    # ==========================================================================
    # City-level multimodal experiments
    # ==========================================================================

    # --- LightCNN + Concat Fusion (baseline) ---
    "mm_cnn_concat": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean"
        )
    ),
    "mm_cnn_concat_median": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_median",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="median"
        )
    ),
    "mm_cnn_concat_trimmed": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_trimmed",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean"
        )
    ),

    # --- LightCNN + Gated Fusion ---
    "mm_cnn_gated": MultiModalExperimentConfig(
        exp_name="mm_cnn_gated",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean"
        )
    ),
    "mm_cnn_gated_trimmed": MultiModalExperimentConfig(
        exp_name="mm_cnn_gated_trimmed",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="trimmed_mean"
        )
    ),

    # --- LightCNN + Attention Fusion ---
    "mm_cnn_attention": MultiModalExperimentConfig(
        exp_name="mm_cnn_attention",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="mean"
        )
    ),
    "mm_cnn_attention_trimmed": MultiModalExperimentConfig(
        exp_name="mm_cnn_attention_trimmed",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="trimmed_mean"
        )
    ),

    # --- LightCNN + FiLM Fusion ---
    "mm_cnn_film": MultiModalExperimentConfig(
        exp_name="mm_cnn_film",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean"
        )
    ),
    "mm_cnn_film_trimmed": MultiModalExperimentConfig(
        exp_name="mm_cnn_film_trimmed",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="trimmed_mean"
        )
    ),

    # --- MLP + Different Fusions ---
    "mm_mlp_concat": MultiModalExperimentConfig(
        exp_name="mm_mlp_concat",
        model_config=MultiModalConfig(
            image_encoder_type="mlp",
            fusion_type="concat",
            aggregation="mean"
        )
    ),
    "mm_mlp_gated": MultiModalExperimentConfig(
        exp_name="mm_mlp_gated",
        model_config=MultiModalConfig(
            image_encoder_type="mlp",
            fusion_type="gated",
            aggregation="mean"
        )
    ),

    # --- ResNet18 + Different Fusions ---
    "mm_resnet18_concat": MultiModalExperimentConfig(
        exp_name="mm_resnet18_concat",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="mean"
        )
    ),
    "mm_resnet18_gated": MultiModalExperimentConfig(
        exp_name="mm_resnet18_gated",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="gated",
            aggregation="mean"
        )
    ),
    "mm_resnet18_film": MultiModalExperimentConfig(
        exp_name="mm_resnet18_film",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="film",
            aggregation="mean"
        )
    ),

    # ==========================================================================
    # Patch-level multimodal experiments
    # ==========================================================================
    "mm_cnn_concat_patch": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_patch",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat"
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "mm_cnn_gated_patch": MultiModalExperimentConfig(
        exp_name="mm_cnn_gated_patch",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated"
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "mm_cnn_film_patch": MultiModalExperimentConfig(
        exp_name="mm_cnn_film_patch",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film"
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),

    # --- ResNet patch-level ---
    "mm_resnet18_concat_patch": MultiModalExperimentConfig(
        exp_name="mm_resnet18_concat_patch",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat"
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean"
        )
    ),

    # ==========================================================================
    # Custom encoder experiments (examples)
    # ==========================================================================

    # Custom LightCNN with smaller channels
    "mm_cnn_small_concat": MultiModalExperimentConfig(
        exp_name="mm_cnn_small_concat",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            light_cnn_channels=[16, 32, 64],  # Smaller than default [32, 64, 128]
            fusion_type="concat",
            aggregation="mean"
        )
    ),

    # ResNet34 with ImageNet pretrain
    "mm_resnet34_pretrained": MultiModalExperimentConfig(
        exp_name="mm_resnet34_pretrained",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean"
        )
    ),
}


def get_multimodal_experiment_config(name: str) -> MultiModalExperimentConfig:
    """Get multimodal experiment configuration by name."""
    if name not in MULTIMODAL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(MULTIMODAL_EXPERIMENTS.keys())}")
    return MULTIMODAL_EXPERIMENTS[name]


def print_multimodal_config(exp_config: MultiModalExperimentConfig):
    """Print multimodal experiment configuration."""
    mc = exp_config.model_config
    tc = exp_config.train_config

    print("=" * 60)
    print(f"Multimodal Experiment: {exp_config.exp_name}")
    print("=" * 60)
    print(f"  Image Encoder: {mc.image_encoder_type}")

    if mc.image_encoder_type == "light_cnn":
        if mc.light_cnn_channels:
            print(f"    Channels: {mc.light_cnn_channels}")
        else:
            print(f"    Channels: default [32, 64, 128]")
    elif mc.image_encoder_type == "resnet":
        print(f"    Model: {mc.resnet_model_name}")
        print(f"    Pretrained: {mc.use_pretrained_resnet}")

    print(f"  Image Feature Dim: {mc.image_feature_dim}")
    print(f"  Policy Feature Dim: {mc.policy_feature_dim} (encoder={mc.use_policy_encoder})")
    print(f"  Fusion Type: {mc.fusion_type}")
    print(f"  Training Mode: {tc.training_mode}")

    if tc.training_mode == "patch_level":
        print(f"  Patch Aggregation: {tc.patch_level_aggregation}")
    else:
        print(f"  Aggregation: {mc.aggregation}")

    print(f"  Batch Size: {tc.batch_size}")
    print(f"  Learning Rate: {tc.learning_rate}")
    print(f"  Epochs: {tc.num_epochs}")
    print("=" * 60)


if __name__ == "__main__":
    print("Available multimodal experiments:")
    for name in MULTIMODAL_EXPERIMENTS:
        print(f"  - {name}")

    print("\n" + "=" * 60)
    print("Example configuration:")
    print("=" * 60)
    exp = get_multimodal_experiment_config("mm_cnn_concat")
    print_multimodal_config(exp)
