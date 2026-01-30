"""
Configuration for multimodal experiments (Image + Policy features)

This extends the base config with multimodal-specific configurations.

Extended to support:
1. Self-supervised pretraining (SimCLR, MAE) for image encoder
2. Position-Aware Aggregation (attention, transformer, etc.)
3. All ResNet depths (10, 18, 34, 50, 101)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import config


# =============================================================================
# Pretrain Configurations (same as single-modal)
# =============================================================================

@dataclass
class SimCLRConfig:
    """SimCLR self-supervised pretraining configuration."""
    name: str = "simclr"
    encoder_type: str = "light_cnn"  # mlp, light_cnn
    projection_dim: int = 128
    temperature: float = 0.5
    # Augmentation
    use_flip: bool = True
    use_rotation: bool = True
    use_noise: bool = True
    noise_std: float = 0.1


@dataclass
class MAEConfig:
    """Masked Autoencoder configuration."""
    name: str = "mae"
    encoder_type: str = "light_cnn"
    mask_ratio: float = 0.75  # Mask 75% of patches
    decoder_dim: int = 256
    decoder_depth: int = 2


# =============================================================================
# Multimodal Model Configuration
# =============================================================================

@dataclass
class MultiModalConfig:
    """Configuration for multimodal model."""
    name: str = "multimodal"

    # Image encoder type
    image_encoder_type: str = "light_cnn"  # mlp, light_cnn, resnet
    image_feature_dim: int = 64  # Project image features to this dim (default 64)

    # MLP specific (when image_encoder_type == "mlp")
    # IMPORTANT: For SimCLR pretrain compatibility, use [256, 128] to match pretrain structure
    mlp_hidden_dims: list = None  # e.g., [256, 128, 64], None = use default [256, 128, 64]

    # LightCNN specific (when image_encoder_type == "light_cnn")
    light_cnn_channels: list = None  # e.g., [32, 64, 128], None = use default
    light_cnn_kernels: list = None   # e.g., [3, 3, 3], None = use default
    use_batch_norm: bool = True

    # ResNet specific (when image_encoder_type == "resnet")
    resnet_model_name: str = "resnet18"  # resnet10, resnet18, resnet34, resnet50, resnet101
    use_pretrained_resnet: bool = False  # Use ImageNet pretrained weights

    # Policy features
    policy_feature_dim: int = 12  # Raw policy feature dim
    policy_hidden_dim: int = 64   # Hidden dim if using policy encoder
    use_policy_encoder: bool = False  # Default: use raw 12-dim features

    # Fusion strategy
    fusion_type: str = "concat"  # concat, gated, attention, film

    # Aggregation (for city-level)
    # Options: mean, median, trimmed_mean, attention, pos_attention,
    #          spatial_attention, transformer, transformer_2d
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

    # Pretrain specific (for SimCLR/MAE pretraining)
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    freeze_encoder_epochs: int = 5  # Freeze encoder for first N epochs during finetune


@dataclass
class MultiModalExperimentConfig:
    """Full experiment configuration for multimodal training."""
    exp_name: str = "multimodal_default"
    model_config: MultiModalConfig = field(default_factory=MultiModalConfig)
    train_config: MultiModalTrainConfig = field(default_factory=MultiModalTrainConfig)
    pretrain_config: object = None  # SimCLRConfig or MAEConfig, None = no pretraining
    use_pretrain: bool = False  # Whether to use self-supervised pretraining
    device: str = "cuda"
    num_workers: int = 8
    wandb_enabled: bool = True
    wandb_project: str = "population-multimodal-a100"


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

    # ==========================================================================
    # NEW: Position-Aware Aggregation Experiments
    # ==========================================================================

    # --- LightCNN + Attention Aggregation ---
    "mm_cnn_concat_attn_agg": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_attn_agg",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="attention"
        )
    ),
    "mm_cnn_concat_pos_attn": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_pos_attn",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="pos_attention"
        )
    ),
    "mm_cnn_concat_spatial_attn": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_spatial_attn",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="spatial_attention"
        )
    ),
    "mm_cnn_concat_transformer": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_transformer",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer"
        )
    ),
    "mm_cnn_concat_transformer_2d": MultiModalExperimentConfig(
        exp_name="mm_cnn_concat_transformer_2d",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d"
        )
    ),

    # --- LightCNN + Gated + Position-Aware Aggregation ---
    "mm_cnn_gated_transformer": MultiModalExperimentConfig(
        exp_name="mm_cnn_gated_transformer",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer"
        )
    ),
    "mm_cnn_gated_transformer_2d": MultiModalExperimentConfig(
        exp_name="mm_cnn_gated_transformer_2d",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer_2d"
        )
    ),

    # --- LightCNN + FiLM + Position-Aware Aggregation ---
    "mm_cnn_film_transformer": MultiModalExperimentConfig(
        exp_name="mm_cnn_film_transformer",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer"
        )
    ),
    "mm_cnn_film_transformer_2d": MultiModalExperimentConfig(
        exp_name="mm_cnn_film_transformer_2d",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer_2d"
        )
    ),

    # --- ResNet18 + Position-Aware Aggregation ---
    "mm_resnet18_concat_transformer": MultiModalExperimentConfig(
        exp_name="mm_resnet18_concat_transformer",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer"
        )
    ),
    "mm_resnet18_concat_transformer_2d": MultiModalExperimentConfig(
        exp_name="mm_resnet18_concat_transformer_2d",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer_2d"
        )
    ),

    # ==========================================================================
    # NEW: Self-Supervised Pretraining Experiments (SimCLR)
    # ==========================================================================

    # --- SimCLR + LightCNN + Concat ---
    "mm_simclr_cnn_concat": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_concat",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_concat_trimmed": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_concat_trimmed",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- SimCLR + LightCNN + Gated ---
    "mm_simclr_cnn_gated": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_gated",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- SimCLR + LightCNN + FiLM ---
    "mm_simclr_cnn_film": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_film",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- SimCLR + LightCNN + Position-Aware Aggregation ---
    "mm_simclr_cnn_transformer": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_transformer",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_transformer_2d": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_transformer_2d",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- SimCLR + MLP ---
    # IMPORTANT: mlp_hidden_dims must match SimCLR pretrain structure [256, 128]
    "mm_simclr_mlp_concat": MultiModalExperimentConfig(
        exp_name="mm_simclr_mlp_concat",
        model_config=MultiModalConfig(
            image_encoder_type="mlp",
            mlp_hidden_dims=[256, 128],  # Must match SimCLR pretrain (simclr.py:65)
            fusion_type="concat",
            aggregation="mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),

    # ==========================================================================
    # NEW: Self-Supervised Pretraining Experiments (MAE)
    # ==========================================================================

    # --- MAE + LightCNN + Concat ---
    "mm_mae_cnn_concat": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_concat",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_concat_trimmed": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_concat_trimmed",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- MAE + LightCNN + Gated ---
    "mm_mae_cnn_gated": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_gated",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- MAE + LightCNN + FiLM ---
    "mm_mae_cnn_film": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_film",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- MAE + LightCNN + Position-Aware Aggregation ---
    "mm_mae_cnn_transformer": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_transformer",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_transformer_2d": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_transformer_2d",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # ==========================================================================
    # NEW: Additional ResNet Depths
    # ==========================================================================

    # --- ResNet10 (lightest) ---
    "mm_resnet10_concat": MultiModalExperimentConfig(
        exp_name="mm_resnet10_concat",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="concat",
            aggregation="mean"
        )
    ),
    "mm_resnet10_gated": MultiModalExperimentConfig(
        exp_name="mm_resnet10_gated",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="gated",
            aggregation="mean"
        )
    ),

    # --- ResNet34 ---
    "mm_resnet34_concat": MultiModalExperimentConfig(
        exp_name="mm_resnet34_concat",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="concat",
            aggregation="mean"
        )
    ),
    "mm_resnet34_gated": MultiModalExperimentConfig(
        exp_name="mm_resnet34_gated",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="gated",
            aggregation="mean"
        )
    ),
    "mm_resnet34_film": MultiModalExperimentConfig(
        exp_name="mm_resnet34_film",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="film",
            aggregation="mean"
        )
    ),

    # --- ResNet50 (bottleneck) ---
    "mm_resnet50_concat": MultiModalExperimentConfig(
        exp_name="mm_resnet50_concat",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="concat",
            aggregation="mean"
        ),
        train_config=MultiModalTrainConfig(batch_size=8)  # Smaller batch for larger model
    ),
    "mm_resnet50_gated": MultiModalExperimentConfig(
        exp_name="mm_resnet50_gated",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="gated",
            aggregation="mean"
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "mm_resnet50_imagenet": MultiModalExperimentConfig(
        exp_name="mm_resnet50_imagenet",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean"
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),

    # --- ResNet101 (deepest) ---
    "mm_resnet101_concat": MultiModalExperimentConfig(
        exp_name="mm_resnet101_concat",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            fusion_type="concat",
            aggregation="mean"
        ),
        train_config=MultiModalTrainConfig(batch_size=4)  # Smallest batch for largest model
    ),
    "mm_resnet101_imagenet": MultiModalExperimentConfig(
        exp_name="mm_resnet101_imagenet",
        model_config=MultiModalConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean"
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),

    # ==========================================================================
    # NEW: Patch-level with Pretraining
    # ==========================================================================
    "mm_simclr_cnn_concat_patch": MultiModalExperimentConfig(
        exp_name="mm_simclr_cnn_concat_patch",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat"
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,  # Smaller for pretrain
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2  # Reduced for pretrain
    ),
    "mm_mae_cnn_concat_patch": MultiModalExperimentConfig(
        exp_name="mm_mae_cnn_concat_patch",
        model_config=MultiModalConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat"
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2
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
        print(f"    ImageNet Pretrained: {mc.use_pretrained_resnet}")

    print(f"  Image Feature Dim: {mc.image_feature_dim}")
    print(f"  Policy Feature Dim: {mc.policy_feature_dim} (encoder={mc.use_policy_encoder})")
    print(f"  Fusion Type: {mc.fusion_type}")
    print(f"  Training Mode: {tc.training_mode}")

    if tc.training_mode == "patch_level":
        print(f"  Patch Aggregation: {tc.patch_level_aggregation}")
    else:
        print(f"  Aggregation: {mc.aggregation}")

    # Show pretraining config
    if exp_config.use_pretrain and exp_config.pretrain_config:
        pc = exp_config.pretrain_config
        print(f"  Self-Supervised Pretrain: {pc.name}")
        if pc.name == "simclr":
            print(f"    Encoder Type: {pc.encoder_type}")
            print(f"    Projection Dim: {pc.projection_dim}")
            print(f"    Temperature: {pc.temperature}")
        elif pc.name == "mae":
            print(f"    Encoder Type: {pc.encoder_type}")
            print(f"    Mask Ratio: {pc.mask_ratio}")
            print(f"    Decoder Dim: {pc.decoder_dim}")
        print(f"    Pretrain Epochs: {tc.pretrain_epochs}")
        print(f"    Pretrain LR: {tc.pretrain_lr}")
    else:
        print(f"  Self-Supervised Pretrain: None")

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
