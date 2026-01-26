"""
Configuration for pretrain experiments
Comparing different pretraining strategies for population growth prediction
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# =============================================================================
# Path Configuration
# =============================================================================
# BASE_DIR = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data")
BASE_DIR = Path("/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data")
DATA_DIR = BASE_DIR / "data_local"  # Local disk for faster I/O (was: "data" on NAS)
PROJECT_DIR = BASE_DIR / "Baseline_Pretrain"

# Input data paths
SATELLITE_DIR = DATA_DIR / "city_satellite_tiles"  # Original TIFF files
PATCHES_DIR = DATA_DIR / "city_patches"  # Preprocessed patches (all 25 patches in one npy)
INDIVIDUAL_PATCHES_DIR = DATA_DIR / "city_individual_patches"  # Individual patch files
# POPULATION_DATA = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/人口数据/人口自然增长率_2018-2024_filtered-empty.xlsx")
POPULATION_DATA = Path("/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/人口数据/人口自然增长率_2018-2024_filtered-empty.xlsx")

# Use preprocessed patches for faster loading
USE_PREPROCESSED_PATCHES = True
USE_INDIVIDUAL_PATCHES = True  # Use individual patch files (fastest for patch-level training)

# Output paths
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"
RESULT_DIR = PROJECT_DIR / "results"

# Create directories
for d in [CHECKPOINT_DIR, LOG_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data Configuration
# =============================================================================
NUM_BANDS = 64              # Alpha Earth embedding dimensions
FULL_SIZE = 1000            # Full tile size (1000x1000 pixels = 10km x 10km)
PIXEL_SIZE = 10             # 10 meters per pixel

# Patch extraction settings
PATCH_SIZE_KM = 2           # 2km x 2km patches
PATCH_SIZE_PIXELS = PATCH_SIZE_KM * 1000 // PIXEL_SIZE  # 200 pixels
NUM_PATCHES_PER_DIM = FULL_SIZE // PATCH_SIZE_PIXELS    # 5 patches per dimension
NUM_PATCHES_TOTAL = NUM_PATCHES_PER_DIM ** 2            # 25 patches total

# Years to consider
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Data split
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20
RANDOM_SEED = 42

# =============================================================================
# Model Configurations
# =============================================================================

@dataclass
class MLPConfig:
    """MLP model configuration"""
    name: str = "mlp"
    input_dim: int = NUM_BANDS  # 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    # Aggregation: mean, trimmed_mean, attention, pos_attention, spatial_attention, transformer, transformer_2d
    aggregation: str = "mean"


@dataclass
class LightCNNConfig:
    """Lightweight CNN configuration"""
    name: str = "light_cnn"
    input_channels: int = NUM_BANDS  # 64
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    fc_dims: List[int] = field(default_factory=lambda: [256, 64])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    # Aggregation: mean, trimmed_mean, attention, pos_attention, spatial_attention, transformer, transformer_2d
    aggregation: str = "mean"


@dataclass
class ResNetConfig:
    """ResNet baseline configuration"""
    name: str = "resnet"
    model_name: str = "resnet18"
    input_channels: int = NUM_BANDS
    hidden_dim: int = 512
    dropout_rate: float = 0.3
    use_pretrained: bool = False
    # Aggregation: mean, trimmed_mean, attention, pos_attention, spatial_attention, transformer, transformer_2d
    aggregation: str = "mean"


@dataclass
class SimCLRConfig:
    """SimCLR self-supervised pretraining configuration"""
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
    """Masked Autoencoder configuration"""
    name: str = "mae"
    encoder_type: str = "light_cnn"
    mask_ratio: float = 0.75  # Mask 75% of patches
    decoder_dim: int = 256
    decoder_depth: int = 2


@dataclass
class TrainConfig:
    """Training configuration"""
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 60  # Increased for cosine_warm_restarts scheduler

    # Scheduler
    scheduler: str = "cosine_warm_restarts"
    t_max: int = 10
    t_mult: int = 2
    eta_min: float = 1e-7

    # Pretrain specific
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    freeze_encoder_epochs: int = 5  # Freeze encoder for first N epochs during finetune

    # Training mode: "city_level" or "patch_level"
    # city_level: 25 patches as one sample -> model aggregates internally -> 1 prediction
    # patch_level: each patch is independent sample -> aggregate predictions at inference
    training_mode: str = "city_level"

    # Patch-level specific settings (for inference aggregation)
    patch_level_aggregation: str = "trimmed_mean"  # mean, median, trimmed_mean
    patch_level_trim_ratio: float = 0.1  # For trimmed_mean


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    exp_name: str = "default"
    model_config: object = None
    train_config: TrainConfig = field(default_factory=TrainConfig)
    pretrain_config: object = None  # SimCLR or MAE config
    use_pretrain: bool = False
    device: str = "cuda"
    num_workers: int = 4  # Optimal for local disk I/O
    wandb_enabled: bool = True  # Enabled for experiment tracking
    wandb_project: str = "population-pretrain-comparison-a100"


# =============================================================================
# Experiment Presets
# =============================================================================

EXPERIMENTS = {
    # ==========================================================================
    # City-level training (original method: 25 patches -> 1 prediction)
    # ==========================================================================
    # Baseline models (no pretraining) - with 3 basic aggregation methods
    # --- MLP with basic aggregations ---
    "mlp_baseline": ExperimentConfig(
        exp_name="mlp_baseline",
        model_config=MLPConfig(aggregation="mean"),
        use_pretrain=False
    ),
    "mlp_median": ExperimentConfig(
        exp_name="mlp_median",
        model_config=MLPConfig(aggregation="median"),
        use_pretrain=False
    ),
    "mlp_trimmed_mean": ExperimentConfig(
        exp_name="mlp_trimmed_mean",
        model_config=MLPConfig(aggregation="trimmed_mean"),
        use_pretrain=False
    ),
    # --- LightCNN with basic aggregations ---
    "light_cnn_baseline": ExperimentConfig(
        exp_name="light_cnn_baseline",
        model_config=LightCNNConfig(aggregation="mean"),
        use_pretrain=False
    ),
    "light_cnn_median": ExperimentConfig(
        exp_name="light_cnn_median",
        model_config=LightCNNConfig(aggregation="median"),
        use_pretrain=False
    ),
    "light_cnn_trimmed_mean": ExperimentConfig(
        exp_name="light_cnn_trimmed_mean",
        model_config=LightCNNConfig(aggregation="trimmed_mean"),
        use_pretrain=False
    ),
    # Self-supervised pretraining with 3 basic aggregation methods
    # --- SimCLR + MLP with basic aggregations ---
    "simclr_mlp": ExperimentConfig(
        exp_name="simclr_mlp",
        model_config=MLPConfig(aggregation="mean"),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    "simclr_mlp_median": ExperimentConfig(
        exp_name="simclr_mlp_median",
        model_config=MLPConfig(aggregation="median"),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    "simclr_mlp_trimmed_mean": ExperimentConfig(
        exp_name="simclr_mlp_trimmed_mean",
        model_config=MLPConfig(aggregation="trimmed_mean"),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    # --- SimCLR + CNN with basic aggregations ---
    "simclr_cnn": ExperimentConfig(
        exp_name="simclr_cnn",
        model_config=LightCNNConfig(aggregation="mean"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "simclr_cnn_median": ExperimentConfig(
        exp_name="simclr_cnn_median",
        model_config=LightCNNConfig(aggregation="median"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "simclr_cnn_trimmed_mean": ExperimentConfig(
        exp_name="simclr_cnn_trimmed_mean",
        model_config=LightCNNConfig(aggregation="trimmed_mean"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    # --- MAE + CNN with basic aggregations ---
    "mae_cnn": ExperimentConfig(
        exp_name="mae_cnn",
        model_config=LightCNNConfig(aggregation="mean"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn_median": ExperimentConfig(
        exp_name="mae_cnn_median",
        model_config=LightCNNConfig(aggregation="median"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn_trimmed_mean": ExperimentConfig(
        exp_name="mae_cnn_trimmed_mean",
        model_config=LightCNNConfig(aggregation="trimmed_mean"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # ==========================================================================
    # Patch-level training (paper method: each patch is independent sample)
    # Training: each patch predicts city's growth rate independently
    # Inference: aggregate all patch predictions using trimmed_mean
    # ==========================================================================
    "mlp_patch_level": ExperimentConfig(
        exp_name="mlp_patch_level",
        model_config=MLPConfig(),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=64,  # Can use larger batch since each sample is smaller
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),
    "light_cnn_patch_level": ExperimentConfig(
        exp_name="light_cnn_patch_level",
        model_config=LightCNNConfig(),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),
    # Patch-level with SimCLR pretraining
    # NOTE: batch_size=8 and num_workers=2 for pretrain phase (city-level data ~512MB/sample)
    # After pretrain, patch-level finetune uses same batch_size (smaller but works fine)
    "simclr_cnn_patch_level": ExperimentConfig(
        exp_name="simclr_cnn_patch_level",
        model_config=LightCNNConfig(),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=8,  # Reduced for SimCLR pretrain (city-level data is large)
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2  # Reduced to avoid OOM during pretrain
    ),

    # ==========================================================================
    # ResNet Scaling Experiments (comparing different model sizes)
    # Models: resnet10 (~5M) < resnet18 (~11M) < resnet34 (~21M) < resnet50 (~25M) < resnet101 (~44M)
    # ==========================================================================

    # --- ResNet10 (lightest, custom implementation) ---
    "resnet10_baseline": ExperimentConfig(
        exp_name="resnet10_baseline",
        model_config=ResNetConfig(model_name="resnet10", aggregation="mean", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet10_median": ExperimentConfig(
        exp_name="resnet10_median",
        model_config=ResNetConfig(model_name="resnet10", aggregation="median", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet10_trimmed_mean": ExperimentConfig(
        exp_name="resnet10_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet10", aggregation="trimmed_mean", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet10_patch_level": ExperimentConfig(
        exp_name="resnet10_patch_level",
        model_config=ResNetConfig(model_name="resnet10", use_pretrained=False),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=64,  # Smaller model, can use larger batch
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),

    # --- ResNet18 ---
    "resnet18_baseline": ExperimentConfig(
        exp_name="resnet18_baseline",
        model_config=ResNetConfig(model_name="resnet18", aggregation="mean", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet18_median": ExperimentConfig(
        exp_name="resnet18_median",
        model_config=ResNetConfig(model_name="resnet18", aggregation="median", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet18_trimmed_mean": ExperimentConfig(
        exp_name="resnet18_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet18", aggregation="trimmed_mean", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet18_imagenet": ExperimentConfig(
        exp_name="resnet18_imagenet",
        model_config=ResNetConfig(model_name="resnet18", aggregation="mean", use_pretrained=True),
        use_pretrain=False
    ),
    "resnet18_imagenet_median": ExperimentConfig(
        exp_name="resnet18_imagenet_median",
        model_config=ResNetConfig(model_name="resnet18", aggregation="median", use_pretrained=True),
        use_pretrain=False
    ),
    "resnet18_imagenet_trimmed_mean": ExperimentConfig(
        exp_name="resnet18_imagenet_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet18", aggregation="trimmed_mean", use_pretrained=True),
        use_pretrain=False
    ),
    "resnet18_patch_level": ExperimentConfig(
        exp_name="resnet18_patch_level",
        model_config=ResNetConfig(model_name="resnet18", use_pretrained=False),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),
    "resnet18_imagenet_patch_level": ExperimentConfig(
        exp_name="resnet18_imagenet_patch_level",
        model_config=ResNetConfig(model_name="resnet18", use_pretrained=True),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),

    # --- ResNet34 ---
    "resnet34_baseline": ExperimentConfig(
        exp_name="resnet34_baseline",
        model_config=ResNetConfig(model_name="resnet34", aggregation="mean", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet34_median": ExperimentConfig(
        exp_name="resnet34_median",
        model_config=ResNetConfig(model_name="resnet34", aggregation="median", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet34_trimmed_mean": ExperimentConfig(
        exp_name="resnet34_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet34", aggregation="trimmed_mean", use_pretrained=False),
        use_pretrain=False
    ),
    "resnet34_imagenet": ExperimentConfig(
        exp_name="resnet34_imagenet",
        model_config=ResNetConfig(model_name="resnet34", aggregation="mean", use_pretrained=True),
        use_pretrain=False
    ),
    "resnet34_imagenet_median": ExperimentConfig(
        exp_name="resnet34_imagenet_median",
        model_config=ResNetConfig(model_name="resnet34", aggregation="median", use_pretrained=True),
        use_pretrain=False
    ),
    "resnet34_imagenet_trimmed_mean": ExperimentConfig(
        exp_name="resnet34_imagenet_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet34", aggregation="trimmed_mean", use_pretrained=True),
        use_pretrain=False
    ),
    "resnet34_patch_level": ExperimentConfig(
        exp_name="resnet34_patch_level",
        model_config=ResNetConfig(model_name="resnet34", use_pretrained=False),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),
    "resnet34_imagenet_patch_level": ExperimentConfig(
        exp_name="resnet34_imagenet_patch_level",
        model_config=ResNetConfig(model_name="resnet34", use_pretrained=True),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),

    # --- ResNet50 (bottleneck blocks, larger feature dim: 2048) ---
    "resnet50_baseline": ExperimentConfig(
        exp_name="resnet50_baseline",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, aggregation="mean", use_pretrained=False),
        train_config=TrainConfig(batch_size=8),  # Reduced batch size for larger model
        use_pretrain=False
    ),
    "resnet50_median": ExperimentConfig(
        exp_name="resnet50_median",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, aggregation="median", use_pretrained=False),
        train_config=TrainConfig(batch_size=8),
        use_pretrain=False
    ),
    "resnet50_trimmed_mean": ExperimentConfig(
        exp_name="resnet50_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, aggregation="trimmed_mean", use_pretrained=False),
        train_config=TrainConfig(batch_size=8),
        use_pretrain=False
    ),
    "resnet50_imagenet": ExperimentConfig(
        exp_name="resnet50_imagenet",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, aggregation="mean", use_pretrained=True),
        train_config=TrainConfig(batch_size=8),
        use_pretrain=False
    ),
    "resnet50_imagenet_median": ExperimentConfig(
        exp_name="resnet50_imagenet_median",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, aggregation="median", use_pretrained=True),
        train_config=TrainConfig(batch_size=8),
        use_pretrain=False
    ),
    "resnet50_imagenet_trimmed_mean": ExperimentConfig(
        exp_name="resnet50_imagenet_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, aggregation="trimmed_mean", use_pretrained=True),
        train_config=TrainConfig(batch_size=8),
        use_pretrain=False
    ),
    "resnet50_patch_level": ExperimentConfig(
        exp_name="resnet50_patch_level",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, use_pretrained=False),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=16,  # Reduced for larger model
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),
    "resnet50_imagenet_patch_level": ExperimentConfig(
        exp_name="resnet50_imagenet_patch_level",
        model_config=ResNetConfig(model_name="resnet50", hidden_dim=1024, use_pretrained=True),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=16,  # Reduced for larger model
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),

    # --- ResNet101 (deepest, bottleneck blocks) ---
    "resnet101_baseline": ExperimentConfig(
        exp_name="resnet101_baseline",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, aggregation="mean", use_pretrained=False),
        train_config=TrainConfig(batch_size=4),  # Small batch for largest model
        use_pretrain=False
    ),
    "resnet101_median": ExperimentConfig(
        exp_name="resnet101_median",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, aggregation="median", use_pretrained=False),
        train_config=TrainConfig(batch_size=4),
        use_pretrain=False
    ),
    "resnet101_trimmed_mean": ExperimentConfig(
        exp_name="resnet101_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, aggregation="trimmed_mean", use_pretrained=False),
        train_config=TrainConfig(batch_size=4),
        use_pretrain=False
    ),
    "resnet101_imagenet": ExperimentConfig(
        exp_name="resnet101_imagenet",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, aggregation="mean", use_pretrained=True),
        train_config=TrainConfig(batch_size=4),
        use_pretrain=False
    ),
    "resnet101_imagenet_median": ExperimentConfig(
        exp_name="resnet101_imagenet_median",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, aggregation="median", use_pretrained=True),
        train_config=TrainConfig(batch_size=4),
        use_pretrain=False
    ),
    "resnet101_imagenet_trimmed_mean": ExperimentConfig(
        exp_name="resnet101_imagenet_trimmed_mean",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, aggregation="trimmed_mean", use_pretrained=True),
        train_config=TrainConfig(batch_size=4),
        use_pretrain=False
    ),
    "resnet101_patch_level": ExperimentConfig(
        exp_name="resnet101_patch_level",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, use_pretrained=False),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=8,  # Reduced for largest model
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),
    "resnet101_imagenet_patch_level": ExperimentConfig(
        exp_name="resnet101_imagenet_patch_level",
        model_config=ResNetConfig(model_name="resnet101", hidden_dim=1024, use_pretrained=True),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=8,  # Reduced for largest model
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        use_pretrain=False
    ),

    # ==========================================================================
    # Position-Aware Aggregation Experiments (City-level)
    # Compare different aggregation strategies that utilize spatial position info
    # ==========================================================================

    # --- MLP with different aggregation methods ---
    "mlp_attention": ExperimentConfig(
        exp_name="mlp_attention",
        model_config=MLPConfig(aggregation="attention"),
        use_pretrain=False
    ),
    "mlp_pos_attention": ExperimentConfig(
        exp_name="mlp_pos_attention",
        model_config=MLPConfig(aggregation="pos_attention"),
        use_pretrain=False
    ),
    "mlp_spatial_attention": ExperimentConfig(
        exp_name="mlp_spatial_attention",
        model_config=MLPConfig(aggregation="spatial_attention"),
        use_pretrain=False
    ),
    "mlp_transformer": ExperimentConfig(
        exp_name="mlp_transformer",
        model_config=MLPConfig(aggregation="transformer"),
        use_pretrain=False
    ),
    "mlp_transformer_2d": ExperimentConfig(
        exp_name="mlp_transformer_2d",
        model_config=MLPConfig(aggregation="transformer_2d"),
        use_pretrain=False
    ),

    # --- LightCNN with different aggregation methods ---
    "light_cnn_attention": ExperimentConfig(
        exp_name="light_cnn_attention",
        model_config=LightCNNConfig(aggregation="attention"),
        use_pretrain=False
    ),
    "light_cnn_pos_attention": ExperimentConfig(
        exp_name="light_cnn_pos_attention",
        model_config=LightCNNConfig(aggregation="pos_attention"),
        use_pretrain=False
    ),
    "light_cnn_spatial_attention": ExperimentConfig(
        exp_name="light_cnn_spatial_attention",
        model_config=LightCNNConfig(aggregation="spatial_attention"),
        use_pretrain=False
    ),
    "light_cnn_transformer": ExperimentConfig(
        exp_name="light_cnn_transformer",
        model_config=LightCNNConfig(aggregation="transformer"),
        use_pretrain=False
    ),
    "light_cnn_transformer_2d": ExperimentConfig(
        exp_name="light_cnn_transformer_2d",
        model_config=LightCNNConfig(aggregation="transformer_2d"),
        use_pretrain=False
    ),

    # --- ResNet18 with different aggregation methods ---
    "resnet18_attention": ExperimentConfig(
        exp_name="resnet18_attention",
        model_config=ResNetConfig(model_name="resnet18", aggregation="attention"),
        use_pretrain=False
    ),
    "resnet18_pos_attention": ExperimentConfig(
        exp_name="resnet18_pos_attention",
        model_config=ResNetConfig(model_name="resnet18", aggregation="pos_attention"),
        use_pretrain=False
    ),
    "resnet18_spatial_attention": ExperimentConfig(
        exp_name="resnet18_spatial_attention",
        model_config=ResNetConfig(model_name="resnet18", aggregation="spatial_attention"),
        use_pretrain=False
    ),
    "resnet18_transformer": ExperimentConfig(
        exp_name="resnet18_transformer",
        model_config=ResNetConfig(model_name="resnet18", aggregation="transformer"),
        use_pretrain=False
    ),
    "resnet18_transformer_2d": ExperimentConfig(
        exp_name="resnet18_transformer_2d",
        model_config=ResNetConfig(model_name="resnet18", aggregation="transformer_2d"),
        use_pretrain=False
    ),

    # --- SimCLR pretraining + different aggregation methods ---
    "simclr_cnn_attention": ExperimentConfig(
        exp_name="simclr_cnn_attention",
        model_config=LightCNNConfig(aggregation="attention"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "simclr_cnn_pos_attention": ExperimentConfig(
        exp_name="simclr_cnn_pos_attention",
        model_config=LightCNNConfig(aggregation="pos_attention"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "simclr_cnn_spatial_attention": ExperimentConfig(
        exp_name="simclr_cnn_spatial_attention",
        model_config=LightCNNConfig(aggregation="spatial_attention"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "simclr_cnn_transformer": ExperimentConfig(
        exp_name="simclr_cnn_transformer",
        model_config=LightCNNConfig(aggregation="transformer"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "simclr_cnn_transformer_2d": ExperimentConfig(
        exp_name="simclr_cnn_transformer_2d",
        model_config=LightCNNConfig(aggregation="transformer_2d"),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),

    # --- MAE pretraining + different aggregation methods ---
    "mae_cnn_patch_level": ExperimentConfig(
        exp_name="mae_cnn_patch_level",
        model_config=LightCNNConfig(),
        train_config=TrainConfig(
            training_mode="patch_level",
            batch_size=8,  # Reduced for MAE pretrain (city-level data is large)
            patch_level_aggregation="trimmed_mean",
            patch_level_trim_ratio=0.1
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2  # Reduced to avoid OOM during pretrain
    ),
    "mae_cnn_attention": ExperimentConfig(
        exp_name="mae_cnn_attention",
        model_config=LightCNNConfig(aggregation="attention"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn_pos_attention": ExperimentConfig(
        exp_name="mae_cnn_pos_attention",
        model_config=LightCNNConfig(aggregation="pos_attention"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn_spatial_attention": ExperimentConfig(
        exp_name="mae_cnn_spatial_attention",
        model_config=LightCNNConfig(aggregation="spatial_attention"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn_transformer": ExperimentConfig(
        exp_name="mae_cnn_transformer",
        model_config=LightCNNConfig(aggregation="transformer"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn_transformer_2d": ExperimentConfig(
        exp_name="mae_cnn_transformer_2d",
        model_config=LightCNNConfig(aggregation="transformer_2d"),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
}


def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name"""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def print_config(config: ExperimentConfig):
    """Print experiment configuration"""
    print("=" * 60)
    print(f"Experiment: {config.exp_name}")
    print("=" * 60)
    print(f"  Model: {config.model_config.name}")
    print(f"  Training Mode: {config.train_config.training_mode}")
    if config.train_config.training_mode == "patch_level":
        print(f"  Patch Aggregation: {config.train_config.patch_level_aggregation}")
        print(f"  Trim Ratio: {config.train_config.patch_level_trim_ratio}")
    else:
        # City-level mode - show model aggregation
        print(f"  Aggregation: {config.model_config.aggregation}")
    print(f"  Use Pretrain: {config.use_pretrain}")
    if config.use_pretrain and config.pretrain_config:
        print(f"  Pretrain Method: {config.pretrain_config.name}")
    print(f"  Batch Size: {config.train_config.batch_size}")
    print(f"  Learning Rate: {config.train_config.learning_rate}")
    print(f"  Epochs: {config.train_config.num_epochs}")
    print("=" * 60)


if __name__ == "__main__":
    # Print all available experiments
    print("Available experiments:")
    for name in EXPERIMENTS:
        print(f"  - {name}")
