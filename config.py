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
BASE_DIR = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data")
DATA_DIR = BASE_DIR / "data"
PROJECT_DIR = BASE_DIR / "Baseline_Pretrain"

# Input data paths
SATELLITE_DIR = DATA_DIR / "city_satellite_tiles"
POPULATION_DATA = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/人口数据/人口自然增长率_2018-2024_filtered-empty.xlsx")

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
    aggregation: str = "mean"  # mean, attention, trimmed_mean


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
    patience: int = 50

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


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    exp_name: str = "default"
    model_config: object = None
    train_config: TrainConfig = field(default_factory=TrainConfig)
    pretrain_config: object = None  # SimCLR or MAE config
    use_pretrain: bool = False
    device: str = "cuda"
    num_workers: int = 4
    wandb_enabled: bool = True
    wandb_project: str = "population-pretrain-comparison"


# =============================================================================
# Experiment Presets
# =============================================================================

EXPERIMENTS = {
    # Baseline models (no pretraining)
    "mlp_baseline": ExperimentConfig(
        exp_name="mlp_baseline",
        model_config=MLPConfig(),
        use_pretrain=False
    ),
    "light_cnn_baseline": ExperimentConfig(
        exp_name="light_cnn_baseline",
        model_config=LightCNNConfig(),
        use_pretrain=False
    ),
    "resnet_baseline": ExperimentConfig(
        exp_name="resnet_baseline",
        model_config=ResNetConfig(use_pretrained=False),
        use_pretrain=False
    ),
    "resnet_imagenet": ExperimentConfig(
        exp_name="resnet_imagenet",
        model_config=ResNetConfig(use_pretrained=True),
        use_pretrain=False
    ),

    # Self-supervised pretraining
    "simclr_mlp": ExperimentConfig(
        exp_name="simclr_mlp",
        model_config=MLPConfig(),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    "simclr_cnn": ExperimentConfig(
        exp_name="simclr_cnn",
        model_config=LightCNNConfig(),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mae_cnn": ExperimentConfig(
        exp_name="mae_cnn",
        model_config=LightCNNConfig(),
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
