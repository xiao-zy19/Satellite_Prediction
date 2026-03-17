"""
Unified Configuration for Multimodal Experiments (Image + Policy features)

This is the unified configuration supporting THREE policy sources:
1. structured: Original 12-dim structured policy features
2. bert: BERT embeddings from cache (default 64-dim)
3. hybrid: BERT + structured concatenated (default 76-dim)

All original multimodal experiments are preserved with policy_source="structured".
New BERT and Hybrid experiments use policy_source="bert" or "hybrid".

Usage:
    python train_multimodal_bert.py --exp mm_cnn_concat       # structured (original)
    python train_multimodal_bert.py --exp bert_cnn_concat     # BERT
    python train_multimodal_bert.py --exp hybrid_cnn_concat   # Hybrid
"""

from dataclasses import dataclass, field
from typing import List, Optional
import config

# Import original configs for compatibility
from config_multimodal import (
    SimCLRConfig,
    MAEConfig,
    MultiModalTrainConfig,
)


# =============================================================================
# Policy Source Configuration
# =============================================================================

@dataclass
class PolicySourceConfig:
    """Configuration for policy feature source."""
    source: str = "structured"  # "structured", "bert", "hybrid"
    bert_cache_dir: str = "policy_bert_cache"
    bert_dim: int = 64  # BERT embedding dimension
    structured_dim: int = 12  # Structured feature dimension

    @property
    def policy_dim(self) -> int:
        """Compute total policy feature dimension."""
        if self.source == "structured":
            return self.structured_dim
        elif self.source == "bert":
            return self.bert_dim
        elif self.source == "hybrid":
            return self.bert_dim + self.structured_dim
        else:
            raise ValueError(f"Unknown policy source: {self.source}")


# =============================================================================
# Unified Multimodal Model Configuration
# =============================================================================

@dataclass
class MultiModalBertConfig:
    """Unified configuration for multimodal model (supports all policy sources)."""
    name: str = "multimodal_unified"

    # Image encoder type
    image_encoder_type: str = "light_cnn"  # mlp, light_cnn, resnet
    image_feature_dim: int = 64

    # MLP specific
    mlp_hidden_dims: list = None

    # LightCNN specific
    light_cnn_channels: list = None
    light_cnn_kernels: list = None
    use_batch_norm: bool = True

    # ResNet specific
    resnet_model_name: str = "resnet18"
    use_pretrained_resnet: bool = False

    # Policy features (unified: structured/bert/hybrid)
    policy_source: PolicySourceConfig = field(default_factory=lambda: PolicySourceConfig(source="structured"))

    # Policy encoder
    policy_hidden_dim: int = 64
    policy_output_dim: int = 64  # Output dim of policy encoder (when use_policy_encoder=True)
    use_policy_encoder: bool = False

    # Fusion strategy
    fusion_type: str = "concat"

    # Aggregation
    aggregation: str = "mean"

    # Common
    dropout_rate: float = 0.3

    @property
    def policy_feature_dim(self) -> int:
        return self.policy_source.policy_dim


@dataclass
class MultiModalBertExperimentConfig:
    """Full experiment configuration."""
    exp_name: str = "multimodal_default"
    model_config: MultiModalBertConfig = field(default_factory=MultiModalBertConfig)
    train_config: MultiModalTrainConfig = field(default_factory=MultiModalTrainConfig)
    pretrain_config: object = None
    use_pretrain: bool = False
    device: str = "cuda"
    num_workers: int = 8
    wandb_enabled: bool = True
    wandb_project: str = "population-multimodal-unified"


# =============================================================================
# Helper functions to create experiments
# =============================================================================

def _structured_config(source="structured"):
    return PolicySourceConfig(source=source)

def _bert_config():
    return PolicySourceConfig(source="bert", bert_dim=768)

def _hybrid_config():
    return PolicySourceConfig(source="hybrid", bert_dim=768)


# =============================================================================
# UNIFIED EXPERIMENT PRESETS
# =============================================================================

BERT_MULTIMODAL_EXPERIMENTS = {

    # =========================================================================
    # PART 1: STRUCTURED POLICY (12-dim) - Original Multimodal Experiments
    # =========================================================================

    # --- LightCNN + Concat Fusion (baseline) ---
    "mm_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_concat_median": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_median",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="median",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_structured_config()
        )
    ),

    # --- LightCNN + Gated Fusion ---
    "mm_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_gated_trimmed": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_gated_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="trimmed_mean",
            policy_source=_structured_config()
        )
    ),

    # --- LightCNN + Attention Fusion ---
    "mm_cnn_attention": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_attention",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_attention_trimmed": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_attention_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="trimmed_mean",
            policy_source=_structured_config()
        )
    ),

    # --- LightCNN + FiLM Fusion ---
    "mm_cnn_film": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_film_trimmed": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_film_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="trimmed_mean",
            policy_source=_structured_config()
        )
    ),

    # --- MLP + Different Fusions ---
    "mm_mlp_concat": MultiModalBertExperimentConfig(
        exp_name="mm_mlp_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_mlp_gated": MultiModalBertExperimentConfig(
        exp_name="mm_mlp_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),

    # --- Position-Aware Aggregation ---
    "mm_cnn_concat_attn_agg": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_attn_agg",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="attention",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_concat_pos_attn": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_pos_attn",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="pos_attention",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_concat_spatial_attn": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_spatial_attn",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="spatial_attention",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_concat_transformer": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_concat_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_structured_config()
        )
    ),

    # --- Gated + Position-Aware ---
    "mm_cnn_gated_transformer": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_gated_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_gated_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_gated_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer_2d",
            policy_source=_structured_config()
        )
    ),

    # --- FiLM + Position-Aware ---
    "mm_cnn_film_transformer": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_film_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer",
            policy_source=_structured_config()
        )
    ),
    "mm_cnn_film_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_film_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer_2d",
            policy_source=_structured_config()
        )
    ),

    # --- Custom LightCNN ---
    "mm_cnn_small_concat": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_small_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            light_cnn_channels=[16, 32, 64],
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),

    # --- ResNet Experiments ---
    "mm_resnet10_concat": MultiModalBertExperimentConfig(
        exp_name="mm_resnet10_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet10_gated": MultiModalBertExperimentConfig(
        exp_name="mm_resnet10_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet18_concat": MultiModalBertExperimentConfig(
        exp_name="mm_resnet18_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet18_gated": MultiModalBertExperimentConfig(
        exp_name="mm_resnet18_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet18_film": MultiModalBertExperimentConfig(
        exp_name="mm_resnet18_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="film",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet18_concat_transformer": MultiModalBertExperimentConfig(
        exp_name="mm_resnet18_concat_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet18_concat_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="mm_resnet18_concat_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet34_concat": MultiModalBertExperimentConfig(
        exp_name="mm_resnet34_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet34_gated": MultiModalBertExperimentConfig(
        exp_name="mm_resnet34_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet34_film": MultiModalBertExperimentConfig(
        exp_name="mm_resnet34_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="film",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet34_pretrained": MultiModalBertExperimentConfig(
        exp_name="mm_resnet34_pretrained",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        )
    ),
    "mm_resnet50_concat": MultiModalBertExperimentConfig(
        exp_name="mm_resnet50_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "mm_resnet50_gated": MultiModalBertExperimentConfig(
        exp_name="mm_resnet50_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "mm_resnet50_imagenet": MultiModalBertExperimentConfig(
        exp_name="mm_resnet50_imagenet",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "mm_resnet101_concat": MultiModalBertExperimentConfig(
        exp_name="mm_resnet101_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),
    "mm_resnet101_imagenet": MultiModalBertExperimentConfig(
        exp_name="mm_resnet101_imagenet",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),

    # --- Patch-level (Structured) ---
    "mm_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "mm_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "mm_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "mm_cnn_attention_patch": MultiModalBertExperimentConfig(
        exp_name="mm_cnn_attention_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),

    "mm_resnet18_concat_patch": MultiModalBertExperimentConfig(
        exp_name="mm_resnet18_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean"
        )
    ),

    # --- SimCLR Pretrain (Structured) ---
    "mm_simclr_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_film": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_simclr_mlp_concat": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_mlp_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            mlp_hidden_dims=[256, 128],
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    "mm_simclr_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2
    ),
    "mm_simclr_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2
    ),
    "mm_simclr_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="mm_simclr_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_structured_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2
    ),

    # --- MAE Pretrain (Structured) ---
    "mm_mae_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_structured_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_film": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_structured_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_structured_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_structured_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "mm_mae_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_structured_config()
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
    "mm_mae_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_structured_config()
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
    "mm_mae_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="mm_mae_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_structured_config()
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

    # =========================================================================
    # PART 2: BERT POLICY (64-dim) - Complete experiments matching structured
    # =========================================================================

    # --- LightCNN + Concat Fusion (baseline) ---
    "bert_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_concat_median": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat_median",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="median",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_bert_config()
        )
    ),

    # --- LightCNN + Gated Fusion ---
    "bert_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_gated_trimmed": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_gated_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="trimmed_mean",
            policy_source=_bert_config()
        )
    ),

    # --- LightCNN + Attention Fusion ---
    "bert_cnn_attention": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_attention",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_attention_trimmed": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_attention_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="trimmed_mean",
            policy_source=_bert_config()
        )
    ),

    # --- LightCNN + FiLM Fusion ---
    "bert_cnn_film": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_film_trimmed": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_film_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="trimmed_mean",
            policy_source=_bert_config()
        )
    ),

    # --- MLP + BERT ---
    "bert_mlp_concat": MultiModalBertExperimentConfig(
        exp_name="bert_mlp_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_mlp_gated": MultiModalBertExperimentConfig(
        exp_name="bert_mlp_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),

    # --- Position-Aware Aggregation (LightCNN + BERT) ---
    "bert_cnn_concat_attn_agg": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat_attn_agg",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="attention",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_concat_pos_attn": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat_pos_attn",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="pos_attention",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_concat_spatial_attn": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat_spatial_attn",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="spatial_attention",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_bert_config()
        )
    ),

    # --- Gated + Position-Aware (BERT) ---
    "bert_cnn_gated_transformer": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_gated_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_gated_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_gated_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer_2d",
            policy_source=_bert_config()
        )
    ),

    # --- FiLM + Position-Aware (BERT) ---
    "bert_cnn_film_transformer": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_film_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer",
            policy_source=_bert_config()
        )
    ),
    "bert_cnn_film_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_film_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer_2d",
            policy_source=_bert_config()
        )
    ),

    # --- Custom LightCNN (BERT) ---
    "bert_cnn_small_concat": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_small_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            light_cnn_channels=[16, 32, 64],
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),

    # --- ResNet10 + BERT ---
    "bert_resnet10_concat": MultiModalBertExperimentConfig(
        exp_name="bert_resnet10_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet10_gated": MultiModalBertExperimentConfig(
        exp_name="bert_resnet10_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),

    # --- ResNet18 + BERT ---
    "bert_resnet18_concat": MultiModalBertExperimentConfig(
        exp_name="bert_resnet18_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet18_gated": MultiModalBertExperimentConfig(
        exp_name="bert_resnet18_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet18_film": MultiModalBertExperimentConfig(
        exp_name="bert_resnet18_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="film",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet18_concat_transformer": MultiModalBertExperimentConfig(
        exp_name="bert_resnet18_concat_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet18_concat_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="bert_resnet18_concat_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_bert_config()
        )
    ),

    # --- ResNet34 + BERT ---
    "bert_resnet34_concat": MultiModalBertExperimentConfig(
        exp_name="bert_resnet34_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet34_gated": MultiModalBertExperimentConfig(
        exp_name="bert_resnet34_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet34_film": MultiModalBertExperimentConfig(
        exp_name="bert_resnet34_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="film",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),
    "bert_resnet34_pretrained": MultiModalBertExperimentConfig(
        exp_name="bert_resnet34_pretrained",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        )
    ),

    # --- ResNet50 + BERT ---
    "bert_resnet50_concat": MultiModalBertExperimentConfig(
        exp_name="bert_resnet50_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "bert_resnet50_gated": MultiModalBertExperimentConfig(
        exp_name="bert_resnet50_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "bert_resnet50_imagenet": MultiModalBertExperimentConfig(
        exp_name="bert_resnet50_imagenet",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),

    # --- ResNet101 + BERT ---
    "bert_resnet101_concat": MultiModalBertExperimentConfig(
        exp_name="bert_resnet101_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),
    "bert_resnet101_imagenet": MultiModalBertExperimentConfig(
        exp_name="bert_resnet101_imagenet",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),

    # --- Patch-level (BERT) ---
    "bert_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "bert_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "bert_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="bert_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "bert_resnet18_concat_patch": MultiModalBertExperimentConfig(
        exp_name="bert_resnet18_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean"
        )
    ),

    # --- SimCLR Pretrained (BERT) ---
    "bert_simclr_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_simclr_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_simclr_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_simclr_cnn_film": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_simclr_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_simclr_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_simclr_mlp_concat": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_mlp_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            mlp_hidden_dims=[256, 128],
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    "bert_simclr_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="bert_simclr_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_bert_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2
    ),

    # --- MAE Pretrained (BERT) ---
    "bert_mae_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_mae_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_bert_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_mae_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_mae_cnn_film": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_bert_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_mae_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_bert_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_mae_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_bert_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "bert_mae_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_bert_config()
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
    "bert_mae_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_bert_config()
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
    "bert_mae_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="bert_mae_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_bert_config()
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

    # =========================================================================
    # PART 3: HYBRID POLICY (76-dim = BERT 64 + Structured 12) - Complete
    # =========================================================================

    # --- LightCNN + Concat Fusion (baseline) ---
    "hybrid_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_concat_median": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat_median",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="median",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- LightCNN + Gated Fusion ---
    "hybrid_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_gated_trimmed": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_gated_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="trimmed_mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- LightCNN + Attention Fusion ---
    "hybrid_cnn_attention": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_attention",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_attention_trimmed": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_attention_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="attention",
            aggregation="trimmed_mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- LightCNN + FiLM Fusion ---
    "hybrid_cnn_film": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_film_trimmed": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_film_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="trimmed_mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- MLP + Hybrid ---
    "hybrid_mlp_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_mlp_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_mlp_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_mlp_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- Position-Aware Aggregation (LightCNN + Hybrid) ---
    "hybrid_cnn_concat_attn_agg": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat_attn_agg",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="attention",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_concat_pos_attn": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat_pos_attn",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="pos_attention",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_concat_spatial_attn": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat_spatial_attn",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="spatial_attention",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_hybrid_config()
        )
    ),

    # --- Gated + Position-Aware (Hybrid) ---
    "hybrid_cnn_gated_transformer": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_gated_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_gated_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_gated_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="transformer_2d",
            policy_source=_hybrid_config()
        )
    ),

    # --- FiLM + Position-Aware (Hybrid) ---
    "hybrid_cnn_film_transformer": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_film_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_cnn_film_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_film_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="transformer_2d",
            policy_source=_hybrid_config()
        )
    ),

    # --- Custom LightCNN (Hybrid) ---
    "hybrid_cnn_small_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_small_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            light_cnn_channels=[16, 32, 64],
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- ResNet10 + Hybrid ---
    "hybrid_resnet10_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet10_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet10_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet10_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet10",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- ResNet18 + Hybrid ---
    "hybrid_resnet18_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet18_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet18_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet18_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet18_film": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet18_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="film",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet18_concat_transformer": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet18_concat_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet18_concat_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet18_concat_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_hybrid_config()
        )
    ),

    # --- ResNet34 + Hybrid ---
    "hybrid_resnet34_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet34_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet34_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet34_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet34_film": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet34_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            fusion_type="film",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),
    "hybrid_resnet34_pretrained": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet34_pretrained",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet34",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        )
    ),

    # --- ResNet50 + Hybrid ---
    "hybrid_resnet50_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet50_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "hybrid_resnet50_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet50_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),
    "hybrid_resnet50_imagenet": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet50_imagenet",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet50",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=8)
    ),

    # --- ResNet101 + Hybrid ---
    "hybrid_resnet101_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet101_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),
    "hybrid_resnet101_imagenet": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet101_imagenet",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet101",
            use_pretrained_resnet=True,
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(batch_size=4)
    ),

    # --- Patch-level (Hybrid) ---
    "hybrid_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "hybrid_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "hybrid_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=64,
            patch_level_aggregation="trimmed_mean"
        )
    ),
    "hybrid_resnet18_concat_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_resnet18_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="resnet",
            resnet_model_name="resnet18",
            fusion_type="concat",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=32,
            patch_level_aggregation="trimmed_mean"
        )
    ),

    # --- SimCLR Pretrained (Hybrid) ---
    "hybrid_simclr_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_simclr_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_simclr_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_simclr_cnn_film": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_simclr_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_simclr_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_simclr_mlp_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_mlp_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="mlp",
            mlp_hidden_dims=[256, 128],
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=SimCLRConfig(encoder_type="mlp"),
        use_pretrain=True
    ),
    "hybrid_simclr_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_simclr_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_hybrid_config()
        ),
        train_config=MultiModalTrainConfig(
            training_mode="patch_level",
            batch_size=8,
            patch_level_aggregation="trimmed_mean"
        ),
        pretrain_config=SimCLRConfig(encoder_type="light_cnn"),
        use_pretrain=True,
        num_workers=2
    ),

    # --- MAE Pretrained (Hybrid) ---
    "hybrid_mae_cnn_concat": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_concat",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_mae_cnn_concat_trimmed": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_concat_trimmed",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="trimmed_mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_mae_cnn_gated": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_gated",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_mae_cnn_film": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_film",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            aggregation="mean",
            policy_source=_hybrid_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_mae_cnn_transformer": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_transformer",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer",
            policy_source=_hybrid_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_mae_cnn_transformer_2d": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_transformer_2d",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            aggregation="transformer_2d",
            policy_source=_hybrid_config()
        ),
        pretrain_config=MAEConfig(encoder_type="light_cnn"),
        use_pretrain=True
    ),
    "hybrid_mae_cnn_concat_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_concat_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            policy_source=_hybrid_config()
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
    "hybrid_mae_cnn_film_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_film_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="film",
            policy_source=_hybrid_config()
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
    "hybrid_mae_cnn_gated_patch": MultiModalBertExperimentConfig(
        exp_name="hybrid_mae_cnn_gated_patch",
        model_config=MultiModalBertConfig(
            image_encoder_type="light_cnn",
            fusion_type="gated",
            policy_source=_hybrid_config()
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


# =============================================================================
# Helper Functions
# =============================================================================

def get_bert_multimodal_experiment_config(name: str) -> MultiModalBertExperimentConfig:
    """Get experiment configuration by name."""
    if name not in BERT_MULTIMODAL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(BERT_MULTIMODAL_EXPERIMENTS.keys())}")
    return BERT_MULTIMODAL_EXPERIMENTS[name]


def list_experiments_by_policy_source(source: str = None):
    """List experiments filtered by policy source."""
    result = {}
    for name, cfg in BERT_MULTIMODAL_EXPERIMENTS.items():
        ps = cfg.model_config.policy_source.source
        if source is None or ps == source:
            if ps not in result:
                result[ps] = []
            result[ps].append(name)
    return result


def print_bert_multimodal_config(exp_config: MultiModalBertExperimentConfig):
    """Print experiment configuration."""
    mc = exp_config.model_config
    tc = exp_config.train_config
    ps = mc.policy_source

    print("=" * 60)
    print(f"Experiment: {exp_config.exp_name}")
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
    print(f"  Policy Source: {ps.source} ({ps.policy_dim}-dim)")
    if ps.policy_dim >= 768:
        print(f"  Policy Projection: {ps.policy_dim}→{mc.policy_hidden_dim}→{mc.policy_output_dim} (trainable E2E)")
    print(f"  Fusion Type: {mc.fusion_type}")
    print(f"  Training Mode: {tc.training_mode}")

    if tc.training_mode == "patch_level":
        print(f"  Patch Aggregation: {tc.patch_level_aggregation}")
    else:
        print(f"  Aggregation: {mc.aggregation}")

    if exp_config.use_pretrain and exp_config.pretrain_config:
        pc = exp_config.pretrain_config
        print(f"  Self-Supervised Pretrain: {pc.name}")
    else:
        print(f"  Self-Supervised Pretrain: None")

    print(f"  Batch Size: {tc.batch_size}")
    print(f"  Learning Rate: {tc.learning_rate}")
    print(f"  Epochs: {tc.num_epochs}")
    print("=" * 60)


if __name__ == "__main__":
    # Count by policy source
    by_source = list_experiments_by_policy_source()
    print("=" * 60)
    print("Unified Multimodal Experiments Summary")
    print("=" * 60)
    total = 0
    for source in ['structured', 'bert', 'hybrid']:
        count = len(by_source.get(source, []))
        total += count
        print(f"  {source:12s}: {count} experiments")
    print(f"  {'TOTAL':12s}: {total} experiments")
    print("=" * 60)

    # List all experiments
    print("\n--- STRUCTURED (Original Multimodal) ---")
    for name in sorted(by_source.get('structured', [])):
        print(f"  {name}")

    print("\n--- BERT ---")
    for name in sorted(by_source.get('bert', [])):
        print(f"  {name}")

    print("\n--- HYBRID ---")
    for name in sorted(by_source.get('hybrid', [])):
        print(f"  {name}")
