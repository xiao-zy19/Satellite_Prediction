"""
Multimodal model for population growth prediction.

Combines satellite image features with structured policy features
using various fusion strategies.

Fusion strategies:
1. concat: Simple concatenation of features
2. gated: Gated fusion with learnable mixing weights
3. attention: Cross-attention between modalities
4. film: Feature-wise Linear Modulation (policy modulates image features)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.light_cnn import LightCNNEncoder
from models.mlp_model import PatchMLP
from models.resnet_baseline import ResNetEncoder
from models.aggregators import get_aggregator


class PolicyEncoder(nn.Module):
    """
    Optional MLP encoder for policy features.

    Can either:
    1. Pass through raw features (use_encoder=False) - recommended for simplicity
    2. Map to different dimension (use_encoder=True) - for dimension alignment
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
        use_encoder: bool = False  # Default: use raw 12-dim features
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.input_dim = input_dim

        if use_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.output_dim = output_dim
        else:
            # Pass through: output_dim = input_dim
            self.encoder = None
            self.output_dim = input_dim

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) policy features

        Returns:
            (batch, output_dim) policy features (encoded or raw)
        """
        if self.use_encoder and self.encoder is not None:
            return self.encoder(x)
        return x


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""

    def __init__(self, image_dim: int, policy_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + policy_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_dim = output_dim

    def forward(self, image_feat, policy_feat):
        """
        Args:
            image_feat: (batch, image_dim)
            policy_feat: (batch, policy_dim)

        Returns:
            (batch, output_dim)
        """
        combined = torch.cat([image_feat, policy_feat], dim=-1)
        return self.fusion(combined)


class GatedFusion(nn.Module):
    """Gated fusion with learnable mixing weights."""

    def __init__(self, image_dim: int, policy_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        # Project both to same dimension
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.policy_proj = nn.Linear(policy_dim, output_dim)

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(image_dim + policy_dim, output_dim),
            nn.Sigmoid()
        )

        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, policy_feat):
        """
        Args:
            image_feat: (batch, image_dim)
            policy_feat: (batch, policy_dim)

        Returns:
            (batch, output_dim)
        """
        image_proj = self.image_proj(image_feat)
        policy_proj = self.policy_proj(policy_feat)

        gate = self.gate(torch.cat([image_feat, policy_feat], dim=-1))

        fused = gate * image_proj + (1 - gate) * policy_proj
        return self.dropout(F.relu(fused))


class AttentionFusion(nn.Module):
    """Cross-attention fusion between modalities."""

    def __init__(self, image_dim: int, policy_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        # Project both to same dimension for attention
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.policy_proj = nn.Linear(policy_dim, output_dim)

        # Multi-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, policy_feat):
        """
        Args:
            image_feat: (batch, image_dim)
            policy_feat: (batch, policy_dim)

        Returns:
            (batch, output_dim)
        """
        # Project features
        image_proj = self.image_proj(image_feat).unsqueeze(1)  # (batch, 1, output_dim)
        policy_proj = self.policy_proj(policy_feat).unsqueeze(1)  # (batch, 1, output_dim)

        # Stack as sequence: [image, policy]
        seq = torch.cat([image_proj, policy_proj], dim=1)  # (batch, 2, output_dim)

        # Self-attention over modalities
        attn_out, _ = self.cross_attn(seq, seq, seq)  # (batch, 2, output_dim)

        # Take mean of attended features
        fused = attn_out.mean(dim=1)  # (batch, output_dim)
        fused = self.norm(fused)

        return self.dropout(F.relu(fused))


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation: policy modulates image features."""

    def __init__(self, image_dim: int, policy_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        # Generate scale (gamma) and shift (beta) from policy
        self.gamma_net = nn.Linear(policy_dim, image_dim)
        self.beta_net = nn.Linear(policy_dim, image_dim)

        # Project modulated features
        self.proj = nn.Linear(image_dim, output_dim)

        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, policy_feat):
        """
        Args:
            image_feat: (batch, image_dim)
            policy_feat: (batch, policy_dim)

        Returns:
            (batch, output_dim)
        """
        gamma = self.gamma_net(policy_feat)  # (batch, image_dim)
        beta = self.beta_net(policy_feat)    # (batch, image_dim)

        # Apply FiLM: y = gamma * x + beta
        modulated = gamma * image_feat + beta

        fused = self.proj(modulated)
        return self.dropout(F.relu(fused))


def get_fusion_layer(
    fusion_type: str,
    image_dim: int,
    policy_dim: int,
    output_dim: int,
    dropout: float = 0.3
) -> nn.Module:
    """Factory function for fusion layers."""
    if fusion_type == "concat":
        return ConcatFusion(image_dim, policy_dim, output_dim, dropout)
    elif fusion_type == "gated":
        return GatedFusion(image_dim, policy_dim, output_dim, dropout)
    elif fusion_type == "attention":
        return AttentionFusion(image_dim, policy_dim, output_dim, dropout=dropout)
    elif fusion_type == "film":
        return FiLMFusion(image_dim, policy_dim, output_dim, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


class MultiModalModel(nn.Module):
    """
    Multimodal model combining satellite images and policy features.

    Architecture:
    1. Image encoder (LightCNN/MLP/ResNet) -> image_dim (e.g., 128)
    2. Aggregation (for city-level mode)
    3. Policy features (raw 12-dim, no encoding by default)
    4. Fusion layer (concat/gated/attention/film)
    5. Regression head

    Supports two training modes:
    - city_level: input (batch, num_patches, 64, H, W) -> aggregate -> fuse -> output (batch, 1)
    - patch_level: input (batch, 64, H, W) -> fuse -> output (batch, 1)
    """

    def __init__(
        self,
        image_encoder_type: str = "light_cnn",
        fusion_type: str = "concat",
        policy_feature_dim: int = 12,
        policy_hidden_dim: int = 64,
        image_feature_dim: int = 64,  # Project image features to this dim (default 64)
        aggregation_type: str = "mean",
        dropout: float = 0.3,
        patch_level: bool = False,
        use_policy_encoder: bool = False,  # Default: use raw 12-dim features
        model_config=None
    ):
        super().__init__()

        self.image_encoder_type = image_encoder_type
        self.fusion_type = fusion_type
        self.aggregation_type = aggregation_type
        self.patch_level = patch_level
        self.use_policy_encoder = use_policy_encoder
        self.image_feature_dim = image_feature_dim

        # Image encoder
        if image_encoder_type == "light_cnn":
            # LightCNN config from model_config (if provided and not None)
            light_cnn_channels = getattr(model_config, 'light_cnn_channels', None) if model_config else None
            if light_cnn_channels is not None:
                light_cnn_kernels = getattr(model_config, 'light_cnn_kernels', None) or [3, 3, 3]
                self.image_encoder = LightCNNEncoder(
                    input_channels=config.NUM_BANDS,
                    hidden_channels=light_cnn_channels,
                    kernel_sizes=light_cnn_kernels,
                    use_batch_norm=getattr(model_config, 'use_batch_norm', True)
                )
            else:
                self.image_encoder = LightCNNEncoder()
            encoder_out_dim = self.image_encoder.output_dim
        elif image_encoder_type == "mlp":
            self.image_encoder = PatchMLP(
                input_dim=config.NUM_BANDS,
                hidden_dims=[256, 128, 64]
            )
            encoder_out_dim = 64
        elif image_encoder_type == "resnet":
            # ResNet config from model_config
            resnet_name = "resnet18"
            resnet_pretrained = False
            if model_config is not None:
                resnet_name = getattr(model_config, 'resnet_model_name', 'resnet18')
                resnet_pretrained = getattr(model_config, 'use_pretrained_resnet', False)
            self.image_encoder = ResNetEncoder(
                model_name=resnet_name,
                input_channels=config.NUM_BANDS,
                pretrained=resnet_pretrained
            )
            encoder_out_dim = self.image_encoder.output_dim
        else:
            raise ValueError(f"Unknown image encoder: {image_encoder_type}")

        # Image feature projection (if needed)
        # Projects encoder output to image_feature_dim
        if encoder_out_dim != image_feature_dim:
            self.image_proj = nn.Sequential(
                nn.Linear(encoder_out_dim, image_feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            image_dim = image_feature_dim
        else:
            self.image_proj = None
            image_dim = encoder_out_dim

        # Aggregator (for city-level mode)
        # Note: aggregator works on encoder_out_dim, projection happens after aggregation
        if not patch_level:
            self.aggregator = get_aggregator(
                aggregation_type=aggregation_type,
                feature_dim=encoder_out_dim,  # Use encoder output dim for aggregation
                num_patches=config.NUM_PATCHES_TOTAL,
                grid_size=config.NUM_PATCHES_PER_DIM,
                dropout=dropout
            )
        else:
            self.aggregator = None

        self.encoder_out_dim = encoder_out_dim  # Store for aggregation

        # Policy encoder (optional, default is pass-through)
        # use_policy_encoder=False: policy_dim = 12 (raw features)
        # use_policy_encoder=True: policy_dim = policy_hidden_dim (encoded)
        self.policy_encoder = PolicyEncoder(
            input_dim=policy_feature_dim,
            hidden_dim=policy_hidden_dim,
            output_dim=policy_hidden_dim,
            dropout=dropout,
            use_encoder=use_policy_encoder
        )
        policy_dim = self.policy_encoder.output_dim  # 12 or policy_hidden_dim

        # Fusion layer
        # Default: image_dim(64) + policy_dim(12) -> 64
        fusion_output_dim = 64
        self.fusion = get_fusion_layer(
            fusion_type=fusion_type,
            image_dim=image_dim,
            policy_dim=policy_dim,
            output_dim=fusion_output_dim,
            dropout=dropout
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_patches, policy_features):
        """
        Args:
            image_patches: (batch, num_patches, 64, H, W) for city_level
                          or (batch, 64, H, W) for patch_level
            policy_features: (batch, policy_feature_dim)

        Returns:
            (batch, 1) prediction
        """
        # Encode image patches
        image_feat = self.image_encoder(image_patches)

        if self.patch_level or image_patches.dim() == 4:
            # Patch-level: image_feat is (batch, encoder_out_dim)
            aggregated_image = image_feat
        else:
            # City-level: image_feat is (batch, num_patches, encoder_out_dim)
            # Aggregate across patches
            if self.aggregation_type == "mean":
                aggregated_image = image_feat.mean(dim=1)
            elif self.aggregation_type == "median":
                aggregated_image = image_feat.median(dim=1).values
            elif self.aggregation_type == "trimmed_mean":
                aggregated_image = self._trimmed_mean(image_feat)
            elif self.aggregator is not None:
                aggregated_image = self.aggregator(image_feat)
            else:
                aggregated_image = image_feat.mean(dim=1)

        # Project image features to image_feature_dim (if needed)
        if self.image_proj is not None:
            aggregated_image = self.image_proj(aggregated_image)

        # Encode policy features (pass-through if use_policy_encoder=False)
        policy_feat = self.policy_encoder(policy_features)

        # Fuse modalities
        fused = self.fusion(aggregated_image, policy_feat)

        # Predict
        output = self.regression_head(fused)
        return output

    def _trimmed_mean(self, features, trim_ratio=0.1):
        batch_size, num_patches, feature_dim = features.shape
        norms = features.norm(dim=2)
        _, indices = norms.sort(dim=1)
        n_trim = int(num_patches * trim_ratio)
        if n_trim == 0:
            n_trim = 1
        keep_indices = indices[:, n_trim:num_patches - n_trim]
        keep_indices = keep_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        kept_features = torch.gather(features, 1, keep_indices)
        return kept_features.mean(dim=1)

    def get_encoder(self):
        """Return image encoder for pretraining."""
        return self.image_encoder

    def load_encoder(self, encoder_state_dict):
        """Load pretrained image encoder weights."""
        self.image_encoder.load_state_dict(encoder_state_dict)

    def freeze_encoder(self):
        """Freeze image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test model
    batch_size = 2
    num_patches = 25
    channels = 64
    patch_size = 200
    policy_dim = 12

    x_img = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)
    x_img_single = torch.randn(batch_size, channels, patch_size, patch_size)
    x_policy = torch.randn(batch_size, policy_dim)

    print("=" * 60)
    print("Default config: image_dim=64, policy_dim=12 (no encoder)")
    print("=" * 60)

    model = MultiModalModel(
        image_encoder_type="light_cnn",
        fusion_type="concat",
        image_feature_dim=64,  # Project 128 -> 64
        patch_level=True,
        use_policy_encoder=False  # Keep policy as 12-dim
    )
    print(f"Image feature dim: {model.image_feature_dim}")
    print(f"Policy feature dim: {model.policy_encoder.output_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        output = model(x_img_single, x_policy)
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("Testing different fusion types (image=64, policy=12)")
    print("=" * 60)

    for fusion_type in ["concat", "gated", "attention", "film"]:
        model_test = MultiModalModel(
            image_encoder_type="light_cnn",
            fusion_type=fusion_type,
            image_feature_dim=64,
            patch_level=True,
            use_policy_encoder=False
        )
        params = sum(p.numel() for p in model_test.parameters())
        print(f"{fusion_type:12s}: {params:,} params")

    print("\n" + "=" * 60)
    print("Comparison: different image_feature_dim values")
    print("=" * 60)

    for img_dim in [32, 64, 128]:
        model_test = MultiModalModel(
            image_encoder_type="light_cnn",
            fusion_type="concat",
            image_feature_dim=img_dim,
            patch_level=True,
            use_policy_encoder=False
        )
        params = sum(p.numel() for p in model_test.parameters())
        has_proj = "with projection" if model_test.image_proj is not None else "no projection"
        print(f"image_dim={img_dim:3d}: {params:,} params ({has_proj})")

    print("\n" + "=" * 60)
    print("City-level mode test")
    print("=" * 60)

    model_city = MultiModalModel(
        image_encoder_type="light_cnn",
        fusion_type="concat",
        image_feature_dim=64,
        aggregation_type="mean",
        patch_level=False,
        use_policy_encoder=False
    )
    print(f"Model parameters: {sum(p.numel() for p in model_city.parameters()):,}")

    with torch.no_grad():
        output_city = model_city(x_img, x_policy)
    print(f"Input: image={x_img.shape}, policy={x_policy.shape}")
    print(f"Output: {output_city.shape}")
