"""
MLP-based model for population growth prediction from 64-dim embeddings

Since the input is already high-level features from Alpha Earth,
a simple MLP might be sufficient and less prone to overfitting.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MLPBlock(nn.Module):
    """MLP block with optional batch norm and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchMLP(nn.Module):
    """MLP encoder for individual patches."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        pool_spatial: bool = True
    ):
        super().__init__()
        self.pool_spatial = pool_spatial
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        # If pooling spatial dims, input is just 64
        # Otherwise, need to flatten: 64 * 200 * 200 (too large, so we pool)

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(MLPBlock(dims[i], dims[i+1], dropout_rate, use_batch_norm))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, 64, H, W) single patch or (batch, num_patches, 64, H, W)
        Returns:
            features: (batch, output_dim) or (batch, num_patches, output_dim)
        """
        if x.dim() == 5:
            # (batch, num_patches, 64, H, W)
            batch_size, num_patches, c, h, w = x.shape
            x = x.view(batch_size * num_patches, c, h, w)
            # Global average pooling
            x = x.mean(dim=(2, 3))  # (batch * num_patches, 64)
            x = self.mlp(x)
            x = x.view(batch_size, num_patches, -1)
        else:
            # (batch, 64, H, W)
            x = x.mean(dim=(2, 3))  # Global average pooling
            x = self.mlp(x)

        return x


class AttentionAggregator(nn.Module):
    """Attention-based aggregation of patch features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        attn_weights = self.attention(x)  # (batch, num_patches, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        out = (x * attn_weights).sum(dim=1)
        return out


class MLPModel(nn.Module):
    """
    MLP-based model for population growth prediction.

    Architecture:
    1. PatchMLP: Encode each patch (global pool + MLP)
    2. Aggregation: Combine patch features (mean/attention) - city_level mode only
    3. Regression head: Predict growth rate

    Supports two training modes:
    - city_level: input (batch, num_patches, 64, H, W) -> aggregate -> output (batch, 1)
    - patch_level: input (batch, 64, H, W) -> output (batch, 1)
    """

    def __init__(self, model_config=None, patch_level: bool = False):
        super().__init__()

        if model_config is None:
            model_config = config.MLPConfig()

        self.config = model_config
        self.aggregation_type = model_config.aggregation
        self.patch_level = patch_level  # If True, expect single patch input

        # Patch encoder
        self.encoder = PatchMLP(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.hidden_dims,
            dropout_rate=model_config.dropout_rate,
            use_batch_norm=model_config.use_batch_norm
        )
        feature_dim = self.encoder.output_dim

        # Aggregation (only used in city_level mode)
        if self.aggregation_type == "attention" and not patch_level:
            self.aggregator = AttentionAggregator(feature_dim)
        else:
            self.aggregator = None

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, 64, H, W) for city_level mode
               (batch, 64, H, W) for patch_level mode
        Returns:
            (batch, 1)
        """
        # Encode patches
        features = self.encoder(x)

        if self.patch_level or x.dim() == 4:
            # Patch-level mode: features is (batch, feature_dim)
            # No aggregation needed
            aggregated = features
        else:
            # City-level mode: features is (batch, num_patches, feature_dim)
            # Aggregate across patches
            if self.aggregation_type == "mean":
                aggregated = features.mean(dim=1)
            elif self.aggregation_type == "attention":
                aggregated = self.aggregator(features)
            elif self.aggregation_type == "trimmed_mean":
                aggregated = self._trimmed_mean(features)
            else:
                aggregated = features.mean(dim=1)

        # Predict
        output = self.regression_head(aggregated)
        return output

    def _trimmed_mean(self, features, trim_ratio=0.1):
        """Compute trimmed mean."""
        batch_size, num_patches, feature_dim = features.shape
        norms = features.norm(dim=2)
        _, indices = norms.sort(dim=1)
        n_trim = int(num_patches * trim_ratio)
        keep_indices = indices[:, n_trim:num_patches - n_trim]
        keep_indices = keep_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        kept_features = torch.gather(features, 1, keep_indices)
        return kept_features.mean(dim=1)

    def get_encoder(self):
        """Return encoder for pretraining."""
        return self.encoder

    def load_encoder(self, encoder_state_dict):
        """Load pretrained encoder weights."""
        self.encoder.load_state_dict(encoder_state_dict)

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test model
    print("=" * 60)
    print("Testing MLP model (city_level mode)...")
    print("=" * 60)

    model = MLPModel(patch_level=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass - city level
    batch_size = 2
    num_patches = 25
    channels = 64
    patch_size = 200

    x = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output: {output.squeeze().tolist()}")

    print("\n" + "=" * 60)
    print("Testing MLP model (patch_level mode)...")
    print("=" * 60)

    model_pl = MLPModel(patch_level=True)
    print(f"Model parameters: {sum(p.numel() for p in model_pl.parameters()):,}")

    # Test forward pass - patch level (single patch)
    x_single = torch.randn(batch_size, channels, patch_size, patch_size)
    print(f"Input shape: {x_single.shape}")

    with torch.no_grad():
        output_pl = model_pl(x_single)

    print(f"Output shape: {output_pl.shape}")
    print(f"Output: {output_pl.squeeze().tolist()}")
