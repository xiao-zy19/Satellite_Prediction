"""
Lightweight CNN model for population growth prediction from 64-dim embeddings

A lightweight CNN that processes spatial structure while being
less complex than ResNet, reducing overfitting risk.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.aggregators import get_aggregator


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and optional pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_pool: bool = True,
        use_batch_norm: bool = True
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if use_pool:
            layers.append(nn.MaxPool2d(2, 2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LightCNNEncoder(nn.Module):
    """Lightweight CNN encoder for patches."""

    def __init__(
        self,
        input_channels: int = 64,
        hidden_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels

        # Build convolutional layers
        layers = []
        in_ch = input_channels
        for i, (out_ch, k) in enumerate(zip(hidden_channels, kernel_sizes)):
            layers.append(ConvBlock(
                in_ch, out_ch, k,
                padding=k // 2,
                use_pool=True,
                use_batch_norm=use_batch_norm
            ))
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.output_channels = hidden_channels[-1]

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Args:
            x: (batch, 64, H, W) or (batch, num_patches, 64, H, W)
        Returns:
            features: (batch, output_channels) or (batch, num_patches, output_channels)
        """
        if x.dim() == 5:
            batch_size, num_patches, c, h, w = x.shape
            x = x.view(batch_size * num_patches, c, h, w)

            x = self.conv_layers(x)
            x = self.global_pool(x)
            x = x.view(batch_size, num_patches, -1)
        else:
            x = self.conv_layers(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)

        return x

    @property
    def output_dim(self):
        return self.output_channels


class LightCNN(nn.Module):
    """
    Lightweight CNN model for population growth prediction.

    Architecture:
    1. LightCNNEncoder: Encode each patch with small CNN
    2. Aggregation: Combine patch features (city_level mode only)
    3. Regression head: Predict growth rate

    Supports two training modes:
    - city_level: input (batch, num_patches, 64, H, W) -> aggregate -> output (batch, 1)
    - patch_level: input (batch, 64, H, W) -> output (batch, 1)
    """

    def __init__(self, model_config=None, patch_level: bool = False):
        super().__init__()

        if model_config is None:
            model_config = config.LightCNNConfig()

        self.config = model_config
        self.aggregation_type = model_config.aggregation
        self.patch_level = patch_level  # If True, expect single patch input

        # Patch encoder
        self.encoder = LightCNNEncoder(
            input_channels=model_config.input_channels,
            hidden_channels=model_config.hidden_channels,
            kernel_sizes=model_config.kernel_sizes,
            use_batch_norm=model_config.use_batch_norm
        )
        feature_dim = self.encoder.output_dim

        # Aggregation (only used in city_level mode)
        if not patch_level:
            self.aggregator = get_aggregator(
                aggregation_type=self.aggregation_type,
                feature_dim=feature_dim,
                num_patches=config.NUM_PATCHES_TOTAL,
                grid_size=config.NUM_PATCHES_PER_DIM,
                dropout=model_config.dropout_rate
            )
        else:
            self.aggregator = None

        # FC layers
        fc_layers = []
        fc_dims = [feature_dim] + model_config.fc_dims
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(model_config.dropout_rate))

        self.fc = nn.Sequential(*fc_layers)

        # Regression head
        self.regression_head = nn.Linear(fc_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
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
            elif self.aggregation_type == "median":
                aggregated = features.median(dim=1).values
            elif self.aggregation_type == "trimmed_mean":
                aggregated = self._trimmed_mean(features)
            elif self.aggregator is not None:
                # Use advanced aggregator (attention, pos_attention, transformer, etc.)
                aggregated = self.aggregator(features)
            else:
                aggregated = features.mean(dim=1)

        # FC layers
        x = self.fc(aggregated)

        # Predict
        output = self.regression_head(x)
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

    def get_features(self, x):
        """Get patch features for visualization."""
        return self.encoder(x)


if __name__ == "__main__":
    # Test model
    print("=" * 60)
    print("Testing LightCNN model (city_level mode)...")
    print("=" * 60)

    model = LightCNN(patch_level=False)
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

    # Test encoder output
    with torch.no_grad():
        features = model.get_features(x)
    print(f"Features shape: {features.shape}")

    print("\n" + "=" * 60)
    print("Testing LightCNN model (patch_level mode)...")
    print("=" * 60)

    model_pl = LightCNN(patch_level=True)
    print(f"Model parameters: {sum(p.numel() for p in model_pl.parameters()):,}")

    # Test forward pass - patch level (single patch)
    x_single = torch.randn(batch_size, channels, patch_size, patch_size)
    print(f"Input shape: {x_single.shape}")

    with torch.no_grad():
        output_pl = model_pl(x_single)

    print(f"Output shape: {output_pl.shape}")
    print(f"Output: {output_pl.squeeze().tolist()}")
