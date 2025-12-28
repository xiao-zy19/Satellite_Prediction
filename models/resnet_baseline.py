"""
ResNet baseline model for population growth prediction

Modified ResNet for 64-channel input (Alpha Earth embeddings)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ResNetEncoder(nn.Module):
    """
    ResNet encoder modified for 64-channel input.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        input_channels: int = 64,
        pretrained: bool = False
    ):
        super().__init__()

        # Load model with or without pretrained weights
        if model_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif model_name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet34(weights=weights)
            self.feature_dim = 512
        elif model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Modify first conv layer for 64 channels
        original_conv = base_model.conv1
        self.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new conv layer
        if pretrained:
            with torch.no_grad():
                mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
                self.conv1.weight = nn.Parameter(
                    mean_weight.repeat(1, input_channels, 1, 1) / input_channels
                )

        # Rest of ResNet
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        """
        Args:
            x: (batch, 64, H, W) or (batch, num_patches, 64, H, W)
        Returns:
            features
        """
        if x.dim() == 5:
            batch_size, num_patches, c, h, w = x.shape
            x = x.view(batch_size * num_patches, c, h, w)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = x.view(batch_size, num_patches, -1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        return x

    @property
    def output_dim(self):
        return self.feature_dim


class AttentionAggregator(nn.Module):
    """Attention-based aggregation."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        out = (x * attn_weights).sum(dim=1)
        return out


class ResNetBaseline(nn.Module):
    """
    ResNet baseline model for population growth prediction.
    """

    def __init__(self, model_config=None):
        super().__init__()

        if model_config is None:
            model_config = config.ResNetConfig()

        self.config = model_config
        self.aggregation_type = model_config.aggregation

        # Encoder
        self.encoder = ResNetEncoder(
            model_name=model_config.model_name,
            input_channels=model_config.input_channels,
            pretrained=model_config.use_pretrained
        )
        feature_dim = self.encoder.output_dim

        # Aggregation
        if self.aggregation_type == "attention":
            self.aggregator = AttentionAggregator(feature_dim, model_config.hidden_dim // 2)
        else:
            self.aggregator = None

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(model_config.hidden_dim // 2, 1)
        )

        self._init_head()

    def _init_head(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, 64, H, W)
        Returns:
            (batch, 1)
        """
        # Encode
        features = self.encoder(x)  # (batch, num_patches, feature_dim)

        # Aggregate
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
        batch_size, num_patches, feature_dim = features.shape
        norms = features.norm(dim=2)
        _, indices = norms.sort(dim=1)
        n_trim = max(1, int(num_patches * trim_ratio))
        keep_indices = indices[:, n_trim:num_patches - n_trim]
        keep_indices = keep_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        kept_features = torch.gather(features, 1, keep_indices)
        return kept_features.mean(dim=1)

    def get_encoder(self):
        return self.encoder

    def load_encoder(self, encoder_state_dict):
        self.encoder.load_state_dict(encoder_state_dict)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    print("Testing ResNet baseline model...")

    # Test without pretrained
    model = ResNetBaseline(config.ResNetConfig(use_pretrained=False))
    print(f"Model parameters (no pretrain): {sum(p.numel() for p in model.parameters()):,}")

    # Test with pretrained
    model_pt = ResNetBaseline(config.ResNetConfig(use_pretrained=True))
    print(f"Model parameters (pretrained): {sum(p.numel() for p in model_pt.parameters()):,}")

    # Test forward pass
    batch_size = 2
    num_patches = 25
    channels = 64
    patch_size = 200

    x = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
