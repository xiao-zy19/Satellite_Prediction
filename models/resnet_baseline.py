"""
ResNet baseline model for population growth prediction

Modified ResNet for 64-channel input (Alpha Earth embeddings)

Supports multiple ResNet variants for scaling experiments:
- resnet10: Custom lightweight ResNet (fewer layers)
- resnet18: Standard ResNet-18
- resnet34: Standard ResNet-34
- resnet50: Standard ResNet-50 (with bottleneck blocks)
- resnet101: Standard ResNet-101 (with bottleneck blocks)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.aggregators import get_aggregator


# =============================================================================
# Custom ResNet10 Implementation (not available in torchvision)
# =============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-10/18/34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet10(nn.Module):
    """
    Custom ResNet-10 implementation.
    Layer configuration: [1, 1, 1, 1] (one block per stage)
    Total: 1 + 2*4 + 1 = 10 conv layers (including first conv and fc)
    """

    def __init__(self, input_channels=64, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # First conv layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers: [1, 1, 1, 1]
        self.layer1 = self._make_layer(64, 1, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
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
        x = self.fc(x)

        return x


# =============================================================================
# ResNet Encoder (supports all variants)
# =============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet encoder modified for 64-channel input.

    Supported models:
    - resnet10: ~5M params, custom implementation
    - resnet18: ~11M params
    - resnet34: ~21M params
    - resnet50: ~25M params (bottleneck)
    - resnet101: ~44M params (bottleneck)
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        input_channels: int = 64,
        pretrained: bool = False
    ):
        super().__init__()
        self.model_name = model_name

        # Load model based on name
        if model_name == "resnet10":
            # Custom ResNet-10 (no pretrained weights available)
            if pretrained:
                print("Warning: No pretrained weights available for ResNet-10, using random init")
            base_model = ResNet10(input_channels=input_channels)
            self.feature_dim = 512
            # For ResNet10, we already have the correct input channels
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
            return  # Skip the rest of __init__

        elif model_name == "resnet18":
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
        elif model_name == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet101(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}. Supported: resnet10, resnet18, resnet34, resnet50, resnet101")

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


class ResNetBaseline(nn.Module):
    """
    ResNet baseline model for population growth prediction.

    Supports two training modes:
    - city_level: input (batch, num_patches, 64, H, W) -> aggregate -> output (batch, 1)
    - patch_level: input (batch, 64, H, W) -> output (batch, 1)
    """

    def __init__(self, model_config=None, patch_level: bool = False):
        super().__init__()

        if model_config is None:
            model_config = config.ResNetConfig()

        self.config = model_config
        self.aggregation_type = model_config.aggregation
        self.patch_level = patch_level  # If True, expect single patch input

        # Encoder
        self.encoder = ResNetEncoder(
            model_name=model_config.model_name,
            input_channels=model_config.input_channels,
            pretrained=model_config.use_pretrained
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
            x: (batch, num_patches, 64, H, W) for city_level mode
               (batch, 64, H, W) for patch_level mode
        Returns:
            (batch, 1)
        """
        # Encode
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
    print("=" * 70)
    print("Testing ResNet baseline models (all variants)")
    print("=" * 70)

    # Test all ResNet variants
    variants = ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101"]

    batch_size = 2
    num_patches = 25
    channels = 64
    patch_size = 200

    # City-level input
    x_city = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)
    # Patch-level input
    x_patch = torch.randn(batch_size, channels, patch_size, patch_size)

    print(f"\nCity-level input shape: {x_city.shape}")
    print(f"Patch-level input shape: {x_patch.shape}")
    print()

    results = []
    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Testing {variant.upper()}")
        print(f"{'='*70}")

        # Create model config
        model_config = config.ResNetConfig(model_name=variant, use_pretrained=False)

        # City-level model
        model_city = ResNetBaseline(model_config, patch_level=False)
        params = sum(p.numel() for p in model_city.parameters())
        print(f"  Parameters: {params:,}")

        # Test forward pass
        with torch.no_grad():
            output_city = model_city(x_city)
            print(f"  City-level output shape: {output_city.shape}")

        # Patch-level model
        model_patch = ResNetBaseline(model_config, patch_level=True)
        with torch.no_grad():
            output_patch = model_patch(x_patch)
            print(f"  Patch-level output shape: {output_patch.shape}")

        results.append({
            'model': variant,
            'params': params,
            'feature_dim': model_city.encoder.feature_dim
        })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: ResNet Scaling")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Parameters':>15} {'Feature Dim':>15}")
    print("-" * 45)
    for r in results:
        print(f"{r['model']:<12} {r['params']:>15,} {r['feature_dim']:>15}")
    print(f"{'='*70}")
