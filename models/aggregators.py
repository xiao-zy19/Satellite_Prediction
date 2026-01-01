"""
Advanced Aggregation Modules for Patch Features

Implements position-aware aggregation mechanisms to combine 25 patch features
from a 5x5 spatial grid into a single city-level representation.

Aggregation methods:
1. mean: Simple average (baseline)
2. trimmed_mean: Average after removing outliers
3. attention: MLP-based attention without position info (existing)
4. pos_attention: Attention with learnable position embeddings
5. spatial_attention: Attention with 2D spatial position encoding
6. transformer: Full transformer with [CLS] token aggregation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionAggregator(nn.Module):
    """
    Simple attention-based aggregation (no position info).
    Kept for backward compatibility.
    """

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


class PositionalAttentionAggregator(nn.Module):
    """
    Attention aggregation with learnable position embeddings.

    The 25 patches are from a 5x5 spatial grid. Position embeddings
    allow the model to learn position-dependent importance weights.
    """

    def __init__(
        self,
        feature_dim: int,
        num_patches: int = 25,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Layer norm before attention
        self.norm = nn.LayerNorm(feature_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        # Add position embeddings
        x = x + self.pos_embed

        # Self-attention
        x = self.norm(x)
        attn_out, attn_weights = self.attention(x, x, x)

        # Residual connection
        x = x + attn_out

        # Global average pooling
        out = x.mean(dim=1)

        # Output projection
        out = self.out_proj(out)

        return out


class SpatialPositionAggregator(nn.Module):
    """
    Aggregation with explicit 2D spatial position encoding.

    Uses separate embeddings for row and column positions,
    capturing the 2D structure of the 5x5 patch grid.
    """

    def __init__(
        self,
        feature_dim: int,
        grid_size: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size

        # 2D position embeddings (row and column)
        # Use half dimension for each, then project to full dimension
        self.pos_embed_dim = feature_dim // 2
        self.row_embed = nn.Embedding(grid_size, self.pos_embed_dim)
        self.col_embed = nn.Embedding(grid_size, self.pos_embed_dim)

        # Initialize embeddings
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

        # Projection to match feature dim (handles both even and odd dimensions)
        self.pos_proj = nn.Linear(self.pos_embed_dim * 2, feature_dim)

        # Layer norm
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

        # Attention weights for final aggregation
        self.agg_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1)
        )

    def get_2d_pos_embed(self, device):
        """Generate 2D position embeddings for the grid."""
        rows = torch.arange(self.grid_size, device=device).repeat_interleave(self.grid_size)
        cols = torch.arange(self.grid_size, device=device).repeat(self.grid_size)

        row_emb = self.row_embed(rows)  # (25, pos_embed_dim)
        col_emb = self.col_embed(cols)  # (25, pos_embed_dim)

        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # (25, pos_embed_dim * 2)
        pos_emb = self.pos_proj(pos_emb)  # (25, feature_dim)

        return pos_emb.unsqueeze(0)  # (1, 25, feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        # Add 2D position embeddings
        pos_emb = self.get_2d_pos_embed(x.device)
        x = x + pos_emb

        # Self-attention block
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out

        # FFN block
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        # Weighted aggregation using attention
        attn_weights = self.agg_attention(x)  # (batch, num_patches, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        out = (x * attn_weights).sum(dim=1)

        return out


class TransformerAggregator(nn.Module):
    """
    Full transformer aggregation with [CLS] token.

    Inspired by ViT, uses a learnable [CLS] token that attends
    to all patch features and serves as the aggregated representation.
    """

    def __init__(
        self,
        feature_dim: int,
        num_patches: int = 25,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embeddings (including [CLS] position)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        batch_size = x.size(0)

        # Expand [CLS] token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate [CLS] with patch features
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + num_patches, feature_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer encoding
        x = self.transformer(x)

        # Extract [CLS] token output
        cls_out = x[:, 0]

        # Final norm
        out = self.norm(cls_out)

        return out


class Transformer2DAggregator(nn.Module):
    """
    Transformer aggregation with explicit 2D spatial position encoding.

    Combines the power of transformer architecture with explicit
    2D spatial awareness for the 5x5 patch grid.
    """

    def __init__(
        self,
        feature_dim: int,
        grid_size: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 2D position embeddings
        # Use half dimension for each, then project to full dimension
        self.pos_embed_dim = feature_dim // 2
        self.row_embed = nn.Embedding(grid_size, self.pos_embed_dim)
        self.col_embed = nn.Embedding(grid_size, self.pos_embed_dim)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

        # [CLS] position embedding
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        # Projection to match feature dim (handles both even and odd dimensions)
        self.pos_proj = nn.Linear(self.pos_embed_dim * 2, feature_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def get_2d_pos_embed(self, device):
        """Generate 2D position embeddings for patches."""
        rows = torch.arange(self.grid_size, device=device).repeat_interleave(self.grid_size)
        cols = torch.arange(self.grid_size, device=device).repeat(self.grid_size)

        row_emb = self.row_embed(rows)  # (25, pos_embed_dim)
        col_emb = self.col_embed(cols)  # (25, pos_embed_dim)

        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # (25, pos_embed_dim * 2)
        pos_emb = self.pos_proj(pos_emb)  # (25, feature_dim)

        return pos_emb.unsqueeze(0)  # (1, 25, feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, feature_dim)
        Returns:
            (batch, feature_dim)
        """
        batch_size = x.size(0)

        # Add 2D position embeddings to patches
        pos_emb = self.get_2d_pos_embed(x.device)
        x = x + pos_emb

        # Expand and add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + self.cls_pos_embed
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Extract [CLS] token output
        cls_out = x[:, 0]
        out = self.norm(cls_out)

        return out


def get_aggregator(
    aggregation_type: str,
    feature_dim: int,
    num_patches: int = 25,
    grid_size: int = 5,
    dropout: float = 0.1,
    **kwargs
) -> Optional[nn.Module]:
    """
    Factory function to create aggregator modules.

    Args:
        aggregation_type: One of ["mean", "trimmed_mean", "attention",
                          "pos_attention", "spatial_attention", "transformer", "transformer_2d"]
        feature_dim: Dimension of input features
        num_patches: Number of patches (default 25 for 5x5 grid)
        grid_size: Grid size for spatial aggregators (default 5)
        dropout: Dropout rate

    Returns:
        Aggregator module or None for mean/trimmed_mean
    """
    if aggregation_type in ["mean", "median", "trimmed_mean"]:
        return None  # Handled in forward pass

    elif aggregation_type == "attention":
        return AttentionAggregator(
            feature_dim=feature_dim,
            hidden_dim=max(64, feature_dim // 2)
        )

    elif aggregation_type == "pos_attention":
        return PositionalAttentionAggregator(
            feature_dim=feature_dim,
            num_patches=num_patches,
            num_heads=4,
            dropout=dropout
        )

    elif aggregation_type == "spatial_attention":
        return SpatialPositionAggregator(
            feature_dim=feature_dim,
            grid_size=grid_size,
            num_heads=4,
            dropout=dropout
        )

    elif aggregation_type == "transformer":
        return TransformerAggregator(
            feature_dim=feature_dim,
            num_patches=num_patches,
            num_layers=kwargs.get('num_layers', 2),
            num_heads=4,
            dropout=dropout
        )

    elif aggregation_type == "transformer_2d":
        return Transformer2DAggregator(
            feature_dim=feature_dim,
            grid_size=grid_size,
            num_layers=kwargs.get('num_layers', 2),
            num_heads=4,
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}. "
                        f"Available: mean, median, trimmed_mean, attention, pos_attention, "
                        f"spatial_attention, transformer, transformer_2d")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Aggregation Modules")
    print("=" * 70)

    batch_size = 4
    num_patches = 25
    feature_dim = 128

    x = torch.randn(batch_size, num_patches, feature_dim)
    print(f"Input shape: {x.shape}")

    aggregators = [
        ("attention", AttentionAggregator(feature_dim)),
        ("pos_attention", PositionalAttentionAggregator(feature_dim, num_patches)),
        ("spatial_attention", SpatialPositionAggregator(feature_dim, grid_size=5)),
        ("transformer", TransformerAggregator(feature_dim, num_patches)),
        ("transformer_2d", Transformer2DAggregator(feature_dim, grid_size=5)),
    ]

    print(f"\n{'Aggregator':<20} {'Parameters':>12} {'Output Shape':>15}")
    print("-" * 50)

    for name, agg in aggregators:
        params = sum(p.numel() for p in agg.parameters())
        with torch.no_grad():
            out = agg(x)
        print(f"{name:<20} {params:>12,} {str(out.shape):>15}")

    # Test factory function
    print("\n" + "=" * 70)
    print("Testing Factory Function")
    print("=" * 70)

    for agg_type in ["attention", "pos_attention", "spatial_attention", "transformer", "transformer_2d"]:
        agg = get_aggregator(agg_type, feature_dim, num_patches)
        with torch.no_grad():
            out = agg(x)
        print(f"{agg_type}: output shape = {out.shape}")

    print("\nAll tests passed!")
