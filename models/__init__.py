"""
Model definitions for population growth prediction
"""

from .mlp_model import MLPModel
from .light_cnn import LightCNN
from .resnet_baseline import ResNetBaseline
from .aggregators import (
    get_aggregator,
    AttentionAggregator,
    PositionalAttentionAggregator,
    SpatialPositionAggregator,
    TransformerAggregator,
    Transformer2DAggregator
)

__all__ = [
    'MLPModel',
    'LightCNN',
    'ResNetBaseline',
    'get_aggregator',
    'AttentionAggregator',
    'PositionalAttentionAggregator',
    'SpatialPositionAggregator',
    'TransformerAggregator',
    'Transformer2DAggregator'
]
