"""
Model definitions for population growth prediction
"""

from .mlp_model import MLPModel
from .light_cnn import LightCNN
from .resnet_baseline import ResNetBaseline

__all__ = ['MLPModel', 'LightCNN', 'ResNetBaseline']
