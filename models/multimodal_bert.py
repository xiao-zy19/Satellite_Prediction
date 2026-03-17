"""
Multimodal BERT model for population growth prediction.

This module re-exports the original MultiModalModel from multimodal.py,
which already supports different policy feature dimensions.

The key difference is that BERT/Hybrid policy features have different dimensions:
- Structured: 12-dim
- BERT: 64-dim (configurable)
- Hybrid: 76-dim (64 + 12)

The original model's policy_feature_dim parameter handles this automatically.
"""

import torch
import torch.nn as nn

# Re-export original components
from models.multimodal import (
    MultiModalModel,
    PolicyEncoder,
    ConcatFusion,
    GatedFusion,
    AttentionFusion,
    FiLMFusion,
    get_fusion_layer,
)


def create_bert_multimodal_model(model_config, patch_level: bool = False):
    """
    Create multimodal model for BERT/Hybrid policy features.

    This is a convenience wrapper that extracts the correct policy_feature_dim
    from the BERT model config. When BERT/Hybrid features are used (768-dim),
    it automatically enables a trainable policy encoder (projection layer)
    so that the 768→64 projection is trained end-to-end.

    Args:
        model_config: MultiModalBertConfig instance
        patch_level: Whether this is patch-level training

    Returns:
        MultiModalModel instance configured for BERT/Hybrid features
    """
    # Get policy dimension from config
    policy_dim = model_config.policy_feature_dim  # This uses the @property

    # Determine policy encoder settings
    use_policy_encoder = getattr(model_config, 'use_policy_encoder', False)
    policy_hidden_dim = model_config.policy_hidden_dim
    policy_output_dim = getattr(model_config, 'policy_output_dim', None)

    # Auto-enable trainable projection for BERT/hybrid (768-dim or 780-dim input)
    # The projection layer (768→hidden→64) is trained end-to-end with the model.
    if policy_dim >= 768 and not use_policy_encoder:
        use_policy_encoder = True
        if policy_hidden_dim < 128:  # Default value too small for 768-dim input
            policy_hidden_dim = 256
        if policy_output_dim is None or policy_output_dim >= 768:
            policy_output_dim = 64
        print(f"[BERT E2E] Auto-enabled trainable policy projection: "
              f"{policy_dim}→{policy_hidden_dim}→{policy_output_dim}")

    return MultiModalModel(
        image_encoder_type=model_config.image_encoder_type,
        fusion_type=model_config.fusion_type,
        policy_feature_dim=policy_dim,
        policy_hidden_dim=policy_hidden_dim,
        policy_output_dim=policy_output_dim,
        image_feature_dim=getattr(model_config, 'image_feature_dim', 64),
        aggregation_type=model_config.aggregation,
        dropout=model_config.dropout_rate,
        patch_level=patch_level,
        use_policy_encoder=use_policy_encoder,
        model_config=model_config
    )


# Alias for backward compatibility
BertMultiModalModel = MultiModalModel


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config_multimodal_bert import get_bert_multimodal_experiment_config

    # Test with different policy sources
    batch_size = 2
    num_patches = 25
    channels = 64
    patch_size = 200

    x_img = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)
    x_img_single = torch.randn(batch_size, channels, patch_size, patch_size)

    print("=" * 60)
    print("Testing BERT policy (64-dim)")
    print("=" * 60)

    exp_bert = get_bert_multimodal_experiment_config("bert_cnn_concat")
    model_bert = create_bert_multimodal_model(exp_bert.model_config, patch_level=True)
    x_policy_bert = torch.randn(batch_size, 64)

    print(f"Policy dim: {exp_bert.model_config.policy_feature_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model_bert.parameters()):,}")

    with torch.no_grad():
        output = model_bert(x_img_single, x_policy_bert)
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("Testing Hybrid policy (76-dim)")
    print("=" * 60)

    exp_hybrid = get_bert_multimodal_experiment_config("hybrid_cnn_concat")
    model_hybrid = create_bert_multimodal_model(exp_hybrid.model_config, patch_level=True)
    x_policy_hybrid = torch.randn(batch_size, 76)

    print(f"Policy dim: {exp_hybrid.model_config.policy_feature_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model_hybrid.parameters()):,}")

    with torch.no_grad():
        output = model_hybrid(x_img_single, x_policy_hybrid)
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("Testing different fusion types with BERT")
    print("=" * 60)

    for fusion_type in ["concat", "gated", "attention", "film"]:
        model_test = MultiModalModel(
            image_encoder_type="light_cnn",
            fusion_type=fusion_type,
            policy_feature_dim=64,  # BERT dim
            image_feature_dim=64,
            patch_level=True,
            use_policy_encoder=False
        )
        params = sum(p.numel() for p in model_test.parameters())
        print(f"{fusion_type:12s}: {params:,} params")

    print("\n" + "=" * 60)
    print("City-level mode test")
    print("=" * 60)

    model_city = create_bert_multimodal_model(exp_bert.model_config, patch_level=False)
    print(f"Model parameters: {sum(p.numel() for p in model_city.parameters()):,}")

    with torch.no_grad():
        output_city = model_city(x_img, x_policy_bert)
    print(f"Input: image={x_img.shape}, policy={x_policy_bert.shape}")
    print(f"Output: {output_city.shape}")
