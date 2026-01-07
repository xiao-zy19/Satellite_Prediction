"""
Modified Qwen2.5-VL model for 64-channel satellite embeddings.
Implements vision encoder modification for non-RGB input.

关键区别 (vs Qwen2-VL):
1. Qwen2.5-VL 使用 Conv3d 作为 Patch Embedding (支持视频处理)
2. 模型类名: Qwen2_5_VLForConditionalGeneration
3. 视觉编码器访问路径: model.model.visual.patch_embed.proj
4. 使用 Window Attention + Full Attention 混合架构
5. 激活函数从 GELU 改为 SiLU (SwiGLU)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class Qwen25VL64Ch(nn.Module):
    """
    Qwen2.5-VL model modified for 64-channel input.

    The key modification is replacing the vision encoder's Conv3d patch embedding
    from 3-channel to 64-channel input.

    Qwen2.5-VL 视觉编码器结构:
    - patch_embed.proj: Conv3d(3, 1280, kernel=[2,14,14], stride=[2,14,14])
    - 32层 Transformer blocks (混合 window attention 和 full attention)
    - merger: 合并 patches
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        input_channels: int = 64,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        init_new_channels: str = "normal",
        init_std: float = 0.02,
        use_flash_attention: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize modified Qwen2.5-VL model.

        Args:
            model_name: HuggingFace model name (default: Qwen2.5-VL-7B)
            input_channels: Number of input channels (64 for satellite embeddings)
            patch_size: Spatial patch size for vision encoder (default: 14)
            temporal_patch_size: Temporal patch size for video (default: 2)
            init_new_channels: How to initialize new channels ("normal", "zero", "copy")
            init_std: Standard deviation for normal initialization
            use_flash_attention: Whether to use flash attention 2
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
            device_map: Device placement strategy
            torch_dtype: Model dtype
        """
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.model_name = model_name

        # Import Qwen2.5-VL specific classes
        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                AutoConfig,
            )
        except ImportError:
            raise ImportError(
                "transformers >= 4.45.0 is required for Qwen2.5-VL. "
                "Please upgrade: pip install --upgrade transformers"
            )

        # Load base model configuration
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Log vision config
        if hasattr(self.config, 'vision_config'):
            logger.info(f"Vision config: {self.config.vision_config}")

        # Prepare loading arguments
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["device_map"] = device_map

        # Load base model
        logger.info(f"Loading base model: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Modify vision encoder for 64 channels
        self._modify_vision_encoder(init_new_channels, init_std)

        logger.info(f"Model initialized with {input_channels} input channels")

    def _get_vision_model(self) -> nn.Module:
        """获取视觉编码器模块"""
        # Qwen2.5-VL 结构: model.model.visual
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            return self.model.model.visual
        elif hasattr(self.model, 'visual'):
            return self.model.visual
        else:
            raise AttributeError("Cannot find visual encoder in the model")

    def _modify_vision_encoder(
        self,
        init_method: str = "normal",
        init_std: float = 0.02,
    ) -> None:
        """
        Modify the vision encoder to accept 64-channel input.

        Qwen2.5-VL 使用 Conv3d 作为 patch embedding:
        - 原始: Conv3d(3, embed_dim, kernel=[temporal, patch, patch])
        - 修改: Conv3d(64, embed_dim, kernel=[temporal, patch, patch])
        """
        vision_model = self._get_vision_model()

        # 找到 patch_embed.proj (Conv3d)
        if hasattr(vision_model, 'patch_embed') and hasattr(vision_model.patch_embed, 'proj'):
            old_proj = vision_model.patch_embed.proj

            if not isinstance(old_proj, nn.Conv3d):
                logger.warning(f"Expected Conv3d but found {type(old_proj)}. Trying alternative approach...")
                self._modify_vision_encoder_fallback(init_method, init_std)
                return

            logger.info(f"Found patch embedding Conv3d:")
            logger.info(f"  Original: in_channels={old_proj.in_channels}, "
                       f"out_channels={old_proj.out_channels}, "
                       f"kernel_size={old_proj.kernel_size}")

            # Create new Conv3d with 64 input channels
            new_proj = nn.Conv3d(
                in_channels=self.input_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None,
            )

            # Initialize weights
            self._initialize_new_conv3d_weights(
                new_proj, old_proj, init_method, init_std
            )

            # Replace the layer
            vision_model.patch_embed.proj = new_proj

            # Update patch_embed's in_channels attribute
            if hasattr(vision_model.patch_embed, 'in_channels'):
                vision_model.patch_embed.in_channels = self.input_channels

            logger.info(f"  Modified: in_channels={self.input_channels}")
            logger.info("Vision encoder modified successfully (Conv3d)")

        else:
            logger.warning("Could not find patch_embed.proj, trying fallback method...")
            self._modify_vision_encoder_fallback(init_method, init_std)

    def _modify_vision_encoder_fallback(
        self,
        init_method: str = "normal",
        init_std: float = 0.02,
    ) -> None:
        """
        Fallback method to find and modify the patch embedding layer.
        遍历模型查找 Conv3d 或 Conv2d 层。
        """
        vision_model = self._get_vision_model()
        modified = False

        for name, module in vision_model.named_modules():
            # 查找输入通道为 3 的卷积层
            if isinstance(module, nn.Conv3d) and module.in_channels == 3:
                logger.info(f"Found Conv3d at: {name}")
                new_conv = nn.Conv3d(
                    in_channels=self.input_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None,
                )
                self._initialize_new_conv3d_weights(new_conv, module, init_method, init_std)

                # 替换模块
                self._set_module_by_name(vision_model, name, new_conv)
                logger.info(f"Modified {name}: 3ch -> {self.input_channels}ch (Conv3d)")
                modified = True
                break

            elif isinstance(module, nn.Conv2d) and module.in_channels == 3:
                logger.info(f"Found Conv2d at: {name}")
                new_conv = nn.Conv2d(
                    in_channels=self.input_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None,
                )
                self._initialize_new_conv2d_weights(new_conv, module, init_method, init_std)

                self._set_module_by_name(vision_model, name, new_conv)
                logger.info(f"Modified {name}: 3ch -> {self.input_channels}ch (Conv2d)")
                modified = True
                break

        if not modified:
            raise ValueError("Could not find patch embedding layer to modify")

    def _initialize_new_conv3d_weights(
        self,
        new_conv: nn.Conv3d,
        old_conv: nn.Conv3d,
        init_method: str,
        init_std: float,
    ) -> None:
        """Initialize weights for new Conv3d layer."""
        with torch.no_grad():
            # Copy weights from original 3 channels
            # Conv3d weight shape: (out_channels, in_channels, D, H, W)
            new_conv.weight[:, :3, :, :, :] = old_conv.weight.clone()

            # Initialize remaining 61 channels
            if init_method == "normal":
                nn.init.normal_(new_conv.weight[:, 3:, :, :, :], std=init_std)
            elif init_method == "zero":
                nn.init.zeros_(new_conv.weight[:, 3:, :, :, :])
            elif init_method == "copy":
                # Copy and tile from first 3 channels
                for i in range(3, self.input_channels, 3):
                    end_idx = min(i + 3, self.input_channels)
                    copy_len = end_idx - i
                    new_conv.weight[:, i:end_idx, :, :, :] = old_conv.weight[:, :copy_len, :, :, :]
            else:
                raise ValueError(f"Unknown init method: {init_method}")

            # Copy bias if exists
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

    def _initialize_new_conv2d_weights(
        self,
        new_conv: nn.Conv2d,
        old_conv: nn.Conv2d,
        init_method: str,
        init_std: float,
    ) -> None:
        """Initialize weights for new Conv2d layer (fallback)."""
        with torch.no_grad():
            # Conv2d weight shape: (out_channels, in_channels, H, W)
            new_conv.weight[:, :3, :, :] = old_conv.weight.clone()

            if init_method == "normal":
                nn.init.normal_(new_conv.weight[:, 3:, :, :], std=init_std)
            elif init_method == "zero":
                nn.init.zeros_(new_conv.weight[:, 3:, :, :])
            elif init_method == "copy":
                for i in range(3, self.input_channels, 3):
                    end_idx = min(i + 3, self.input_channels)
                    copy_len = end_idx - i
                    new_conv.weight[:, i:end_idx, :, :] = old_conv.weight[:, :copy_len, :, :]

            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

    def _set_module_by_name(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
        """Set a module by its dotted name path."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: Image tensor of shape (B, 64, T, H, W) for video
                         or processed by processor for images
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments for the base model

        Returns:
            Model outputs
        """
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text response for given image and prompt.

        Args:
            pixel_values: Image tensor
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to sample

        Returns:
            Generated text
        """
        # Format messages for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.processor(
            text=[text],
            return_tensors="pt"
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Add pixel values
        inputs['pixel_values'] = pixel_values.to(device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )

        # Decode
        generated = self.processor.decode(outputs[0], skip_special_tokens=True)

        return generated

    def get_vision_encoder(self) -> nn.Module:
        """Return the vision encoder module."""
        return self._get_vision_model()

    def get_language_model(self) -> nn.Module:
        """Return the language model module."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            return self.model.model.language_model
        return self.model.model

    def freeze_vision_encoder(self) -> None:
        """Freeze vision encoder parameters."""
        vision_model = self._get_vision_model()
        for param in vision_model.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen")

    def unfreeze_vision_encoder(self) -> None:
        """Unfreeze vision encoder parameters."""
        vision_model = self._get_vision_model()
        for param in vision_model.parameters():
            param.requires_grad = True
        logger.info("Vision encoder unfrozen")

    def freeze_language_model(self) -> None:
        """Freeze language model parameters."""
        lm = self.get_language_model()
        for param in lm.parameters():
            param.requires_grad = False
        logger.info("Language model frozen")

    def unfreeze_language_model(self) -> None:
        """Unfreeze language model parameters."""
        lm = self.get_language_model()
        for param in lm.parameters():
            param.requires_grad = True
        logger.info("Language model unfrozen")

    def save_pretrained(self, save_path: str) -> None:
        """Save the modified model."""
        import json
        from pathlib import Path

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # Save modification info
        info = {
            'input_channels': self.input_channels,
            'patch_size': self.patch_size,
            'temporal_patch_size': self.temporal_patch_size,
            'base_model': self.model_name,
            'model_type': 'qwen2.5-vl-64ch',
        }
        with open(save_path / '64ch_config.json', 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, save_path: str, **kwargs) -> 'Qwen25VL64Ch':
        """Load a saved modified model."""
        import json
        from pathlib import Path

        save_path = Path(save_path)

        # Load modification info
        with open(save_path / '64ch_config.json', 'r') as f:
            info = json.load(f)

        # Create instance
        model = cls(
            model_name=str(save_path),
            input_channels=info['input_channels'],
            patch_size=info['patch_size'],
            temporal_patch_size=info.get('temporal_patch_size', 2),
            **kwargs
        )

        return model


def modify_qwen25_vision_encoder_for_64ch(
    model,
    input_channels: int = 64,
    init_std: float = 0.02,
):
    """
    Standalone function to modify a Qwen2.5-VL model for 64-channel input.

    Args:
        model: Original Qwen2.5-VL model (Qwen2_5_VLForConditionalGeneration)
        input_channels: Number of input channels
        init_std: Standard deviation for weight initialization

    Returns:
        Modified model
    """
    # Get vision model
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        vision_model = model.model.visual
    elif hasattr(model, 'visual'):
        vision_model = model.visual
    else:
        raise AttributeError("Cannot find visual encoder")

    # Find and modify patch embedding
    if hasattr(vision_model, 'patch_embed') and hasattr(vision_model.patch_embed, 'proj'):
        old_proj = vision_model.patch_embed.proj

        if isinstance(old_proj, nn.Conv3d):
            new_proj = nn.Conv3d(
                in_channels=input_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None,
            )

            with torch.no_grad():
                new_proj.weight[:, :3, :, :, :] = old_proj.weight.clone()
                nn.init.normal_(new_proj.weight[:, 3:, :, :, :], std=init_std)
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)

            vision_model.patch_embed.proj = new_proj
            if hasattr(vision_model.patch_embed, 'in_channels'):
                vision_model.patch_embed.in_channels = input_channels

            logger.info(f"Modified patch_embed.proj: 3ch -> {input_channels}ch (Conv3d)")
    else:
        # Fallback: search for Conv layers
        for name, module in vision_model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d)) and module.in_channels == 3:
                if isinstance(module, nn.Conv3d):
                    new_layer = nn.Conv3d(
                        input_channels, module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        bias=module.bias is not None,
                    )
                    with torch.no_grad():
                        new_layer.weight[:, :3, :, :, :] = module.weight.clone()
                        nn.init.normal_(new_layer.weight[:, 3:, :, :, :], std=init_std)
                else:
                    new_layer = nn.Conv2d(
                        input_channels, module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        bias=module.bias is not None,
                    )
                    with torch.no_grad():
                        new_layer.weight[:, :3, :, :] = module.weight.clone()
                        nn.init.normal_(new_layer.weight[:, 3:, :, :], std=init_std)

                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)

                # Replace
                parts = name.split('.')
                parent = vision_model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_layer)

                logger.info(f"Modified {name}: 3ch -> {input_channels}ch")
                break

    return model


def load_pretrained_64ch_model(
    model_path: str,
    device: str = "cuda",
) -> Tuple['Qwen25VL64Ch', Any]:
    """
    Load a pretrained 64-channel model.

    Args:
        model_path: Path to saved model
        device: Device to load on

    Returns:
        Tuple of (model, processor)
    """
    model = Qwen25VL64Ch.from_pretrained(model_path)
    if device != "auto":
        model = model.to(device)
    return model, model.processor


# ============ 测试代码 ============
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Qwen2.5-VL 64-channel modification")
    print("=" * 60)

    # 仅测试配置加载
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        print("\n✓ Config loaded successfully")
        print(f"Vision config: {config.vision_config}")
    except Exception as e:
        print(f"\n✗ Config loading failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("To fully test the model, run:")
    print("  model = Qwen25VL64Ch()")
    print("=" * 60)
