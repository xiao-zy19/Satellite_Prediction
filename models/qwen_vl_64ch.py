"""
Modified Qwen2-VL model for 64-channel satellite embeddings.
Implements vision encoder modification for non-RGB input.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLVisionEmbeddings,
)
import logging

logger = logging.getLogger(__name__)


class Qwen2VL64Ch(nn.Module):
    """
    Qwen2-VL model modified for 64-channel input.

    The key modification is replacing the vision encoder's first layer
    from 3-channel to 64-channel input.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        input_channels: int = 64,
        patch_size: int = 14,
        init_new_channels: str = "normal",
        init_std: float = 0.02,
        use_flash_attention: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize modified Qwen2-VL model.

        Args:
            model_name: HuggingFace model name
            input_channels: Number of input channels (64 for satellite embeddings)
            patch_size: Patch size for vision encoder
            init_new_channels: How to initialize new channels ("normal", "zero", "copy")
            init_std: Standard deviation for normal initialization
            use_flash_attention: Whether to use flash attention
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
        """
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.model_name = model_name

        # Load base model configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Prepare loading arguments
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }

        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["device_map"] = "auto"

        # Load base model
        logger.info(f"Loading base model: {model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Modify vision encoder for 64 channels
        self._modify_vision_encoder(init_new_channels, init_std)

        logger.info(f"Model initialized with {input_channels} input channels")

    def _modify_vision_encoder(
        self,
        init_method: str = "normal",
        init_std: float = 0.02,
    ) -> None:
        """
        Modify the vision encoder to accept 64-channel input.

        The modification is done on the patch embedding layer.
        """
        # Access vision encoder
        vision_model = self.model.visual

        # Find the patch embedding layer
        # Qwen2-VL uses a specific architecture, need to find the right layer
        if hasattr(vision_model, 'patch_embed'):
            old_embed = vision_model.patch_embed
        elif hasattr(vision_model, 'embeddings'):
            old_embed = vision_model.embeddings.patch_embeddings
        else:
            # Try to find it in the model structure
            for name, module in vision_model.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    logger.info(f"Found patch embedding layer at: {name}")
                    old_embed = module
                    break
            else:
                raise ValueError("Could not find patch embedding layer")

        # Get old layer parameters
        out_channels = old_embed.out_channels
        kernel_size = old_embed.kernel_size
        stride = old_embed.stride
        padding = old_embed.padding
        bias = old_embed.bias is not None

        logger.info(f"Original patch embed: in_channels=3, out_channels={out_channels}")
        logger.info(f"Modifying to: in_channels={self.input_channels}")

        # Create new embedding layer
        new_embed = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Initialize weights
        with torch.no_grad():
            # Copy weights from original 3 channels
            new_embed.weight[:, :3, :, :] = old_embed.weight.clone()

            # Initialize remaining 61 channels
            if init_method == "normal":
                nn.init.normal_(new_embed.weight[:, 3:, :, :], std=init_std)
            elif init_method == "zero":
                nn.init.zeros_(new_embed.weight[:, 3:, :, :])
            elif init_method == "copy":
                # Copy and tile from first 3 channels
                for i in range(3, self.input_channels, 3):
                    end_idx = min(i + 3, self.input_channels)
                    copy_len = end_idx - i
                    new_embed.weight[:, i:end_idx, :, :] = old_embed.weight[:, :copy_len, :, :]

            # Copy bias if exists
            if bias:
                new_embed.bias.copy_(old_embed.bias)

        # Replace the layer
        # This depends on where the layer is in the model structure
        if hasattr(vision_model, 'patch_embed'):
            vision_model.patch_embed = new_embed
        elif hasattr(vision_model, 'embeddings'):
            vision_model.embeddings.patch_embeddings = new_embed
        else:
            # Set using the name we found
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = vision_model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, new_embed)

        logger.info("Vision encoder modified successfully")

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
            pixel_values: Image tensor of shape (B, 64, H, W)
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
            pixel_values: Image tensor of shape (1, 64, H, W)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to sample

        Returns:
            Generated text
        """
        # Format messages
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
        ).to(self.model.device)

        # Add pixel values
        inputs['pixel_values'] = pixel_values

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
        return self.model.visual

    def get_language_model(self) -> nn.Module:
        """Return the language model module."""
        return self.model.model

    def freeze_vision_encoder(self) -> None:
        """Freeze vision encoder parameters."""
        for param in self.model.visual.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen")

    def unfreeze_vision_encoder(self) -> None:
        """Unfreeze vision encoder parameters."""
        for param in self.model.visual.parameters():
            param.requires_grad = True
        logger.info("Vision encoder unfrozen")

    def freeze_language_model(self) -> None:
        """Freeze language model parameters."""
        for param in self.model.model.parameters():
            param.requires_grad = False
        logger.info("Language model frozen")

    def unfreeze_language_model(self) -> None:
        """Unfreeze language model parameters."""
        for param in self.model.model.parameters():
            param.requires_grad = True
        logger.info("Language model unfrozen")

    def save_pretrained(self, save_path: str) -> None:
        """Save the modified model."""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # Save modification info
        import json
        info = {
            'input_channels': self.input_channels,
            'patch_size': self.patch_size,
            'base_model': self.model_name,
        }
        with open(f"{save_path}/64ch_config.json", 'w') as f:
            json.dump(info, f)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, save_path: str) -> 'Qwen2VL64Ch':
        """Load a saved modified model."""
        import json

        # Load modification info
        with open(f"{save_path}/64ch_config.json", 'r') as f:
            info = json.load(f)

        # Create instance
        model = cls(
            model_name=save_path,  # Load from saved path
            input_channels=info['input_channels'],
            patch_size=info['patch_size'],
        )

        return model


def modify_vision_encoder_for_64ch(
    model: Qwen2VLForConditionalGeneration,
    input_channels: int = 64,
    init_std: float = 0.02,
) -> Qwen2VLForConditionalGeneration:
    """
    Standalone function to modify a Qwen2-VL model for 64-channel input.

    Args:
        model: Original Qwen2-VL model
        input_channels: Number of input channels
        init_std: Standard deviation for weight initialization

    Returns:
        Modified model
    """
    vision_model = model.visual

    # Find and replace patch embedding
    for name, module in vision_model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            # Create new layer
            new_layer = nn.Conv2d(
                in_channels=input_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
            )

            # Initialize
            with torch.no_grad():
                new_layer.weight[:, :3, :, :] = module.weight.clone()
                nn.init.normal_(new_layer.weight[:, 3:, :, :], std=init_std)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)

            # Replace
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = vision_model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, new_layer)

            logger.info(f"Modified {name}: 3ch -> {input_channels}ch")
            break

    return model


def load_pretrained_64ch_model(
    model_path: str,
    device: str = "cuda",
) -> Tuple[Qwen2VL64Ch, AutoProcessor]:
    """
    Load a pretrained 64-channel model.

    Args:
        model_path: Path to saved model
        device: Device to load on

    Returns:
        Tuple of (model, processor)
    """
    model = Qwen2VL64Ch.from_pretrained(model_path)
    model = model.to(device)
    return model, model.processor
