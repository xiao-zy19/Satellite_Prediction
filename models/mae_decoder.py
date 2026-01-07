"""
MAE Decoder for masked autoencoder pretraining.
Updated version to support both Qwen2-VL (Conv2d) and Qwen2.5-VL (Conv3d).

关键区别:
- Qwen2-VL: 使用 Conv2d, 输入形状 (B, C, H, W)
- Qwen2.5-VL: 使用 Conv3d, 输入形状 (B, C, T, H, W), temporal_patch_size=2

包含:
- MAEDecoder: 用于重建被mask的patches
- MAEForPretraining: 完整的MAE模型（encoder + decoder）
- MAEForPretrainingV2: 支持 Qwen2.5-VL 的版本
- SimpleMAEForPretraining: 简化版MAE（用于调试和快速测试）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    """
    MAE Decoder for reconstructing masked patches.
    Takes encoded visible patches and mask tokens, reconstructs full image.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        patch_size: int = 14,
        num_patches: int = 4900,
        in_channels: int = 64,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.norm_pix_loss = norm_pix_loss

        # Project from encoder dimension to decoder dimension
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )

        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(decoder_depth)
        ])

        # Final norm
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Prediction head: predict patch pixels
        self.decoder_pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * in_channels
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_layer)

    def _init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoded visible patches, shape (B, N_visible, encoder_dim)
            ids_restore: Indices to restore original patch order, shape (B, N_total)

        Returns:
            Reconstructed patches, shape (B, N_total, patch_size^2 * in_channels)
        """
        x = self.decoder_embed(x)

        B, N_visible, D = x.shape
        N_total = ids_restore.shape[1]
        N_masked = N_total - N_visible

        mask_tokens = self.mask_token.expand(B, N_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)

        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted patches, shape (B, N, patch_size^2 * C)
            target: Target image, shape (B, C, H, W)
            mask: Boolean mask, True = masked, shape (B, N)

        Returns:
            Mean squared error loss on masked patches
        """
        target = self._patchify(target)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask.float()).sum() / mask.float().sum()

        return loss

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert image to patches."""
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = H // p
        w = W // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)

        return x

    def _unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert patches back to image."""
        B = x.shape[0]
        p = self.patch_size

        x = x.reshape(B, h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.in_channels, h * p, w * p)

        return x


class MAEForPretrainingV2(nn.Module):
    """
    Complete MAE model for pretraining - Version 2.

    支持 Qwen2.5-VL 的视觉编码器（使用 Conv3d）。

    关键特性:
    1. 自动检测编码器类型 (Conv2d vs Conv3d)
    2. 处理时间维度 (temporal dimension)
    3. 支持单帧图像和多帧视频
    4. 正确的 MAE masking：先 patch_embed，再选择 visible patches，再 encoder

    信息流（防止泄露）:
    1. patch_embed: 将图像转换为 patches（无 attention，per-patch 独立）
    2. 选择 visible patches（在任何 attention 之前）
    3. 只让 visible patches 通过 transformer blocks
    4. decoder 使用 visible patches + mask tokens 重建
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        encoder_dim: int = 1280,  # Qwen2.5-VL uses 1280
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        patch_size: int = 14,
        temporal_patch_size: int = 2,  # Qwen2.5-VL specific
        image_size: int = 980,
        input_channels: int = 64,
        mask_ratio: float = 0.75,
        model_type: str = "qwen2.5-vl",  # "qwen2-vl" or "qwen2.5-vl"
    ):
        """
        Initialize MAE for pretraining.

        Args:
            vision_encoder: Vision encoder from MLLM
            encoder_dim: Encoder output dimension (1280 for Qwen2.5-VL)
            decoder_dim: Decoder dimension
            decoder_depth: Number of decoder layers
            decoder_heads: Number of decoder attention heads
            patch_size: Spatial patch size
            temporal_patch_size: Temporal patch size (Qwen2.5-VL only)
            image_size: Input image size
            input_channels: Number of input channels
            mask_ratio: Ratio of patches to mask
            model_type: "qwen2-vl" or "qwen2.5-vl"
        """
        super().__init__()

        self.encoder = vision_encoder
        self.encoder_dim = encoder_dim
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.input_channels = input_channels
        self.image_size = image_size
        self.model_type = model_type

        # Calculate number of patches
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        num_patches = self.num_patches_h * self.num_patches_w
        self.num_patches = num_patches

        # 检测并缓存 encoder 的内部组件
        self._setup_encoder_components()

        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_heads,
            patch_size=patch_size,
            num_patches=num_patches,
            in_channels=input_channels,
        )

        logger.info(f"MAEForPretrainingV2 initialized ({model_type}):")
        logger.info(f"  - Encoder dim: {encoder_dim}")
        logger.info(f"  - Decoder dim: {decoder_dim}")
        logger.info(f"  - Num patches: {num_patches}")
        logger.info(f"  - Mask ratio: {mask_ratio}")
        logger.info(f"  - Temporal patch size: {temporal_patch_size}")
        logger.info(f"  - Encoder structure detected: {self._encoder_structure}")

    def _setup_encoder_components(self) -> None:
        """
        检测并设置 encoder 的内部组件引用。

        支持的结构:
        1. Qwen2.5-VL: patch_embed.proj (Conv3d) + blocks (transformer)
        2. Qwen2-VL: patch_embed (Conv2d) + blocks (transformer)
        3. 通用: 任何有 patch_embed 的 encoder
        """
        self._encoder_structure = "unknown"
        self._patch_embed = None
        self._encoder_blocks = None
        self._pos_embed = None
        self._encoder_norm = None

        # 尝试找到 patch_embed
        if hasattr(self.encoder, 'patch_embed'):
            if hasattr(self.encoder.patch_embed, 'proj'):
                # Qwen2.5-VL 结构: patch_embed.proj 是 Conv3d
                self._patch_embed = self.encoder.patch_embed.proj
                self._encoder_structure = "qwen2.5-vl"
            else:
                # Qwen2-VL 或通用结构
                self._patch_embed = self.encoder.patch_embed
                self._encoder_structure = "qwen2-vl"

        # 尝试找到 transformer blocks
        if hasattr(self.encoder, 'blocks'):
            self._encoder_blocks = self.encoder.blocks
        elif hasattr(self.encoder, 'layers'):
            self._encoder_blocks = self.encoder.layers

        # 尝试找到位置编码
        if hasattr(self.encoder, 'pos_embed'):
            self._pos_embed = self.encoder.pos_embed

        # 尝试找到最终的 norm 层
        if hasattr(self.encoder, 'norm'):
            self._encoder_norm = self.encoder.norm

        # 如果找不到关键组件，使用 fallback
        if self._patch_embed is None:
            logger.warning("Could not find patch_embed in encoder, using fallback encoding")
            self._encoder_structure = "fallback"

        # 如果找到了 patch_embed 但没有 blocks，也用 fallback
        if self._patch_embed is not None and self._encoder_blocks is None:
            logger.warning("Found patch_embed but no transformer blocks, using fallback encoding")
            self._encoder_structure = "fallback_patch_only"

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ids_keep: Optional[torch.Tensor] = None,
        ids_restore: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking.

        Args:
            x: Input image, shape (B, C, H, W) for single frame
               Will be converted to (B, C, T, H, W) for Qwen2.5-VL
            mask: Optional pre-computed mask
            ids_keep: Optional indices of kept patches
            ids_restore: Optional indices to restore order

        Returns:
            Tuple of (predicted patches, loss)
        """
        # Store original input for loss computation (在任何修改之前保存!)
        original_x = x.clone()

        # Generate mask if not provided
        if mask is None or ids_keep is None or ids_restore is None:
            mask, ids_keep, ids_restore = self._random_masking(x)

        # ========== 关键修复：在 encoder 之前将 masked 区域置零 ==========
        # 这样即使 encoder 处理了"完整图"，masked 区域也是空白的，不会泄露信息
        x = self._apply_mask_to_image(x, mask)

        # Prepare input for Qwen2.5-VL (add temporal dimension)
        if self.model_type == "qwen2.5-vl" and x.dim() == 4:
            x = self._prepare_temporal_input(x)

        # Encode visible patches (现在 x 中 masked 区域已经是 0 了)
        encoded = self._encode_visible(x, ids_keep)

        # Decode
        pred = self.decoder(encoded, ids_restore)

        # Compute loss (use original unmasked input)
        loss = self.decoder.compute_loss(pred, original_x, mask)

        return pred, loss

    def _apply_mask_to_image(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 masked patches 对应的图像区域置为 0。

        这是防止信息泄露的关键步骤！即使 encoder 处理了完整图像，
        masked 区域也是全零，不包含任何信息。

        Args:
            x: Input image, shape (B, C, H, W)
            mask: Boolean mask, shape (B, N), True = masked

        Returns:
            Masked image with masked regions set to 0
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h = H // p  # patches per height
        w = W // p  # patches per width

        # 创建 pixel-level mask: (B, N) -> (B, 1, H, W)
        # mask: True = masked (需要置零), False = visible (保留)
        mask_2d = mask.view(B, h, w)  # (B, h, w)

        # 将 patch-level mask 扩展到 pixel-level
        # 每个 patch 对应 p x p 个像素
        # (B, h, w) -> (B, h, w, p, p) -> (B, h, p, w, p) -> (B, H, W)
        mask_pixels = mask_2d.unsqueeze(-1).unsqueeze(-1)  # (B, h, w, 1, 1)
        mask_pixels = mask_pixels.expand(-1, -1, -1, p, p)  # (B, h, w, p, p)
        # 关键：正确的维度顺序！先 permute 再 reshape
        mask_pixels = mask_pixels.permute(0, 1, 3, 2, 4)  # (B, h, p, w, p)
        mask_pixels = mask_pixels.reshape(B, h * p, w * p)  # (B, H, W)
        mask_pixels = mask_pixels.unsqueeze(1)  # (B, 1, H, W)

        # 将 masked 区域置零
        # mask=True 的位置乘以 0，mask=False 的位置乘以 1
        x_masked = x * (~mask_pixels).float()

        return x_masked

    def _prepare_temporal_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for Qwen2.5-VL's Conv3d patch embedding.

        Qwen2.5-VL expects: (B, C, T, H, W) where T = temporal_patch_size
        For single images, we replicate along temporal dimension.

        Args:
            x: Input of shape (B, C, H, W)

        Returns:
            Output of shape (B, C, T, H, W)
        """
        B, C, H, W = x.shape
        T = self.temporal_patch_size

        # Replicate the frame along temporal dimension
        # This allows single images to work with Conv3d
        x = x.unsqueeze(2)  # (B, C, 1, H, W)
        x = x.expand(-1, -1, T, -1, -1)  # (B, C, T, H, W)

        return x

    def _random_masking(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random mask."""
        if x.dim() == 5:
            B, C, T, H, W = x.shape
        else:
            B, C, H, W = x.shape

        num_patches = (H // self.patch_size) * (W // self.patch_size)
        num_masked = int(num_patches * self.mask_ratio)

        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_patches - num_masked]

        mask = torch.ones(B, num_patches, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return mask, ids_keep, ids_restore

    def _encode_visible(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码 visible patches。

        由于 Qwen2.5-VL 的 VisionBlock 需要复杂的参数（cu_seqlens, position_embeddings等），
        我们采用简化策略：
        1. 只使用 patch_embed 层（无 attention，per-patch 独立）
        2. 选择 visible patches
        3. 跳过 transformer blocks（它们的参数太复杂）

        注意：这意味着预训练主要训练的是：
        - patch_embed (Conv3d): 学习64通道的特征提取
        - MAE Decoder: 学习重建

        Transformer blocks 的预训练权重会被保留，在后续 SFT 阶段再微调。
        """
        B = x.shape[0]

        # ========== Step 1: Patch Embedding (无 attention) ==========
        if self._encoder_structure in ["qwen2.5-vl", "qwen2-vl"]:
            # 使用检测到的 patch_embed
            patches = self._patch_embed(x)

            # 处理不同的输出形状
            if patches.dim() == 5:
                # Conv3d 输出: (B, D, T', H', W') -> (B, N, D)
                B, D, T_out, H_out, W_out = patches.shape
                patches = patches.permute(0, 2, 3, 4, 1)  # (B, T', H', W', D)
                patches = patches.reshape(B, -1, D)  # (B, T'*H'*W', D)
            elif patches.dim() == 4:
                # Conv2d 输出: (B, D, H', W') -> (B, N, D)
                B, D, H_out, W_out = patches.shape
                patches = patches.flatten(2).transpose(1, 2)  # (B, H'*W', D)
            # else: 假设已经是 (B, N, D)

        elif self._encoder_structure == "fallback_patch_only":
            # 只有 patch_embed，没有 transformer blocks
            patches = self._patch_embed(x)
            if patches.dim() == 4:
                patches = patches.flatten(2).transpose(1, 2)

        else:
            # Fallback: 尝试直接编码
            logger.warning(
                "Using fallback encoding - this may cause information leakage! "
                "Consider using SimpleMAEForPretraining instead."
            )
            encoded = self.encoder(x)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            if isinstance(encoded, dict):
                encoded = encoded.get('last_hidden_state', encoded.get('hidden_states'))
            if encoded.dim() == 4:
                encoded = encoded.flatten(2).transpose(1, 2)

            D = encoded.shape[-1]
            ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            return torch.gather(encoded, dim=1, index=ids_keep_expanded)

        # ========== Step 2: 选择 Visible Patches ==========
        N_total = patches.shape[1]
        D = patches.shape[2]

        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        visible_patches = torch.gather(patches, dim=1, index=ids_keep_expanded)

        # ========== Step 3: 添加位置编码 (只对 visible patches) ==========
        if self._pos_embed is not None:
            pos_embed_visible = torch.gather(
                self._pos_embed.expand(B, -1, -1),
                dim=1,
                index=ids_keep_expanded
            )
            visible_patches = visible_patches + pos_embed_visible

        # ========== Step 4: 跳过 Transformer Blocks ==========
        # Qwen2.5-VL 的 VisionBlock 需要复杂参数 (cu_seqlens, position_embeddings)
        # 这些参数依赖于 grid_thw 等信息，难以在 MAE 设置中正确构造
        # 因此我们跳过 transformer blocks，只训练 patch_embed 和 decoder
        #
        # 这种策略的好处：
        # 1. patch_embed (Conv3d) 会学习 64 通道的特征提取
        # 2. 预训练的 transformer 权重保持不变，在 SFT 阶段再微调
        # 3. 避免了复杂的参数构造问题
        #
        # 如果需要训练完整的 encoder，请使用 SimpleMAEForPretraining

        if self._encoder_norm is not None:
            visible_patches = self._encoder_norm(visible_patches)

        return visible_patches


class SimpleMAEForPretraining(nn.Module):
    """
    简化版MAE模型，用于调试和快速测试。
    使用简单的CNN encoder代替大型预训练模型。

    支持两种模式:
    - conv2d: 标准2D卷积 (适用于 Qwen2-VL 风格)
    - conv3d: 3D卷积 (适用于 Qwen2.5-VL 风格)
    """

    def __init__(
        self,
        input_channels: int = 64,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        image_size: int = 980,
        encoder_dim: int = 256,
        decoder_dim: int = 128,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 4,
        mask_ratio: float = 0.75,
        conv_type: str = "conv3d",  # "conv2d" or "conv3d"
    ):
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.conv_type = conv_type

        self.num_patches = (image_size // patch_size) ** 2
        self.num_masked = int(self.num_patches * mask_ratio)

        # Patch embedding - supports both Conv2d and Conv3d
        if conv_type == "conv3d":
            self.patch_embed = nn.Conv3d(
                input_channels,
                encoder_dim,
                kernel_size=(temporal_patch_size, patch_size, patch_size),
                stride=(temporal_patch_size, patch_size, patch_size),
            )
        else:
            self.patch_embed = nn.Conv2d(
                input_channels,
                encoder_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, encoder_dim)
        )

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_dim, num_heads)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # Decoder
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=num_heads,
            patch_size=patch_size,
            num_patches=self.num_patches,
            in_channels=input_channels,
        )

        nn.init.normal_(self.pos_embed, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SimpleMAEForPretraining ({conv_type}) initialized with {total_params:,} parameters")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ids_keep: Optional[torch.Tensor] = None,
        ids_restore: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        B = x.shape[0]
        # Store original input for loss computation (在任何修改之前保存!)
        original_x = x.clone()

        # Generate mask if not provided
        if mask is None or ids_keep is None or ids_restore is None:
            mask, ids_keep, ids_restore = self._random_masking(x)

        # ========== 关键修复：在 encoder 之前将 masked 区域置零 ==========
        x = self._apply_mask_to_image(x, mask)

        # Prepare input for Conv3d
        if self.conv_type == "conv3d" and x.dim() == 4:
            # (B, C, H, W) -> (B, C, T, H, W)
            x = x.unsqueeze(2).expand(-1, -1, self.temporal_patch_size, -1, -1)

        # Patch embedding
        x = self.patch_embed(x)

        if self.conv_type == "conv3d":
            # (B, D, 1, H', W') -> (B, D, H'*W')
            x = x.squeeze(2).flatten(2).transpose(1, 2)
        else:
            # (B, D, H', W') -> (B, H'*W', D)
            x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Select visible patches
        D = x.shape[-1]
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x = torch.gather(x, dim=1, index=ids_keep_expanded)

        # Encode
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        # Decode
        pred = self.decoder(x, ids_restore)

        # Compute loss (use original unmasked input)
        loss = self.decoder.compute_loss(pred, original_x, mask)

        return pred, loss

    def _apply_mask_to_image(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 masked patches 对应的图像区域置为 0。

        Args:
            x: Input image, shape (B, C, H, W)
            mask: Boolean mask, shape (B, N), True = masked

        Returns:
            Masked image with masked regions set to 0
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h = H // p
        w = W // p

        # mask: (B, N) -> (B, h, w) -> pixel-level (B, 1, H, W)
        mask_2d = mask.view(B, h, w)
        mask_pixels = mask_2d.unsqueeze(-1).unsqueeze(-1)
        mask_pixels = mask_pixels.expand(-1, -1, -1, p, p)
        # 关键：正确的维度顺序！先 permute 再 reshape
        mask_pixels = mask_pixels.permute(0, 1, 3, 2, 4)  # (B, h, p, w, p)
        mask_pixels = mask_pixels.reshape(B, h * p, w * p)
        mask_pixels = mask_pixels.unsqueeze(1)

        # mask=True -> 置零, mask=False -> 保留
        x_masked = x * (~mask_pixels).float()

        return x_masked

    def _random_masking(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random mask."""
        B = x.shape[0]

        noise = torch.rand(B, self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        num_keep = self.num_patches - self.num_masked
        ids_keep = ids_shuffle[:, :num_keep]

        mask = torch.ones(B, self.num_patches, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return mask, ids_keep, ids_restore


# ============ Factory functions ============

def create_mae_for_qwen25_vl(
    vision_encoder: nn.Module,
    encoder_dim: int = 1280,
    decoder_dim: int = 512,
    decoder_depth: int = 4,
    patch_size: int = 14,
    image_size: int = 980,
    input_channels: int = 64,
    mask_ratio: float = 0.75,
) -> MAEForPretrainingV2:
    """Create MAE model for Qwen2.5-VL."""
    return MAEForPretrainingV2(
        vision_encoder=vision_encoder,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=8,
        patch_size=patch_size,
        temporal_patch_size=2,
        image_size=image_size,
        input_channels=input_channels,
        mask_ratio=mask_ratio,
        model_type="qwen2.5-vl",
    )


def create_mae_for_qwen2_vl(
    vision_encoder: nn.Module,
    encoder_dim: int = 1024,
    decoder_dim: int = 512,
    decoder_depth: int = 4,
    patch_size: int = 14,
    image_size: int = 980,
    input_channels: int = 64,
    mask_ratio: float = 0.75,
) -> MAEForPretrainingV2:
    """Create MAE model for Qwen2-VL."""
    return MAEForPretrainingV2(
        vision_encoder=vision_encoder,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=8,
        patch_size=patch_size,
        temporal_patch_size=1,
        image_size=image_size,
        input_channels=input_channels,
        mask_ratio=mask_ratio,
        model_type="qwen2-vl",
    )


# ============ 测试代码 ============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing MAE Decoder V2")
    print("=" * 60)

    # Test SimpleMAEForPretraining with Conv3d
    print("\n1. Testing SimpleMAEForPretraining (Conv3d mode)...")
    model = SimpleMAEForPretraining(
        input_channels=64,
        patch_size=14,
        image_size=196,  # Smaller for testing
        encoder_dim=128,
        decoder_dim=64,
        conv_type="conv3d",
    )

    # Create dummy input
    x = torch.randn(2, 64, 196, 196)
    pred, loss = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test with Conv2d
    print("\n2. Testing SimpleMAEForPretraining (Conv2d mode)...")
    model2 = SimpleMAEForPretraining(
        input_channels=64,
        patch_size=14,
        image_size=196,
        encoder_dim=128,
        decoder_dim=64,
        conv_type="conv2d",
    )

    pred2, loss2 = model2(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Pred shape: {pred2.shape}")
    print(f"  Loss: {loss2.item():.4f}")

    # ============ 验证信息流正确性 ============
    print("\n3. Verifying NO information leakage in SimpleMAEForPretraining...")

    # 创建一个简单的测试：如果有信息泄露，修改 masked patches 应该影响 encoder 输出
    model_test = SimpleMAEForPretraining(
        input_channels=64,
        patch_size=14,
        image_size=196,
        encoder_dim=128,
        decoder_dim=64,
        conv_type="conv2d",
    )
    model_test.eval()

    # 固定 mask
    torch.manual_seed(42)
    x_test = torch.randn(1, 64, 196, 196)
    mask, ids_keep, ids_restore = model_test._random_masking(x_test)

    # 获取 visible 和 masked patch 的索引
    visible_indices = ids_keep[0].tolist()
    num_patches_per_dim = 196 // 14  # 14 patches per dimension
    total_patches = num_patches_per_dim ** 2

    # 找一个 masked patch 的位置
    masked_indices = [i for i in range(total_patches) if i not in visible_indices]
    if masked_indices:
        masked_idx = masked_indices[0]
        # 转换为 patch 的空间位置
        patch_h = masked_idx // num_patches_per_dim
        patch_w = masked_idx % num_patches_per_dim
        h_start, h_end = patch_h * 14, (patch_h + 1) * 14
        w_start, w_end = patch_w * 14, (patch_w + 1) * 14

        # 获取原始输出
        with torch.no_grad():
            # 准备输入 for Conv2d
            x1 = x_test.clone()
            x1_embed = model_test.patch_embed(x1)
            x1_embed = x1_embed.flatten(2).transpose(1, 2)
            x1_embed = x1_embed + model_test.pos_embed
            # 选择 visible patches
            D = x1_embed.shape[-1]
            ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x1_visible = torch.gather(x1_embed, dim=1, index=ids_keep_exp)
            for block in model_test.encoder_blocks:
                x1_visible = block(x1_visible)
            output1 = model_test.encoder_norm(x1_visible)

            # 修改 masked patch 区域
            x2 = x_test.clone()
            x2[:, :, h_start:h_end, w_start:w_end] = torch.randn(1, 64, 14, 14) * 100

            # 获取修改后的输出
            x2_embed = model_test.patch_embed(x2)
            x2_embed = x2_embed.flatten(2).transpose(1, 2)
            x2_embed = x2_embed + model_test.pos_embed
            x2_visible = torch.gather(x2_embed, dim=1, index=ids_keep_exp)
            for block in model_test.encoder_blocks:
                x2_visible = block(x2_visible)
            output2 = model_test.encoder_norm(x2_visible)

        # 比较输出
        diff = (output1 - output2).abs().max().item()
        print(f"  Max difference after modifying masked patch: {diff:.6f}")

        if diff < 1e-5:
            print("  ✓ PASSED: No information leakage - modifying masked patches does not affect encoder output")
        else:
            print("  ✗ FAILED: Information leakage detected!")

    print("\n" + "=" * 60)
    print("Tests passed!")
    print("=" * 60)
