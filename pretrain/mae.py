"""
Masked Autoencoder (MAE) for self-supervised pretraining

Adapted for 64-channel Alpha Earth embeddings.
Learns representations by reconstructing masked patches.

FIXED: Now uses the same encoder architecture as downstream models
to enable proper weight transfer.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.light_cnn import LightCNNEncoder
from models.mlp_model import PatchMLP
from utils import AverageMeter, format_time, save_checkpoint


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    """MAE decoder that reconstructs masked patch features."""

    def __init__(
        self,
        num_patches: int = 25,
        encoder_dim: int = 128,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        num_heads: int = 4
    ):
        super().__init__()

        self.num_patches = num_patches
        self.decoder_dim = decoder_dim

        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads, 4.0)
            for _ in range(decoder_depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim)

        # Prediction head - reconstruct encoder features
        self.pred = nn.Linear(decoder_dim, encoder_dim)

    def forward(self, x_visible, visible_mask):
        """
        Args:
            x_visible: (batch, num_visible, encoder_dim)
            visible_mask: (batch, num_patches) - True for visible patches

        Returns:
            pred: (batch, num_patches, encoder_dim) - reconstructed features
        """
        batch_size = x_visible.size(0)

        # Project to decoder dimension
        x = self.decoder_embed(x_visible)

        # Create full sequence with mask tokens
        full_seq = self.mask_token.expand(batch_size, self.num_patches, -1).clone()

        # Fill in visible patches
        for b in range(batch_size):
            vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            num_vis = min(vis_idx.size(0), x.size(1))
            full_seq[b, vis_idx[:num_vis]] = x[b, :num_vis]

        # Add positional embedding
        full_seq = full_seq + self.decoder_pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            full_seq = block(full_seq)

        full_seq = self.norm(full_seq)

        # Predict encoder features
        pred = self.pred(full_seq)

        return pred


class MAE(nn.Module):
    """
    Masked Autoencoder for self-supervised pretraining.

    Uses the same encoder architecture as downstream models (LightCNNEncoder or PatchMLP)
    to enable proper weight transfer after pretraining.
    """

    def __init__(
        self,
        encoder_type: str = "light_cnn",
        num_patches: int = 25,
        in_channels: int = 64,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        mask_ratio: float = 0.75
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        self.encoder_type = encoder_type

        # Use the same encoder as downstream models
        if encoder_type == "light_cnn":
            self.encoder = LightCNNEncoder(
                input_channels=in_channels,
                hidden_channels=[32, 64, 128],
                use_batch_norm=True
            )
            encoder_dim = self.encoder.output_dim  # 128
        elif encoder_type == "mlp":
            self.encoder = PatchMLP(
                input_dim=in_channels,
                hidden_dims=[256, 128],
                dropout_rate=0.0,  # No dropout for pretraining
                use_batch_norm=True
            )
            encoder_dim = self.encoder.output_dim  # 128
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.encoder_dim = encoder_dim

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Decoder
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth
        )

    def random_masking(self, batch_size: int, device):
        """Generate random mask."""
        num_mask = int(self.num_patches * self.mask_ratio)

        mask = torch.zeros(batch_size, self.num_patches, device=device, dtype=torch.bool)
        for b in range(batch_size):
            mask_idx = torch.randperm(self.num_patches, device=device)[:num_mask]
            mask[b, mask_idx] = True

        return mask

    def forward(self, x, return_features=False):
        """
        Args:
            x: (batch, num_patches, C, H, W)
            return_features: if True, return encoder features without masking

        Returns:
            loss, pred, mask (or features if return_features=True)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode all patches first to get target
        with torch.no_grad():
            target_features = self.encoder(x)  # (batch, num_patches, encoder_dim)

        if return_features:
            return target_features

        # Random masking
        mask = self.random_masking(batch_size, device)
        visible_mask = ~mask

        # Encode visible patches only
        # We need to mask the input and only encode visible patches
        visible_features_list = []
        for b in range(batch_size):
            vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            vis_patches = x[b, vis_idx]  # (num_visible, C, H, W)
            vis_patches = vis_patches.unsqueeze(0)  # (1, num_visible, C, H, W)
            vis_feat = self.encoder(vis_patches)  # (1, num_visible, encoder_dim)
            visible_features_list.append(vis_feat.squeeze(0))

        # Pad to same length
        max_visible = max(vf.size(0) for vf in visible_features_list)
        visible_features = torch.zeros(batch_size, max_visible, self.encoder_dim, device=device)
        for b, vf in enumerate(visible_features_list):
            visible_features[b, :vf.size(0)] = vf

        # Add positional embedding to visible features
        for b in range(batch_size):
            vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            for i, idx in enumerate(vis_idx):
                if i < visible_features.size(1):
                    visible_features[b, i] = visible_features[b, i] + self.pos_embed[0, idx]

        # Decode
        pred = self.decoder(visible_features, visible_mask)

        # Compute loss only on masked patches
        loss = self.compute_loss(pred, target_features, mask)

        return loss, pred, mask

    def compute_loss(self, pred, target, mask):
        """Compute MSE loss on masked patches."""
        # pred, target: (batch, num_patches, encoder_dim)
        # mask: (batch, num_patches) - True for masked

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (batch, num_patches)

        # Only compute loss on masked patches
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)

        return loss

    def get_encoder(self):
        """Return encoder for downstream task."""
        return self.encoder

    def get_encoder_state_dict(self):
        """Get encoder state dict for weight transfer."""
        return self.encoder.state_dict()


class MAETrainer:
    """Trainer for MAE pretraining with checkpoint saving."""

    def __init__(
        self,
        model: MAE,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = None,
        logger=None,
        exp_name: str = "mae"
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.exp_name = exp_name

        self.model = self.model.to(self.device)

        # Track best model
        self.best_loss = float('inf')
        self.best_state_dict = None

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"MAE Pretrain Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Handle both contrastive (view1, view2) and non-contrastive formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                patches = batch[0]  # Use first view
            else:
                patches = batch

            patches = patches.to(self.device)

            # Forward
            loss, pred, mask = self.model(patches)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), patches.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        if self.scheduler:
            self.scheduler.step()

        return loss_meter.avg

    def train(self, num_epochs: int):
        """Full pretraining loop with checkpoint saving."""
        print(f"\nStarting MAE pretraining for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mask ratio: {self.model.mask_ratio}")
        print(f"Encoder type: {self.model.encoder_type}")

        start_time = time.time()
        history = {'loss': []}

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(epoch)
            history['loss'].append(loss)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{num_epochs} ({format_time(epoch_time)}) - Loss: {loss:.4f}")

            if self.logger:
                self.logger.info(f"MAE Pretrain Epoch {epoch}: loss={loss:.4f}")

            # Save best model
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_state_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'encoder_state_dict': self.model.get_encoder_state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss
                }
                print(f"  >> New best MAE model! (loss: {loss:.4f})")

        total_time = time.time() - start_time
        print(f"\nMAE Pretraining completed in {format_time(total_time)}")

        # Save pretrain checkpoint
        if self.best_state_dict:
            checkpoint_path = save_checkpoint(
                self.best_state_dict,
                f"{self.exp_name}_pretrain",
                'mae_pretrain_best.pth'
            )
            print(f"MAE pretrain checkpoint saved to: {checkpoint_path}")

        return history

    def get_encoder_state_dict(self):
        """Get best encoder state dict for downstream task."""
        if self.best_state_dict:
            return self.best_state_dict['encoder_state_dict']
        return self.model.get_encoder_state_dict()

    def get_model(self):
        return self.model


def pretrain_mae(
    pretrain_loader: DataLoader,
    mae_config,
    train_config,
    device: torch.device,
    logger=None,
    exp_name: str = "mae"
):
    """
    Run MAE pretraining.

    Returns:
        encoder_state_dict: Pretrained encoder weights for transfer
    """
    # Create model with same encoder type as downstream
    model = MAE(
        encoder_type=mae_config.encoder_type,
        num_patches=config.NUM_PATCHES_TOTAL,
        in_channels=config.NUM_BANDS,
        mask_ratio=mae_config.mask_ratio,
        decoder_dim=mae_config.decoder_dim,
        decoder_depth=mae_config.decoder_depth
    )

    print(f"MAE model created:")
    print(f"  - Encoder type: {mae_config.encoder_type}")
    print(f"  - Encoder dim: {model.encoder_dim}")
    print(f"  - Mask ratio: {mae_config.mask_ratio}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.pretrain_lr,
        weight_decay=train_config.weight_decay
    )

    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config.pretrain_epochs,
        eta_min=1e-6
    )

    # Trainer
    trainer = MAETrainer(
        model=model,
        train_loader=pretrain_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        exp_name=exp_name
    )

    # Train
    trainer.train(train_config.pretrain_epochs)

    # Return encoder state dict for transfer to downstream model
    return trainer.get_encoder_state_dict()


if __name__ == "__main__":
    print("Testing MAE...")

    # Create dummy data
    batch_size = 4
    num_patches = 25
    channels = 64
    patch_size = 200

    x = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)

    # Test with light_cnn encoder
    print("\nTesting with LightCNN encoder:")
    model = MAE(encoder_type="light_cnn", num_patches=num_patches, in_channels=channels)
    print(f"MAE parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    loss, pred, mask = model(x)
    print(f"Loss: {loss.item():.4f}")
    print(f"Pred shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked ratio: {mask.float().mean().item():.2f}")

    # Test encoder state dict
    encoder_state = model.get_encoder_state_dict()
    print(f"Encoder state dict keys: {list(encoder_state.keys())[:5]}...")

    # Test with MLP encoder
    print("\nTesting with MLP encoder:")
    model_mlp = MAE(encoder_type="mlp", num_patches=num_patches, in_channels=channels)
    print(f"MAE (MLP) parameters: {sum(p.numel() for p in model_mlp.parameters()):,}")
    loss_mlp, _, _ = model_mlp(x)
    print(f"Loss: {loss_mlp.item():.4f}")
