"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

Adapted for 64-channel Alpha Earth embeddings.

FIXED: Added checkpoint saving for best model during pretraining.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mlp_model import PatchMLP
from models.light_cnn import LightCNNEncoder
from utils import AverageMeter, format_time, save_checkpoint


class ProjectionHead(nn.Module):
    """Projection head for SimCLR."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised pretraining.

    Learns representations by maximizing agreement between
    differently augmented views of the same data.
    """

    def __init__(
        self,
        encoder_type: str = "light_cnn",
        projection_dim: int = 128,
        temperature: float = 0.5
    ):
        super().__init__()

        self.temperature = temperature

        # Create encoder based on type
        if encoder_type == "mlp":
            self.encoder = PatchMLP(
                input_dim=config.NUM_BANDS,
                hidden_dims=[256, 128],
                dropout_rate=0.0,  # No dropout for pretraining
                use_batch_norm=True
            )
            encoder_dim = 128
        elif encoder_type == "light_cnn":
            self.encoder = LightCNNEncoder(
                input_channels=config.NUM_BANDS,
                hidden_channels=[32, 64, 128],
                use_batch_norm=True
            )
            encoder_dim = 128
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Projection head
        self.projection = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=256,
            output_dim=projection_dim
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, 64, H, W)
        Returns:
            projections: (batch, projection_dim)
        """
        # Encode patches
        features = self.encoder(x)  # (batch, num_patches, encoder_dim)

        # Aggregate patches (mean pooling)
        features = features.mean(dim=1)  # (batch, encoder_dim)

        # Project
        projections = self.projection(features)

        return projections

    def get_encoder(self):
        return self.encoder

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (batch, dim) - projections of first view
            z_j: (batch, dim) - projections of second view
        Returns:
            loss: scalar
        """
        batch_size = z_i.size(0)

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, dim)

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2*batch, 2*batch)

        # Create mask for positive pairs
        # Positive pairs are (i, i+batch) and (i+batch, i)
        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
        mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
        mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)

        # Mask out self-similarity
        diag_mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(diag_mask, -1e9)

        # Compute loss
        # For each sample, the positive is at a known position
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(z.device)

        loss = F.cross_entropy(sim, labels)

        return loss


class SimCLRTrainer:
    """Trainer for SimCLR pretraining with checkpoint saving."""

    def __init__(
        self,
        model: SimCLR,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = None,
        logger=None,
        exp_name: str = "simclr"
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.exp_name = exp_name

        self.criterion = NTXentLoss(temperature=model.temperature)
        self.model = self.model.to(self.device)

        # Track best model
        self.best_loss = float('inf')
        self.best_state_dict = None

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"SimCLR Pretrain Epoch {epoch}")
        for batch_idx, (view1, view2) in enumerate(pbar):
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)

            # Forward
            z1 = self.model(view1)
            z2 = self.model(view2)

            # Compute loss
            loss = self.criterion(z1, z2)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), view1.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        if self.scheduler:
            self.scheduler.step()

        return loss_meter.avg

    def train(self, num_epochs: int):
        """Full pretraining loop with checkpoint saving."""
        print(f"\nStarting SimCLR pretraining for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Temperature: {self.model.temperature}")

        start_time = time.time()
        history = {'loss': []}

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(epoch)
            history['loss'].append(loss)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{num_epochs} ({format_time(epoch_time)}) - Loss: {loss:.4f}")

            if self.logger:
                self.logger.info(f"SimCLR Pretrain Epoch {epoch}: loss={loss:.4f}")

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
                print(f"  >> New best SimCLR model! (loss: {loss:.4f})")

        total_time = time.time() - start_time
        print(f"\nSimCLR Pretraining completed in {format_time(total_time)}")

        # Save pretrain checkpoint
        if self.best_state_dict:
            checkpoint_path = save_checkpoint(
                self.best_state_dict,
                f"{self.exp_name}_pretrain",
                'simclr_pretrain_best.pth'
            )
            print(f"SimCLR pretrain checkpoint saved to: {checkpoint_path}")

        return history

    def get_encoder_state_dict(self):
        """Get best encoder state dict for downstream task."""
        if self.best_state_dict:
            return self.best_state_dict['encoder_state_dict']
        return self.model.get_encoder_state_dict()


def pretrain_simclr(
    pretrain_loader: DataLoader,
    simclr_config,
    train_config,
    device: torch.device,
    logger=None,
    exp_name: str = "simclr"
):
    """
    Run SimCLR pretraining.

    Args:
        pretrain_loader: DataLoader for pretraining (returns two views)
        simclr_config: SimCLR configuration
        train_config: Training configuration
        device: Device to use
        logger: Logger instance
        exp_name: Experiment name for checkpoint saving

    Returns:
        encoder_state_dict: Pretrained encoder weights
    """
    # Create model
    model = SimCLR(
        encoder_type=simclr_config.encoder_type,
        projection_dim=simclr_config.projection_dim,
        temperature=simclr_config.temperature
    )

    print(f"SimCLR model created:")
    print(f"  - Encoder type: {simclr_config.encoder_type}")
    print(f"  - Projection dim: {simclr_config.projection_dim}")
    print(f"  - Temperature: {simclr_config.temperature}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(
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
    trainer = SimCLRTrainer(
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

    return trainer.get_encoder_state_dict()


if __name__ == "__main__":
    print("Testing SimCLR...")

    # Create dummy data
    batch_size = 4
    num_patches = 25
    channels = 64
    patch_size = 200

    view1 = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)
    view2 = torch.randn(batch_size, num_patches, channels, patch_size, patch_size)

    # Create model
    model = SimCLR(encoder_type="light_cnn")
    print(f"SimCLR parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    z1 = model(view1)
    z2 = model(view2)
    print(f"Projection shape: {z1.shape}")

    # Compute loss
    criterion = NTXentLoss()
    loss = criterion(z1, z2)
    print(f"Loss: {loss.item():.4f}")
