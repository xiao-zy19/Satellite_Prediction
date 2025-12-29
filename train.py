"""
Unified training script for all experiments

Supports:
1. Baseline models (MLP, LightCNN, ResNet)
2. Self-supervised pretraining (SimCLR, MAE) + finetuning

FIXED:
- Proper weight transfer from SimCLR/MAE pretrained encoders
- Checkpoint saving for pretrained models
- Better logging and error handling
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from config import get_experiment_config, print_config, ExperimentConfig
from dataset import get_dataloaders, get_pretrain_dataloader, get_patch_level_dataloaders
from models import MLPModel, LightCNN, ResNetBaseline
from pretrain.simclr import pretrain_simclr
from pretrain.mae import pretrain_mae
from utils import (
    set_seed, get_device, setup_logging, get_optimizer, get_scheduler,
    compute_metrics, EarlyStopping, AverageMeter, save_checkpoint,
    print_metrics, format_time, count_parameters
)

# Wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")


def create_model(model_config, patch_level: bool = False):
    """Create model based on configuration.

    Args:
        model_config: Model configuration
        patch_level: If True, create model for patch-level training (no internal aggregation)
    """
    if model_config.name == "mlp":
        return MLPModel(model_config, patch_level=patch_level)
    elif model_config.name == "light_cnn":
        return LightCNN(model_config, patch_level=patch_level)
    elif model_config.name == "resnet":
        return ResNetBaseline(model_config, patch_level=patch_level)
    else:
        raise ValueError(f"Unknown model: {model_config.name}")


def init_wandb(exp_config: ExperimentConfig, dataset_info: dict) -> bool:
    """Initialize wandb."""
    if not exp_config.wandb_enabled or not WANDB_AVAILABLE:
        return False

    # 在实验名称中加入时间戳，便于查找
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{exp_config.exp_name}_{timestamp}"

    wandb_config = {
        "experiment": exp_config.exp_name,
        "model": exp_config.model_config.name,
        "use_pretrain": exp_config.use_pretrain,
        "pretrain_method": exp_config.pretrain_config.name if exp_config.pretrain_config else "none",
        "training_mode": exp_config.train_config.training_mode,
        "batch_size": exp_config.train_config.batch_size,
        "learning_rate": exp_config.train_config.learning_rate,
        "num_epochs": exp_config.train_config.num_epochs,
        "num_train": dataset_info.get('num_train', 0),
        "num_val": dataset_info.get('num_val', 0),
        "num_test": dataset_info.get('num_test', 0),
    }

    wandb.init(
        project=exp_config.wandb_project,
        name=run_name,
        config=wandb_config,
        reinit=True
    )

    return True


class Trainer:
    """Unified trainer for all models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        exp_config: ExperimentConfig,
        device: torch.device,
        logger=None,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_config = exp_config
        self.train_config = exp_config.train_config
        self.device = device
        self.logger = logger
        self.use_wandb = use_wandb

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer and scheduler
        self.optimizer = get_optimizer(
            model,
            self.train_config.learning_rate,
            self.train_config.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, self.train_config)

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pearson_r': [],
            'val_r2': [],
            'val_mae': [],
            'learning_rate': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        all_labels = []
        all_preds = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for patches, labels, _ in pbar:
            patches = patches.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(patches)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), patches.size(0))
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        train_metrics = compute_metrics(all_labels, all_preds)

        return loss_meter.avg, train_metrics

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        loss_meter = AverageMeter()
        all_labels = []
        all_preds = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for patches, labels, _ in pbar:
            patches = patches.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(patches)
            loss = self.criterion(outputs, labels)

            loss_meter.update(loss.item(), patches.size(0))
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        metrics = compute_metrics(all_labels, all_preds)

        return loss_meter.avg, metrics, all_labels, all_preds

    def train(self, num_epochs: int = None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.train_config.num_epochs

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")

        early_stopping = EarlyStopping(patience=self.train_config.patience, mode='min')
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics, val_labels, val_preds = self.validate(epoch)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_pearson_r'].append(val_metrics['pearson_r'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['learning_rate'].append(current_lr)

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/pearson_r": train_metrics['pearson_r'],
                    "val/loss": val_loss,
                    "val/pearson_r": val_metrics['pearson_r'],
                    "val/r2": val_metrics['r2'],
                    "val/mae": val_metrics['mae'],
                    "learning_rate": current_lr,
                }, step=epoch)

            # Print summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} ({format_time(epoch_time)})")
            print(f"  Train Loss: {train_loss:.4f} | Pearson r: {train_metrics['pearson_r']:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Pearson r: {val_metrics['pearson_r']:.4f} | R²: {val_metrics['r2']:.4f}")
            print(f"  LR: {current_lr:.2e}")

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"pearson_r={val_metrics['pearson_r']:.4f}"
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }
                print(f"  >> New best model!")

                save_checkpoint(
                    self.best_model_state,
                    self.exp_config.exp_name,
                    'best_model.pth'
                )

            # Early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best model at epoch {self.best_epoch} with val_loss: {self.best_val_loss:.4f}")

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])

        return self.history

    def reset_optimizer(self, lr: float = None):
        """Reset optimizer with new learning rate."""
        if lr is None:
            lr = self.train_config.finetune_lr
        self.optimizer = get_optimizer(
            self.model,
            lr,
            self.train_config.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, self.train_config)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_labels = []
    all_preds = []
    all_info = []

    for patches, labels, info in tqdm(data_loader, desc="Evaluating"):
        patches = patches.to(device)
        outputs = model(patches)

        all_labels.append(labels.numpy())
        all_preds.append(outputs.cpu().numpy())

        for i in range(len(info['city'])):
            all_info.append({
                'city': info['city'][i],
                'year': info['year'][i].item() if torch.is_tensor(info['year'][i]) else info['year'][i]
            })

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)

    return y_true, y_pred, metrics, all_info


def run_experiment(exp_name: str, gpu_id: int = 3):
    """Run a single experiment."""
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Get config
    exp_config = get_experiment_config(exp_name)
    print_config(exp_config)

    # Set seed
    set_seed()

    # Setup logging
    logger = setup_logging(exp_name)
    logger.info(f"Starting experiment: {exp_name}")

    # Device
    device = get_device(exp_config.device)
    logger.info(f"Using device: {device}")

    # Determine training mode
    training_mode = exp_config.train_config.training_mode
    is_patch_level = (training_mode == "patch_level")
    logger.info(f"Training mode: {training_mode}")

    # Load data
    logger.info("Loading data...")
    if is_patch_level:
        # Patch-level training: each patch is an independent sample
        train_loader, val_loader, test_loader, dataset_info = get_patch_level_dataloaders(
            batch_size=exp_config.train_config.batch_size,
            num_workers=exp_config.num_workers
        )
        logger.info(f"Patch-level mode: {dataset_info['num_train_patches']} training patches "
                    f"from {dataset_info['num_train']} city-year samples")
    else:
        # City-level training: 25 patches per sample, aggregated internally
        train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
            batch_size=exp_config.train_config.batch_size,
            num_workers=exp_config.num_workers
        )

    # Init wandb
    use_wandb = init_wandb(exp_config, dataset_info)

    # Create model
    logger.info("Creating model...")
    model = create_model(exp_config.model_config, patch_level=is_patch_level)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Self-supervised pretraining if needed
    encoder_state_dict = None
    if exp_config.use_pretrain and exp_config.pretrain_config:
        pretrain_method = exp_config.pretrain_config.name
        logger.info(f"Running self-supervised pretraining: {pretrain_method}")

        # Get pretrain dataloader
        pretrain_loader = get_pretrain_dataloader(
            batch_size=exp_config.train_config.batch_size,
            num_workers=exp_config.num_workers,
            contrastive=(pretrain_method == "simclr")
        )

        if pretrain_method == "simclr":
            encoder_state_dict = pretrain_simclr(
                pretrain_loader,
                exp_config.pretrain_config,
                exp_config.train_config,
                device,
                logger,
                exp_name=exp_config.exp_name
            )
            logger.info("SimCLR pretraining completed")

        elif pretrain_method == "mae":
            encoder_state_dict = pretrain_mae(
                pretrain_loader,
                exp_config.pretrain_config,
                exp_config.train_config,
                device,
                logger,
                exp_name=exp_config.exp_name
            )
            logger.info("MAE pretraining completed")

        # Load pretrained encoder weights to downstream model
        if encoder_state_dict is not None:
            try:
                model.load_encoder(encoder_state_dict)
                logger.info("Successfully loaded pretrained encoder weights to downstream model")
                print("=" * 60)
                print("Pretrained encoder weights loaded successfully!")
                print("=" * 60)
            except Exception as e:
                logger.error(f"Failed to load pretrained encoder: {e}")
                print(f"Warning: Failed to load pretrained encoder: {e}")
                print("Continuing with random initialization...")

        # Freeze encoder for initial epochs if specified
        if exp_config.train_config.freeze_encoder_epochs > 0:
            model.freeze_encoder()
            logger.info(f"Encoder frozen for first {exp_config.train_config.freeze_encoder_epochs} epochs")

    # Move model to device
    model = model.to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_config=exp_config,
        device=device,
        logger=logger,
        use_wandb=use_wandb
    )

    # Train with frozen encoder first (if applicable)
    if exp_config.use_pretrain and exp_config.train_config.freeze_encoder_epochs > 0:
        logger.info(f"Phase 1: Training with frozen encoder for {exp_config.train_config.freeze_encoder_epochs} epochs")
        trainer.train(exp_config.train_config.freeze_encoder_epochs)

        # Unfreeze and continue
        model.unfreeze_encoder()
        trainer.reset_optimizer(exp_config.train_config.finetune_lr)
        logger.info("Phase 2: Encoder unfrozen, continuing training with full model")

    # Full training (or continue training if encoder was frozen)
    remaining_epochs = exp_config.train_config.num_epochs
    if exp_config.use_pretrain and exp_config.train_config.freeze_encoder_epochs > 0:
        remaining_epochs -= exp_config.train_config.freeze_encoder_epochs

    if remaining_epochs > 0:
        history = trainer.train(remaining_epochs)
    else:
        history = trainer.history

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    y_true, y_pred, test_metrics, test_info = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("Test Set Results:")
    print("=" * 60)
    print_metrics(test_metrics, prefix="  ")

    # Log to wandb
    if use_wandb:
        wandb.run.summary["test/pearson_r"] = test_metrics['pearson_r']
        wandb.run.summary["test/r2"] = test_metrics['r2']
        wandb.run.summary["test/mae"] = test_metrics['mae']
        wandb.run.summary["test/rmse"] = test_metrics['rmse']
        wandb.run.summary["best_epoch"] = trainer.best_epoch
        wandb.finish()

    # Save results
    results = {
        'exp_name': exp_name,
        'history': history,
        'test_metrics': test_metrics,
        'test_y_true': y_true,
        'test_y_pred': y_pred,
        'test_info': test_info,
        'best_epoch': trainer.best_epoch,
        'model_params': count_parameters(model),
        'use_pretrain': exp_config.use_pretrain,
        'pretrain_method': exp_config.pretrain_config.name if exp_config.pretrain_config else None,
        'training_mode': training_mode
    }

    results_path = os.path.join(config.RESULT_DIR, f'{exp_name}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run pretrain experiments")
    parser.add_argument('--exp', type=str, required=True,
                        help=f"Experiment name. Available: {list(config.EXPERIMENTS.keys())}")
    parser.add_argument('--gpu', type=int, default=3, help="GPU ID to use")
    args = parser.parse_args()

    run_experiment(args.exp, args.gpu)


if __name__ == "__main__":
    main()
