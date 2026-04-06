"""
Training script for multimodal BERT experiments (Image + BERT Policy features)

Based on train_multimodal.py, extended to support:
- BERT policy embeddings from pre-computed cache
- Hybrid mode (BERT + structured features)
- All original features (SimCLR, MAE, Position-Aware Aggregation)

Usage:
    python train_multimodal_bert.py --exp bert_cnn_concat --gpu 0
    python train_multimodal_bert.py --exp hybrid_cnn_gated --gpu 0 --seed 123
    python train_multimodal_bert.py --exp bert_simclr_cnn_concat --gpu 0  # with SimCLR pretrain
"""

import os
import sys
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import config
from config_multimodal_bert import (
    get_bert_multimodal_experiment_config,
    print_bert_multimodal_config,
    MultiModalBertExperimentConfig,
    BERT_MULTIMODAL_EXPERIMENTS,
    SimCLRConfig,
    MAEConfig
)
from dataset_policy_bert import get_bert_policy_dataloaders, get_patch_level_bert_policy_dataloaders
from dataset import get_pretrain_dataloader
from models.multimodal_bert import create_bert_multimodal_model
from policy_bert import get_policy_bert_cache
from utils import (
    set_seed, get_device, setup_logging, get_optimizer,
    compute_metrics, EarlyStopping, AverageMeter, save_checkpoint,
    format_time, count_parameters
)

# Pretrain modules
from pretrain.simclr import pretrain_simclr
from pretrain.mae import pretrain_mae

# Wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")


def get_scheduler(optimizer, train_config):
    """Get learning rate scheduler."""
    if train_config.scheduler == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_config.t_max,
            T_mult=train_config.t_mult,
            eta_min=train_config.eta_min
        )
    elif train_config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.num_epochs,
            eta_min=train_config.eta_min
        )
    else:
        return None


def init_wandb(exp_config: MultiModalBertExperimentConfig, dataset_info: dict, seed: int, run_id: str) -> bool:
    """Initialize wandb."""
    if not exp_config.wandb_enabled or not WANDB_AVAILABLE:
        return False

    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{run_id}_{timestamp}"

    mc = exp_config.model_config
    ps = mc.policy_source

    wandb_config = {
        "experiment": exp_config.exp_name,
        "run_id": run_id,
        "seed": seed,
        "image_encoder": mc.image_encoder_type,
        "fusion_type": mc.fusion_type,
        "policy_source": ps.source,
        "policy_dim": ps.policy_dim,
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


def _gpu_normalize(patches: torch.Tensor) -> torch.Tensor:
    """
    Perform per-patch per-channel normalization on GPU.

    Args:
        patches: (batch, 25, 64, 200, 200) for city-level
                or (batch, 64, 200, 200) for patch-level

    Returns:
        Normalized patches with same shape
    """
    if patches.dim() == 5:
        mean = patches.mean(dim=(-2, -1), keepdim=True)
        std = patches.std(dim=(-2, -1), keepdim=True) + 1e-8
    elif patches.dim() == 4:
        mean = patches.mean(dim=(-2, -1), keepdim=True)
        std = patches.std(dim=(-2, -1), keepdim=True) + 1e-8
    else:
        raise ValueError(f"Unexpected patches dimension: {patches.dim()}")
    return (patches - mean) / std


class MultiModalBertTrainer:
    """Trainer for multimodal BERT models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        exp_config: MultiModalBertExperimentConfig,
        device: torch.device,
        logger=None,
        use_wandb: bool = False,
        test_loader: DataLoader = None,
        is_patch_level: bool = False,
        run_id: str = None,
        normalize_on_gpu: bool = False,
        freeze_encoder_epochs: int = 0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.exp_config = exp_config
        self.train_config = exp_config.train_config
        self.device = device
        self.logger = logger
        self.use_wandb = use_wandb
        self.is_patch_level = is_patch_level
        self.run_id = run_id if run_id else exp_config.exp_name
        self.normalize_on_gpu = normalize_on_gpu
        self.freeze_encoder_epochs = freeze_encoder_epochs

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer, self.train_config)

        # History
        if is_patch_level:
            self.history = {
                'train_loss': [],
                'learning_rate': [],
                'val_mean_r2': [], 'val_mean_pearson_r': [], 'val_mean_mae': [],
                'val_median_r2': [], 'val_median_pearson_r': [], 'val_median_mae': [],
                'val_trimmed_mean_r2': [], 'val_trimmed_mean_pearson_r': [], 'val_trimmed_mean_mae': [],
                'test_mean_r2': [], 'test_mean_pearson_r': [], 'test_mean_mae': [],
                'test_median_r2': [], 'test_median_pearson_r': [], 'test_median_mae': [],
                'test_trimmed_mean_r2': [], 'test_trimmed_mean_pearson_r': [], 'test_trimmed_mean_mae': [],
            }
        else:
            self.history = {
                'train_loss': [],
                'learning_rate': [],
                'val_r2': [], 'val_pearson_r': [], 'val_mae': [],
                'test_r2': [], 'test_pearson_r': [], 'test_mae': [],
            }

        self.best_val_metric = float('-inf')
        self.best_epoch = 0
        self.best_model_state = None

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        all_labels = []
        all_preds = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for patches, policy_feat, labels, _ in pbar:
            patches = patches.to(self.device)
            if self.normalize_on_gpu:
                patches = _gpu_normalize(patches)
            policy_feat = policy_feat.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(patches, policy_feat)
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
    def validate_on_loader(self, data_loader: DataLoader, desc: str = "Eval"):
        """Validate the model on a given data loader (city-level mode)."""
        self.model.eval()
        loss_meter = AverageMeter()
        all_labels = []
        all_preds = []

        pbar = tqdm(data_loader, desc=desc, leave=False)
        for patches, policy_feat, labels, _ in pbar:
            patches = patches.to(self.device)
            if self.normalize_on_gpu:
                patches = _gpu_normalize(patches)
            policy_feat = policy_feat.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(patches, policy_feat)
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

        print(f"\nStarting multimodal BERT training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        if self.is_patch_level:
            print("Mode: PATCH-LEVEL (evaluating with 3 aggregation methods)")
        if self.normalize_on_gpu:
            print("[GPU Normalization ENABLED]")
        if self.freeze_encoder_epochs > 0:
            print(f"Encoder will be FROZEN for first {self.freeze_encoder_epochs} epochs")

        early_stopping = EarlyStopping(
            patience=self.train_config.patience,
            mode='max'
        )
        start_time = time.time()

        # Initially freeze encoder if specified
        if self.freeze_encoder_epochs > 0:
            self.model.freeze_encoder()
            print("  [Encoder FROZEN]")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Unfreeze encoder after freeze_encoder_epochs
            if self.freeze_encoder_epochs > 0 and epoch == self.freeze_encoder_epochs + 1:
                self.model.unfreeze_encoder()
                print(f"\n  [Encoder UNFROZEN at epoch {epoch}]")

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(current_lr)

            if self.is_patch_level:
                # Patch-level: evaluate with 3 aggregation methods
                trim_ratio = self.train_config.patch_level_trim_ratio
                val_results = evaluate_patch_level(
                    self.model, self.val_loader, self.device,
                    normalize_on_gpu=self.normalize_on_gpu,
                    trim_ratio=trim_ratio
                )
                test_results = None
                if self.test_loader is not None:
                    test_results = evaluate_patch_level(
                        self.model, self.test_loader, self.device,
                        normalize_on_gpu=self.normalize_on_gpu,
                        trim_ratio=trim_ratio
                    )

                # Update history for all 3 methods
                for agg_method in ['mean', 'median', 'trimmed_mean']:
                    m = val_results[agg_method]['metrics']
                    self.history[f'val_{agg_method}_r2'].append(m['r2'])
                    self.history[f'val_{agg_method}_pearson_r'].append(m['pearson_r'])
                    self.history[f'val_{agg_method}_mae'].append(m['mae'])

                    if test_results is not None:
                        tm = test_results[agg_method]['metrics']
                        self.history[f'test_{agg_method}_r2'].append(tm['r2'])
                        self.history[f'test_{agg_method}_pearson_r'].append(tm['pearson_r'])
                        self.history[f'test_{agg_method}_mae'].append(tm['mae'])

                # Use configured aggregation method for primary metric
                agg_method_for_primary = self.train_config.patch_level_aggregation
                primary_metric = val_results[agg_method_for_primary]['metrics']['r2']

                # Log to wandb
                if self.use_wandb:
                    log_dict = {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/pearson_r": train_metrics['pearson_r'],
                        "learning_rate": current_lr,
                    }
                    for agg_method in ['mean', 'median', 'trimmed_mean']:
                        vm = val_results[agg_method]['metrics']
                        log_dict[f"val_{agg_method}/r2"] = vm['r2']
                        log_dict[f"val_{agg_method}/pearson_r"] = vm['pearson_r']
                        log_dict[f"val_{agg_method}/mae"] = vm['mae']
                        if test_results is not None:
                            tm = test_results[agg_method]['metrics']
                            log_dict[f"test_{agg_method}/r2"] = tm['r2']
                            log_dict[f"test_{agg_method}/pearson_r"] = tm['pearson_r']
                            log_dict[f"test_{agg_method}/mae"] = tm['mae']
                    wandb.log(log_dict, step=epoch)

                # Print summary
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{num_epochs} ({format_time(epoch_time)})")
                print(f"  Train Loss: {train_loss:.4f} | Pearson r: {train_metrics['pearson_r']:.4f}")
                print(f"  Val (3 agg methods, primary: {agg_method_for_primary}):")
                for agg_method in ['mean', 'median', 'trimmed_mean']:
                    vm = val_results[agg_method]['metrics']
                    marker = " *" if agg_method == agg_method_for_primary else ""
                    print(f"    {agg_method:12s}: R²={vm['r2']:.4f} | r={vm['pearson_r']:.4f} | MAE={vm['mae']:.4f}{marker}")
                print(f"  LR: {current_lr:.2e}")

                if self.logger:
                    vm = val_results[agg_method_for_primary]['metrics']
                    self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_{agg_method_for_primary}_r2={vm['r2']:.4f}")

                # Save best model
                if primary_metric > self.best_val_metric:
                    self.best_val_metric = primary_metric
                    self.best_epoch = epoch
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': {agg: val_results[agg]['metrics'] for agg in val_results}
                    }
                    print(f"  >> New best model! ({agg_method_for_primary} R²={primary_metric:.4f})")
                    save_checkpoint(self.best_model_state, self.run_id, 'best_model.pth')

                if early_stopping(primary_metric):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            else:
                # City-level mode
                val_loss, val_metrics, _, _ = self.validate_on_loader(self.val_loader, f"Epoch {epoch} [Val]")
                test_metrics = None
                if self.test_loader is not None:
                    _, test_metrics, _, _ = self.validate_on_loader(self.test_loader, f"Epoch {epoch} [Test]")

                self.history['val_r2'].append(val_metrics['r2'])
                self.history['val_pearson_r'].append(val_metrics['pearson_r'])
                self.history['val_mae'].append(val_metrics['mae'])
                if test_metrics is not None:
                    self.history['test_r2'].append(test_metrics['r2'])
                    self.history['test_pearson_r'].append(test_metrics['pearson_r'])
                    self.history['test_mae'].append(test_metrics['mae'])

                primary_metric = val_metrics['r2']

                # Log to wandb
                if self.use_wandb:
                    log_dict = {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/pearson_r": train_metrics['pearson_r'],
                        "val/r2": val_metrics['r2'],
                        "val/pearson_r": val_metrics['pearson_r'],
                        "val/mae": val_metrics['mae'],
                        "learning_rate": current_lr,
                    }
                    if test_metrics is not None:
                        log_dict["test/r2"] = test_metrics['r2']
                        log_dict["test/pearson_r"] = test_metrics['pearson_r']
                        log_dict["test/mae"] = test_metrics['mae']
                    wandb.log(log_dict, step=epoch)

                # Print summary
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{num_epochs} ({format_time(epoch_time)})")
                print(f"  Train Loss: {train_loss:.4f} | Pearson r: {train_metrics['pearson_r']:.4f}")
                print(f"  Val: R²={val_metrics['r2']:.4f} | r={val_metrics['pearson_r']:.4f} | MAE={val_metrics['mae']:.4f}")
                if test_metrics is not None:
                    print(f"  Test: R²={test_metrics['r2']:.4f} | r={test_metrics['pearson_r']:.4f} | MAE={test_metrics['mae']:.4f}")
                print(f"  LR: {current_lr:.2e}")

                if self.logger:
                    self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_r2={val_metrics['r2']:.4f}")

                # Save best model
                if primary_metric > self.best_val_metric:
                    self.best_val_metric = primary_metric
                    self.best_epoch = epoch
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics
                    }
                    print(f"  >> New best model! (R²={primary_metric:.4f})")
                    save_checkpoint(self.best_model_state, self.run_id, 'best_model.pth')

                if early_stopping(primary_metric):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best model at epoch {self.best_epoch} with val metric: {self.best_val_metric:.4f}")

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])

        return self.history


@torch.no_grad()
def evaluate_city_level(model, data_loader, device, normalize_on_gpu: bool = False):
    """Evaluate multimodal BERT model on a dataset (city-level mode)."""
    model.eval()
    all_labels = []
    all_preds = []
    all_info = []

    for patches, policy_feat, labels, info in tqdm(data_loader, desc="Evaluating"):
        patches = patches.to(device)
        if normalize_on_gpu:
            patches = _gpu_normalize(patches)
        policy_feat = policy_feat.to(device)
        outputs = model(patches, policy_feat)

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


@torch.no_grad()
def evaluate_patch_level(model, data_loader, device, num_patches: int = 25, normalize_on_gpu: bool = False, trim_ratio: float = 0.1):
    """Evaluate patch-level model with THREE aggregation methods."""
    model.eval()

    sample_predictions = {}

    for patches, policy_feat, labels, info in tqdm(data_loader, desc="Collecting predictions", leave=False):
        patches = patches.to(device)
        if normalize_on_gpu:
            patches = _gpu_normalize(patches)
        policy_feat = policy_feat.to(device)
        outputs = model(patches, policy_feat)

        batch_size = patches.size(0)
        for i in range(batch_size):
            sample_idx = info['sample_idx'][i].item() if torch.is_tensor(info['sample_idx'][i]) else info['sample_idx'][i]
            patch_idx = info['patch_idx'][i].item() if torch.is_tensor(info['patch_idx'][i]) else info['patch_idx'][i]

            if sample_idx not in sample_predictions:
                sample_predictions[sample_idx] = {
                    'predictions': {},
                    'label': labels[i].item(),
                    'city': info['city'][i],
                    'year': info['year'][i].item() if torch.is_tensor(info['year'][i]) else info['year'][i]
                }

            sample_predictions[sample_idx]['predictions'][patch_idx] = outputs[i].item()

    # Verify completeness
    complete_samples = {}
    incomplete_count = 0
    for sample_idx, data in sample_predictions.items():
        if len(data['predictions']) == num_patches:
            complete_samples[sample_idx] = data
        else:
            incomplete_count += 1

    if incomplete_count > 0:
        print(f"Warning: {incomplete_count} samples have incomplete patches")

    if len(complete_samples) == 0:
        print("Error: No complete samples found!")
        return None

    # Three aggregation functions
    def aggregate_mean(preds):
        return np.mean(preds)

    def aggregate_median(preds):
        return np.median(preds)

    def aggregate_trimmed_mean(preds, tr=trim_ratio):
        sorted_preds = np.sort(preds)
        n = len(sorted_preds)
        n_trim = int(n * tr)
        if n_trim == 0:
            return np.mean(sorted_preds)
        return np.mean(sorted_preds[n_trim:-n_trim])

    aggregation_methods = {
        'mean': aggregate_mean,
        'median': aggregate_median,
        'trimmed_mean': aggregate_trimmed_mean
    }

    results = {}

    for agg_name, agg_func in aggregation_methods.items():
        y_true = []
        y_pred = []
        sample_info = []

        for sample_idx in sorted(complete_samples.keys()):
            data = complete_samples[sample_idx]
            preds = [data['predictions'][p_idx] for p_idx in range(num_patches)]
            aggregated_pred = agg_func(preds)

            y_true.append(data['label'])
            y_pred.append(aggregated_pred)
            sample_info.append({
                'city': data['city'],
                'year': data['year']
            })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        metrics = compute_metrics(y_true, y_pred)

        results[agg_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': metrics,
            'sample_info': sample_info
        }

    return results


def run_bert_multimodal_experiment(
    exp_name: str,
    gpu_id: int = 0,
    seed: int = config.RANDOM_SEED,
    normalize_on_gpu: bool = False,
    cache_dir: str = "policy_bert_cache",
    no_wandb: bool = False,
    temporal_split: bool = False,
    train_years: list[int] = None,
    val_years: list[int] = None,
    test_years: list[int] = None
):
    """Run a single BERT multimodal experiment.

    Args:
        exp_name: Name of the experiment configuration
        gpu_id: GPU device ID to use
        seed: Random seed for reproducibility
        normalize_on_gpu: If True, perform normalization on GPU instead of CPU
        cache_dir: BERT cache directory
        no_wandb: If True, disable wandb logging
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Get config
    exp_config = get_bert_multimodal_experiment_config(exp_name)

    # Override wandb setting
    if no_wandb:
        exp_config.wandb_enabled = False

    print_bert_multimodal_config(exp_config)

    # Create run_id
    if seed != config.RANDOM_SEED:
        run_id = f"{exp_name}_seed{seed}"
    else:
        run_id = exp_name
    if temporal_split:
        if not train_years or not val_years or not test_years:
            raise ValueError("--temporal_split requires --train_years, --val_years, and --test_years")
        import hashlib
        split_key = f"{sorted(train_years)}|{sorted(val_years)}|{sorted(test_years)}"
        split_hash = hashlib.md5(split_key.encode()).hexdigest()[:6]
        run_id = f"{run_id}_temporal_{split_hash}"

    # Get policy source config (needed for log directory)
    policy_source_config = exp_config.model_config.policy_source
    policy_source = policy_source_config.source

    # Set seed
    print(f"\nSetting random seed: {seed}")
    print(f"Run ID: {run_id}")
    print(f"Policy source: {policy_source}")
    if normalize_on_gpu:
        print("[GPU Normalization ENABLED]")
    set_seed(seed)

    # Setup logging to dedicated subfolder based on policy source
    # structured -> logs/Multimodal/, bert -> logs/MultimodalBert/, hybrid -> logs/MultimodalHybrid/
    policy_source_to_dir = {
        'structured': 'Multimodal',
        'bert': 'MultimodalBert',
        'hybrid': 'MultimodalHybrid'
    }
    log_subdir = policy_source_to_dir.get(policy_source, 'Multimodal')
    log_dir = os.path.join(config.LOG_DIR, log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(run_id, log_dir=log_dir)
    logger.info(f"Starting BERT multimodal experiment: {exp_name} (run_id: {run_id}, seed: {seed})")
    logger.info(f"GPU Normalization: {normalize_on_gpu}")
    logger.info(f"Policy source: {policy_source}, dim: {policy_source_config.policy_dim}")

    # Device
    device = get_device(exp_config.device)
    logger.info(f"Using device: {device}")

    # Determine training mode
    training_mode = exp_config.train_config.training_mode
    is_patch_level = (training_mode == "patch_level")
    logger.info(f"Training mode: {training_mode}")

    # ==========================================================================
    # Load BERT Cache (if needed)
    # ==========================================================================
    bert_cache = None
    if policy_source in ["bert", "hybrid"]:
        logger.info(f"Loading BERT cache from {cache_dir}...")
        bert_cache = get_policy_bert_cache(
            cache_dir=cache_dir,
            embedding_dim=policy_source_config.bert_dim
        )
        logger.info(f"BERT cache loaded: {bert_cache.embedding_dim}-dim, {len(bert_cache.embeddings)} entries")

    # ==========================================================================
    # Self-Supervised Pretraining (if enabled)
    # ==========================================================================
    encoder_state_dict = None
    if exp_config.use_pretrain and exp_config.pretrain_config is not None:
        pretrain_config = exp_config.pretrain_config
        train_config = exp_config.train_config

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Self-Supervised Pretraining: {pretrain_config.name}")
        logger.info(f"{'='*60}")

        is_contrastive = (pretrain_config.name == "simclr")
        pretrain_loader = get_pretrain_dataloader(
            batch_size=train_config.batch_size,
            num_workers=exp_config.num_workers,
            contrastive=is_contrastive
        )
        logger.info(f"Pretrain loader created (contrastive={is_contrastive})")

        if pretrain_config.name == "simclr":
            encoder_state_dict = pretrain_simclr(
                pretrain_loader=pretrain_loader,
                simclr_config=pretrain_config,
                train_config=train_config,
                device=device,
                logger=logger,
                exp_name=run_id
            )
        elif pretrain_config.name == "mae":
            encoder_state_dict = pretrain_mae(
                pretrain_loader=pretrain_loader,
                mae_config=pretrain_config,
                train_config=train_config,
                device=device,
                logger=logger,
                exp_name=run_id
            )
        else:
            raise ValueError(f"Unknown pretrain method: {pretrain_config.name}")

        logger.info(f"Pretraining completed. Encoder weights obtained.")
        logger.info(f"{'='*60}\n")

    # ==========================================================================
    # Load Multimodal Data
    # ==========================================================================
    logger.info("Loading data with BERT policy features...")
    if temporal_split:
        logger.info(f"Temporal split: train={train_years}, val={val_years}, test={test_years}")
    if is_patch_level:
        train_loader, val_loader, test_loader, dataset_info = get_patch_level_bert_policy_dataloaders(
            policy_source=policy_source,
            bert_cache=bert_cache,
            batch_size=exp_config.train_config.batch_size,
            num_workers=exp_config.num_workers,
            seed=seed,
            normalize_on_gpu=normalize_on_gpu,
            temporal_split=temporal_split,
            train_years=train_years,
            val_years=val_years,
            test_years=test_years
        )
        logger.info(f"Patch-level mode: {dataset_info['num_train_patches']} training patches "
                    f"from {dataset_info['num_train']} city-year samples")
    else:
        train_loader, val_loader, test_loader, dataset_info = get_bert_policy_dataloaders(
            policy_source=policy_source,
            bert_cache=bert_cache,
            batch_size=exp_config.train_config.batch_size,
            num_workers=exp_config.num_workers,
            seed=seed,
            normalize_on_gpu=normalize_on_gpu,
            temporal_split=temporal_split,
            train_years=train_years,
            val_years=val_years,
            test_years=test_years
        )

    # Init wandb
    use_wandb = init_wandb(exp_config, dataset_info, seed, run_id)

    # ==========================================================================
    # Create Multimodal Model
    # ==========================================================================
    logger.info("Creating multimodal BERT model...")
    model = create_bert_multimodal_model(exp_config.model_config, patch_level=is_patch_level)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Load pretrained encoder weights (if available)
    if encoder_state_dict is not None:
        logger.info("Loading pretrained encoder weights into multimodal model...")
        try:
            model.load_encoder(encoder_state_dict)
            logger.info("Pretrained encoder weights loaded successfully!")

            freeze_epochs = exp_config.train_config.freeze_encoder_epochs
            if freeze_epochs > 0:
                logger.info(f"Will freeze encoder for first {freeze_epochs} epochs")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.warning("Continuing with randomly initialized encoder...")

    # Move model to device
    model = model.to(device)

    # Create trainer
    freeze_encoder_epochs = 0
    if encoder_state_dict is not None:
        freeze_encoder_epochs = exp_config.train_config.freeze_encoder_epochs

    trainer = MultiModalBertTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_config=exp_config,
        device=device,
        logger=logger,
        use_wandb=use_wandb,
        test_loader=test_loader,
        is_patch_level=is_patch_level,
        run_id=run_id,
        normalize_on_gpu=normalize_on_gpu,
        freeze_encoder_epochs=freeze_encoder_epochs
    )

    # Train
    history = trainer.train()

    # ==========================================================================
    # Evaluate on test set
    # ==========================================================================
    logger.info("\nEvaluating on test set...")

    if is_patch_level:
        trim_ratio = exp_config.train_config.patch_level_trim_ratio
        val_results = evaluate_patch_level(
            model, val_loader, device,
            normalize_on_gpu=normalize_on_gpu,
            trim_ratio=trim_ratio
        )
        test_results = evaluate_patch_level(
            model, test_loader, device,
            normalize_on_gpu=normalize_on_gpu,
            trim_ratio=trim_ratio
        )

        primary_agg = exp_config.train_config.patch_level_aggregation

        print("\n" + "=" * 60)
        print(f"Final Results (BERT Multimodal Patch-Level, primary agg: {primary_agg})")
        print("=" * 60)

        print("\nValidation Set:")
        for agg_method in ['mean', 'median', 'trimmed_mean']:
            vm = val_results[agg_method]['metrics']
            print(f"  {agg_method:12s}: R²={vm['r2']:.4f} | r={vm['pearson_r']:.4f} | MAE={vm['mae']:.4f} | RMSE={vm['rmse']:.4f}")

        print("\nTest Set:")
        for agg_method in ['mean', 'median', 'trimmed_mean']:
            tm = test_results[agg_method]['metrics']
            print(f"  {agg_method:12s}: R²={tm['r2']:.4f} | r={tm['pearson_r']:.4f} | MAE={tm['mae']:.4f} | RMSE={tm['rmse']:.4f}")

        # Log to wandb
        if use_wandb:
            for agg_method in ['mean', 'median', 'trimmed_mean']:
                vm = val_results[agg_method]['metrics']
                tm = test_results[agg_method]['metrics']
                wandb.run.summary[f"final_val_{agg_method}/r2"] = vm['r2']
                wandb.run.summary[f"final_val_{agg_method}/pearson_r"] = vm['pearson_r']
                wandb.run.summary[f"final_val_{agg_method}/mae"] = vm['mae']
                wandb.run.summary[f"final_test_{agg_method}/r2"] = tm['r2']
                wandb.run.summary[f"final_test_{agg_method}/pearson_r"] = tm['pearson_r']
                wandb.run.summary[f"final_test_{agg_method}/mae"] = tm['mae']
            wandb.run.summary["best_epoch"] = trainer.best_epoch
            wandb.finish()

        # Save results
        results = {
            'exp_name': exp_name,
            'seed': seed,
            'history': history,
            'training_mode': training_mode,
            'policy_source': policy_source,
            'policy_dim': policy_source_config.policy_dim,
            'best_epoch': trainer.best_epoch,
            'model_params': count_parameters(model),
            'fusion_type': exp_config.model_config.fusion_type,
            'image_encoder': exp_config.model_config.image_encoder_type,
            'temporal_split': temporal_split,
            'val_results': {
                agg: {
                    'metrics': val_results[agg]['metrics'],
                    'y_true': val_results[agg]['y_true'],
                    'y_pred': val_results[agg]['y_pred'],
                    'sample_info': val_results[agg]['sample_info']
                } for agg in ['mean', 'median', 'trimmed_mean']
            },
            'test_results': {
                agg: {
                    'metrics': test_results[agg]['metrics'],
                    'y_true': test_results[agg]['y_true'],
                    'y_pred': test_results[agg]['y_pred'],
                    'sample_info': test_results[agg]['sample_info']
                } for agg in ['mean', 'median', 'trimmed_mean']
            }
        }
        if temporal_split:
            results['train_years'] = train_years
            results['val_years'] = val_years
            results['test_years'] = test_years

    else:
        # City-level
        val_y_true, val_y_pred, val_metrics, val_info = evaluate_city_level(model, val_loader, device, normalize_on_gpu=normalize_on_gpu)
        test_y_true, test_y_pred, test_metrics, test_info = evaluate_city_level(model, test_loader, device, normalize_on_gpu=normalize_on_gpu)

        print("\n" + "=" * 60)
        print(f"Final Results (BERT Multimodal City-Level, fusion: {exp_config.model_config.fusion_type})")
        print(f"Policy Source: {policy_source} ({policy_source_config.policy_dim}-dim)")
        print("=" * 60)

        print(f"\nValidation Set:")
        print(f"  R²={val_metrics['r2']:.4f} | r={val_metrics['pearson_r']:.4f} | MAE={val_metrics['mae']:.4f} | RMSE={val_metrics['rmse']:.4f}")

        print(f"\nTest Set:")
        print(f"  R²={test_metrics['r2']:.4f} | r={test_metrics['pearson_r']:.4f} | MAE={test_metrics['mae']:.4f} | RMSE={test_metrics['rmse']:.4f}")

        # Log to wandb
        if use_wandb:
            wandb.run.summary["final_val/r2"] = val_metrics['r2']
            wandb.run.summary["final_val/pearson_r"] = val_metrics['pearson_r']
            wandb.run.summary["final_val/mae"] = val_metrics['mae']
            wandb.run.summary["final_test/r2"] = test_metrics['r2']
            wandb.run.summary["final_test/pearson_r"] = test_metrics['pearson_r']
            wandb.run.summary["final_test/mae"] = test_metrics['mae']
            wandb.run.summary["best_epoch"] = trainer.best_epoch
            wandb.finish()

        # Save results
        results = {
            'exp_name': exp_name,
            'seed': seed,
            'history': history,
            'training_mode': training_mode,
            'policy_source': policy_source,
            'policy_dim': policy_source_config.policy_dim,
            'best_epoch': trainer.best_epoch,
            'model_params': count_parameters(model),
            'fusion_type': exp_config.model_config.fusion_type,
            'image_encoder': exp_config.model_config.image_encoder_type,
            'model_aggregation': exp_config.model_config.aggregation,
            'val_metrics': val_metrics,
            'val_y_true': val_y_true,
            'val_y_pred': val_y_pred,
            'val_info': val_info,
            'test_metrics': test_metrics,
            'test_y_true': test_y_true,
            'test_y_pred': test_y_pred,
            'test_info': test_info,
            'temporal_split': temporal_split,
        }
        if temporal_split:
            results['train_years'] = train_years
            results['val_years'] = val_years
            results['test_years'] = test_years

    # Save results to dedicated subfolder based on policy source
    # structured -> Multimodal/, bert -> MultimodalBert/, hybrid -> MultimodalHybrid/
    policy_source_to_dir = {
        'structured': 'Multimodal',
        'bert': 'MultimodalBert',
        'hybrid': 'MultimodalHybrid'
    }
    result_subdir = policy_source_to_dir.get(policy_source, 'Multimodal')
    result_dir = os.path.join(config.RESULT_DIR, result_subdir)
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, f'{run_id}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BERT multimodal experiments")
    parser.add_argument('--exp', type=str, required=True,
                        help=f"Experiment name. Available: {list(BERT_MULTIMODAL_EXPERIMENTS.keys())}")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {config.RANDOM_SEED})")
    parser.add_argument('--normalize_on_gpu', action='store_true',
                        help="Perform normalization on GPU instead of CPU (improves training speed)")
    parser.add_argument('--cache-dir', type=str, default='policy_bert_cache',
                        help="BERT cache directory")
    parser.add_argument('--no-wandb', action='store_true',
                        help="Disable wandb logging")
    parser.add_argument('--temporal_split', action='store_true',
                        help="Use temporal (year-based) split instead of random city split")
    parser.add_argument('--train_years', type=int, nargs='+', default=None,
                        help="Training years for temporal split (e.g. 2018 2019 2020 2021)")
    parser.add_argument('--val_years', type=int, nargs='+', default=None,
                        help="Validation years for temporal split (e.g. 2022)")
    parser.add_argument('--test_years', type=int, nargs='+', default=None,
                        help="Test years for temporal split (e.g. 2023)")
    args = parser.parse_args()

    run_bert_multimodal_experiment(
        args.exp,
        args.gpu,
        args.seed,
        normalize_on_gpu=args.normalize_on_gpu,
        cache_dir=args.cache_dir,
        no_wandb=args.no_wandb,
        temporal_split=args.temporal_split,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years
    )


if __name__ == "__main__":
    main()
