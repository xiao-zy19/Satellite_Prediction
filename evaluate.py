"""
Evaluation script for trained models

Usage:
    python evaluate.py --exp mlp_baseline --gpu 0
    python evaluate.py --checkpoint checkpoints/mlp_baseline/best_model.pth --model mlp --gpu 0
"""

import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import config
from config import get_experiment_config
from dataset import get_dataloaders, get_patch_level_dataloaders
from models import MLPModel, LightCNN, ResNetBaseline
from utils import set_seed, get_device, compute_metrics, print_metrics


def trimmed_mean(values, trim_ratio=0.1):
    """Compute trimmed mean by removing extreme values."""
    values = np.array(values)
    n = len(values)
    n_trim = max(1, int(n * trim_ratio))
    if n <= 2 * n_trim:
        return np.mean(values)
    sorted_values = np.sort(values)
    return np.mean(sorted_values[n_trim:n - n_trim])


def create_model(model_config, patch_level: bool = False):
    """Create model based on configuration.

    Args:
        model_config: Model configuration
        patch_level: If True, create model for patch-level training
    """
    if model_config.name == "mlp":
        return MLPModel(model_config, patch_level=patch_level)
    elif model_config.name == "light_cnn":
        return LightCNN(model_config, patch_level=patch_level)
    elif model_config.name == "resnet":
        return ResNetBaseline(model_config, patch_level=patch_level)
    else:
        raise ValueError(f"Unknown model: {model_config.name}")


def load_model(checkpoint_path: str, model_config, patch_level: bool = False):
    """Load model from checkpoint."""
    model = create_model(model_config, patch_level=patch_level)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, data_loader, device, verbose=True):
    """
    Evaluate model on a dataset.

    Returns:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: Dictionary of evaluation metrics
        all_info: List of sample info (city, year)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_info = []

    iterator = tqdm(data_loader, desc="Evaluating") if verbose else data_loader
    for patches, labels, info in iterator:
        patches = patches.to(device)
        outputs = model(patches)

        all_labels.append(labels.numpy())
        all_preds.append(outputs.cpu().numpy())

        for i in range(len(info['city'])):
            all_info.append({
                'city': info['city'][i],
                'year': info['year'][i].item() if torch.is_tensor(info['year'][i]) else info['year'][i]
            })

    y_true = np.concatenate(all_labels).flatten()
    y_pred = np.concatenate(all_preds).flatten()
    metrics = compute_metrics(y_true, y_pred)

    return y_true, y_pred, metrics, all_info


@torch.no_grad()
def evaluate_patch_level(model, data_loader, device, aggregation="trimmed_mean",
                         trim_ratio=0.1, verbose=True):
    """
    Evaluate patch-level model with city-level aggregation.

    In patch-level training, each patch predicts independently.
    At inference, we aggregate all patch predictions for each city-year
    into a single prediction.

    Args:
        model: Patch-level model
        data_loader: PatchLevelDataset loader
        device: Device to run on
        aggregation: How to aggregate patch predictions ("mean", "median", "trimmed_mean")
        trim_ratio: Ratio for trimmed mean
        verbose: Whether to show progress bar

    Returns:
        y_true: Ground truth values (one per city-year)
        y_pred: Aggregated predicted values (one per city-year)
        metrics: Evaluation metrics
        all_info: List of city-year info
    """
    model.eval()

    # Collect predictions grouped by city-year
    city_year_predictions = {}  # key: (city, year) -> list of predictions
    city_year_labels = {}       # key: (city, year) -> label

    iterator = tqdm(data_loader, desc="Evaluating (patch-level)") if verbose else data_loader
    for patch, label, info in iterator:
        patch = patch.to(device)
        output = model(patch)

        # Get batch size
        batch_size = patch.size(0)

        for i in range(batch_size):
            city = info['city'][i]
            year = info['year'][i].item() if torch.is_tensor(info['year'][i]) else info['year'][i]
            key = (city, year)

            pred = output[i].cpu().numpy().item()
            true_label = label[i].numpy().item()

            if key not in city_year_predictions:
                city_year_predictions[key] = []
                city_year_labels[key] = true_label

            city_year_predictions[key].append(pred)

    # Aggregate predictions for each city-year
    y_true = []
    y_pred = []
    all_info = []

    for key in sorted(city_year_predictions.keys()):
        city, year = key
        predictions = city_year_predictions[key]
        true_label = city_year_labels[key]

        # Aggregate predictions
        if aggregation == "mean":
            agg_pred = np.mean(predictions)
        elif aggregation == "median":
            agg_pred = np.median(predictions)
        elif aggregation == "trimmed_mean":
            agg_pred = trimmed_mean(predictions, trim_ratio)
        else:
            agg_pred = np.mean(predictions)

        y_true.append(true_label)
        y_pred.append(agg_pred)
        all_info.append({'city': city, 'year': year, 'num_patches': len(predictions)})

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = compute_metrics(y_true, y_pred)

    if verbose:
        print(f"\nPatch-level evaluation: {len(city_year_predictions)} city-year samples")
        print(f"Aggregation method: {aggregation}")
        if aggregation == "trimmed_mean":
            print(f"Trim ratio: {trim_ratio}")

    return y_true, y_pred, metrics, all_info


def plot_predictions(y_true, y_pred, metrics, output_path, title=""):
    """Plot predicted vs true values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_pred, y_true, alpha=0.6, s=40, c='steelblue', edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Predicted Growth Rate', fontsize=12)
    ax.set_ylabel('True Growth Rate', fontsize=12)

    title_str = title if title else "Prediction Results"
    ax.set_title(f"{title_str}\n"
                 f"Pearson r = {metrics['pearson_r']:.4f}, "
                 f"R² = {metrics['r2']:.4f}, "
                 f"MAE = {metrics['mae']:.4f}",
                 fontsize=12)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to: {output_path}")


def plot_residuals(y_true, y_pred, output_path, title=""):
    """Plot residuals (prediction errors)."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.6, s=40, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Value', fontsize=12)
    ax.set_ylabel('Residual (Pred - True)', fontsize=12)
    ax.set_title('Residuals vs Predicted Values', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title if title else "Residual Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Residual plot saved to: {output_path}")


def save_predictions(y_true, y_pred, all_info, output_path):
    """Save predictions to CSV file."""
    import pandas as pd

    df = pd.DataFrame({
        'city': [info['city'] for info in all_info],
        'year': [info['year'] for info in all_info],
        'true_value': y_true,
        'predicted_value': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true)
    })

    df = df.sort_values('abs_error', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    return df


def print_detailed_results(metrics, df):
    """Print detailed evaluation results."""
    print("\n" + "=" * 60)
    print("Detailed Evaluation Results")
    print("=" * 60)

    print("\nOverall Metrics:")
    print_metrics(metrics, prefix="  ")

    print("\nError Statistics:")
    errors = df['error'].values
    abs_errors = df['abs_error'].values
    print(f"  Mean Error: {errors.mean():.4f}")
    print(f"  Std Error: {errors.std():.4f}")
    print(f"  Median Abs Error: {np.median(abs_errors):.4f}")
    print(f"  90th percentile Abs Error: {np.percentile(abs_errors, 90):.4f}")
    print(f"  Max Abs Error: {abs_errors.max():.4f}")

    print("\nTop 10 Worst Predictions:")
    print(df.head(10)[['city', 'year', 'true_value', 'predicted_value', 'error']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--exp', type=str, default=None,
                        help=f"Experiment name. Available: {list(config.EXPERIMENTS.keys())}")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to model checkpoint (alternative to --exp)")
    parser.add_argument('--model', type=str, default="light_cnn",
                        choices=["mlp", "light_cnn", "resnet"],
                        help="Model type (required if using --checkpoint)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--split', type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to evaluate on")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Validate arguments
    if args.exp is None and args.checkpoint is None:
        parser.error("Either --exp or --checkpoint must be specified")

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set seed for reproducibility
    set_seed()

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load model and determine training mode
    if args.exp:
        exp_config = get_experiment_config(args.exp)
        model_config = exp_config.model_config
        train_config = exp_config.train_config
        training_mode = train_config.training_mode
        checkpoint_path = config.CHECKPOINT_DIR / args.exp / "best_model.pth"

        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print("Make sure you have trained this experiment first.")
            return

        exp_name = args.exp
    else:
        # Use checkpoint directly
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return

        # Create model config
        if args.model == "mlp":
            model_config = config.MLPConfig()
        elif args.model == "light_cnn":
            model_config = config.LightCNNConfig()
        elif args.model == "resnet":
            model_config = config.ResNetConfig()

        # Default to city_level if using checkpoint directly
        training_mode = "city_level"
        train_config = config.TrainConfig()

        exp_name = checkpoint_path.stem

    is_patch_level = (training_mode == "patch_level")
    print(f"\nTraining mode: {training_mode}")

    print(f"Loading model from: {checkpoint_path}")
    model, checkpoint = load_model(str(checkpoint_path), model_config, patch_level=is_patch_level)
    model = model.to(device)
    print(f"Model loaded successfully")

    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    # Load data based on training mode
    print("\nLoading data...")
    if is_patch_level:
        train_loader, val_loader, test_loader, dataset_info = get_patch_level_dataloaders(
            batch_size=64,
            num_workers=4
        )
    else:
        train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
            batch_size=16,
            num_workers=4
        )

    # Select data split
    if args.split == "train":
        data_loader = train_loader
        split_name = "Training"
    elif args.split == "val":
        data_loader = val_loader
        split_name = "Validation"
    else:
        data_loader = test_loader
        split_name = "Test"

    print(f"Evaluating on {split_name} set ({len(data_loader.dataset)} samples)...")

    # Evaluate based on training mode
    if is_patch_level:
        # Patch-level evaluation with aggregation
        y_true, y_pred, metrics, all_info = evaluate_patch_level(
            model, data_loader, device,
            aggregation=train_config.patch_level_aggregation,
            trim_ratio=train_config.patch_level_trim_ratio
        )
    else:
        # City-level evaluation (standard)
        y_true, y_pred, metrics, all_info = evaluate_model(model, data_loader, device)

    # Output directory
    output_dir = args.output_dir or str(config.RESULT_DIR / f"eval_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    df = save_predictions(
        y_true, y_pred, all_info,
        os.path.join(output_dir, f"predictions_{args.split}.csv")
    )

    # Print results
    print_detailed_results(metrics, df)

    # Create plots
    plot_predictions(
        y_true, y_pred, metrics,
        os.path.join(output_dir, f"predictions_{args.split}.png"),
        title=f"{exp_name} - {split_name} Set"
    )

    plot_residuals(
        y_true, y_pred,
        os.path.join(output_dir, f"residuals_{args.split}.png"),
        title=f"{exp_name} - {split_name} Set"
    )

    # Save metrics
    results = {
        'exp_name': exp_name,
        'split': args.split,
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'info': all_info
    }

    with open(os.path.join(output_dir, f"eval_results_{args.split}.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
