"""
Evaluation script for MLLM models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Handle edge cases
    if len(y_true) < 2:
        return {
            'r2': 0.0,
            'pearson_r': 0.0,
            'p_value': 1.0,
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf'),
        }

    metrics = {}

    # R² score
    metrics['r2'] = float(r2_score(y_true, y_pred))

    # Pearson correlation
    r, p = pearsonr(y_true, y_pred)
    metrics['pearson_r'] = float(r)
    metrics['p_value'] = float(p)

    # MAE
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))

    # RMSE
    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE (Mean Absolute Percentage Error)
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        metrics['mape'] = float(mape)
    else:
        metrics['mape'] = float('inf')

    # Additional statistics
    metrics['mean_true'] = float(np.mean(y_true))
    metrics['mean_pred'] = float(np.mean(y_pred))
    metrics['std_true'] = float(np.std(y_true))
    metrics['std_pred'] = float(np.std(y_pred))

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    return_predictions: bool = False,
) -> Dict:
    """
    Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        return_predictions: Whether to return predictions

    Returns:
        Dictionary with metrics and optionally predictions
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_cities = []
    all_years = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        labels = batch['labels']

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(images)

            if isinstance(outputs, dict):
                preds = outputs.get('regression', outputs.get('logits'))
            else:
                preds = outputs

        # Collect results
        all_preds.extend(preds.squeeze().cpu().tolist())
        all_labels.extend(labels.squeeze().tolist())

        if 'cities' in batch:
            all_cities.extend(batch['cities'])
        if 'years' in batch:
            all_years.extend(batch['years'])

    # Compute metrics
    metrics = compute_regression_metrics(all_labels, all_preds)

    result = {'metrics': metrics}

    if return_predictions:
        result['predictions'] = {
            'y_true': all_labels,
            'y_pred': all_preds,
            'cities': all_cities,
            'years': all_years,
        }

    return result


def evaluate_checkpoint(
    checkpoint_path: str,
    model_class,
    dataloader: DataLoader,
    device: str = "cuda",
    **model_kwargs
) -> Dict:
    """
    Evaluate a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model_class: Model class
        dataloader: Data loader
        device: Device
        **model_kwargs: Additional model arguments

    Returns:
        Evaluation results
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = model_class(**model_kwargs)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Evaluate
    results = evaluate_model(model, dataloader, device, return_predictions=True)

    # Add checkpoint info
    results['checkpoint_info'] = {
        'path': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'step': checkpoint.get('global_step', 'unknown'),
    }

    return results


def evaluate_multiple_checkpoints(
    checkpoint_dir: str,
    model_class,
    dataloader: DataLoader,
    pattern: str = "*.pt",
    device: str = "cuda",
    **model_kwargs
) -> List[Dict]:
    """
    Evaluate all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_class: Model class
        dataloader: Data loader
        pattern: Glob pattern for checkpoints
        device: Device
        **model_kwargs: Model arguments

    Returns:
        List of results for each checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(pattern))

    results = []
    for ckpt_path in checkpoints:
        logger.info(f"Evaluating {ckpt_path}")
        try:
            result = evaluate_checkpoint(
                str(ckpt_path),
                model_class,
                dataloader,
                device,
                **model_kwargs
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {ckpt_path}: {e}")

    return results


def print_evaluation_report(results: Dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)

    metrics = results['metrics']

    print(f"\nRegression Metrics:")
    print(f"  R² Score:           {metrics['r2']:.4f}")
    print(f"  Pearson Correlation: {metrics['pearson_r']:.4f} (p={metrics['p_value']:.4e})")
    print(f"  MAE:                {metrics['mae']:.4f}")
    print(f"  RMSE:               {metrics['rmse']:.4f}")
    print(f"  MAPE:               {metrics['mape']:.2f}%")

    print(f"\nDistribution Statistics:")
    print(f"  True Mean:  {metrics['mean_true']:.4f} (std={metrics['std_true']:.4f})")
    print(f"  Pred Mean:  {metrics['mean_pred']:.4f} (std={metrics['std_pred']:.4f})")

    if 'checkpoint_info' in results:
        info = results['checkpoint_info']
        print(f"\nCheckpoint Info:")
        print(f"  Path:  {info['path']}")
        print(f"  Epoch: {info['epoch']}")
        print(f"  Step:  {info['step']}")

    print("=" * 60 + "\n")


def save_evaluation_results(
    results: Dict,
    output_path: str,
) -> None:
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def main():
    """Main evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MLLM model")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--labels-file", required=True, help="Labels file")
    parser.add_argument("--output", default="results/evaluation.json", help="Output file")
    parser.add_argument("--device", default="cuda", help="Device")

    args = parser.parse_args()

    # Import here to avoid circular imports
    from data.sft_dataset import get_sft_dataloader
    from models.qwen_vl_64ch import Qwen2VL64Ch

    # Create data loader
    test_loader = get_sft_dataloader(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        split="test",
        batch_size=4,
    )

    # Evaluate
    results = evaluate_checkpoint(
        args.checkpoint,
        Qwen2VL64Ch,
        test_loader,
        args.device,
    )

    # Print and save
    print_evaluation_report(results)
    save_evaluation_results(results, args.output)


if __name__ == "__main__":
    main()
