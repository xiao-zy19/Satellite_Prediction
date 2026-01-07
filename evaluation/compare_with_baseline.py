"""
Compare MLLM results with Baseline_Pretrain results.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_baseline_results(
    results_dir: str = "../Baseline_Pretrain/results",
) -> Dict[str, Dict]:
    """
    Load results from Baseline_Pretrain experiments.

    Args:
        results_dir: Directory containing baseline result files

    Returns:
        Dictionary of experiment name -> results
    """
    results_dir = Path(results_dir)
    results = {}

    # Load pickle files
    for pkl_file in results_dir.glob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                exp_results = pickle.load(f)
                results[pkl_file.stem] = exp_results
        except Exception as e:
            logger.warning(f"Could not load {pkl_file}: {e}")

    # Load JSON files
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                exp_results = json.load(f)
                results[json_file.stem] = exp_results
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")

    logger.info(f"Loaded {len(results)} baseline experiment results")
    return results


def load_mllm_results(
    results_dir: str = "results",
) -> Dict[str, Dict]:
    """
    Load results from MLLM experiments.

    Args:
        results_dir: Directory containing MLLM result files

    Returns:
        Dictionary of experiment name -> results
    """
    results_dir = Path(results_dir)
    results = {}

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                exp_results = json.load(f)
                results[json_file.stem] = exp_results
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")

    logger.info(f"Loaded {len(results)} MLLM experiment results")
    return results


def compare_with_baseline(
    mllm_results: Dict,
    baseline_results: Dict,
    metrics: List[str] = ['r2', 'mae', 'rmse', 'pearson_r'],
) -> pd.DataFrame:
    """
    Compare MLLM results with baseline results.

    Args:
        mllm_results: MLLM experiment results
        baseline_results: Baseline experiment results
        metrics: Metrics to compare

    Returns:
        Comparison DataFrame
    """
    comparison_data = []

    # Add MLLM results
    for exp_name, results in mllm_results.items():
        if 'metrics' in results:
            row = {
                'experiment': exp_name,
                'method': 'MLLM',
            }
            for metric in metrics:
                row[metric] = results['metrics'].get(metric, None)
            comparison_data.append(row)

    # Add baseline results
    for exp_name, results in baseline_results.items():
        # Handle different result formats
        if isinstance(results, dict):
            if 'test_metrics' in results:
                metrics_dict = results['test_metrics']
            elif 'metrics' in results:
                metrics_dict = results['metrics']
            else:
                metrics_dict = results
        else:
            continue

        row = {
            'experiment': exp_name,
            'method': 'Baseline',
        }
        for metric in metrics:
            row[metric] = metrics_dict.get(metric, None)
        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by R² score
    if 'r2' in df.columns:
        df = df.sort_values('r2', ascending=False)

    return df


def print_comparison_report(
    comparison_df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("MLLM vs Baseline Comparison Report")
    print("=" * 80)

    # Summary statistics
    mllm_df = comparison_df[comparison_df['method'] == 'MLLM']
    baseline_df = comparison_df[comparison_df['method'] == 'Baseline']

    print(f"\nNumber of experiments:")
    print(f"  MLLM:     {len(mllm_df)}")
    print(f"  Baseline: {len(baseline_df)}")

    # Best results from each method
    if not mllm_df.empty and 'r2' in mllm_df.columns:
        best_mllm = mllm_df.loc[mllm_df['r2'].idxmax()]
        print(f"\nBest MLLM Result:")
        print(f"  Experiment: {best_mllm['experiment']}")
        print(f"  R²: {best_mllm['r2']:.4f}")
        if 'mae' in best_mllm:
            print(f"  MAE: {best_mllm['mae']:.4f}")
        if 'rmse' in best_mllm:
            print(f"  RMSE: {best_mllm['rmse']:.4f}")

    if not baseline_df.empty and 'r2' in baseline_df.columns:
        best_baseline = baseline_df.loc[baseline_df['r2'].idxmax()]
        print(f"\nBest Baseline Result:")
        print(f"  Experiment: {best_baseline['experiment']}")
        print(f"  R²: {best_baseline['r2']:.4f}")
        if 'mae' in best_baseline:
            print(f"  MAE: {best_baseline['mae']:.4f}")
        if 'rmse' in best_baseline:
            print(f"  RMSE: {best_baseline['rmse']:.4f}")

    # Comparison
    if not mllm_df.empty and not baseline_df.empty:
        mllm_best_r2 = mllm_df['r2'].max()
        baseline_best_r2 = baseline_df['r2'].max()

        print(f"\nR² Comparison:")
        print(f"  Best MLLM:     {mllm_best_r2:.4f}")
        print(f"  Best Baseline: {baseline_best_r2:.4f}")
        print(f"  Difference:    {mllm_best_r2 - baseline_best_r2:+.4f}")

        if mllm_best_r2 > baseline_best_r2:
            print("  Winner: MLLM")
        else:
            print("  Winner: Baseline")

    # Top N results
    print(f"\nTop {top_n} Results (by R²):")
    print("-" * 80)

    top_df = comparison_df.head(top_n)
    for idx, row in top_df.iterrows():
        print(f"  {row['experiment']:40s} [{row['method']:8s}] R²={row['r2']:.4f}")

    print("=" * 80 + "\n")


def generate_comparison_plots(
    comparison_df: pd.DataFrame,
    output_dir: str = "results/plots",
) -> None:
    """
    Generate comparison plots.

    Args:
        comparison_df: Comparison DataFrame
        output_dir: Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # R² comparison bar plot
    plt.figure(figsize=(12, 6))

    # Get top experiments from each method
    mllm_top = comparison_df[comparison_df['method'] == 'MLLM'].nlargest(5, 'r2')
    baseline_top = comparison_df[comparison_df['method'] == 'Baseline'].nlargest(5, 'r2')

    top_df = pd.concat([mllm_top, baseline_top])

    sns.barplot(
        data=top_df,
        x='experiment',
        y='r2',
        hue='method',
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('R² Score Comparison: MLLM vs Baseline')
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png', dpi=150)
    plt.close()

    # Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['r2', 'mae', 'rmse', 'pearson_r']
    for ax, metric in zip(axes.flat, metrics):
        if metric in comparison_df.columns:
            sns.boxplot(
                data=comparison_df,
                x='method',
                y=metric,
                ax=ax,
            )
            ax.set_title(f'{metric.upper()} Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
    plt.close()

    logger.info(f"Saved plots to {output_dir}")


def save_comparison_results(
    comparison_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save comparison results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    comparison_df.to_csv(output_path.with_suffix('.csv'), index=False)

    # Save as JSON
    comparison_df.to_json(output_path.with_suffix('.json'), orient='records', indent=2)

    logger.info(f"Saved comparison results to {output_path}")


def main():
    """Main comparison script."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare MLLM with Baseline")
    parser.add_argument("--mllm-results", default="results", help="MLLM results directory")
    parser.add_argument("--baseline-results",
                        default="../Baseline_Pretrain/results",
                        help="Baseline results directory")
    parser.add_argument("--output", default="results/comparison", help="Output path")
    parser.add_argument("--generate-plots", action="store_true", help="Generate comparison plots")

    args = parser.parse_args()

    # Load results
    mllm_results = load_mllm_results(args.mllm_results)
    baseline_results = load_baseline_results(args.baseline_results)

    # Compare
    comparison_df = compare_with_baseline(mllm_results, baseline_results)

    # Print report
    print_comparison_report(comparison_df)

    # Save results
    save_comparison_results(comparison_df, args.output)

    # Generate plots
    if args.generate_plots:
        generate_comparison_plots(comparison_df, str(Path(args.output).parent / "plots"))


if __name__ == "__main__":
    main()
