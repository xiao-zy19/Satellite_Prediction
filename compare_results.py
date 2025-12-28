"""
Compare results from all experiments
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

import config


def load_all_results():
    """Load results from all experiments."""
    results = {}
    result_dir = Path(config.RESULT_DIR)

    for pkl_file in result_dir.glob("*_results.pkl"):
        exp_name = pkl_file.stem.replace("_results", "")
        try:
            with open(pkl_file, 'rb') as f:
                results[exp_name] = pickle.load(f)
            print(f"Loaded: {exp_name}")
        except Exception as e:
            print(f"Error loading {exp_name}: {e}")

    return results


def create_comparison_table(results):
    """Create comparison table of all experiments."""
    rows = []

    for exp_name, data in results.items():
        test_metrics = data.get('test_metrics', {})
        rows.append({
            'Experiment': exp_name,
            'Pearson r': test_metrics.get('pearson_r', np.nan),
            'R²': test_metrics.get('r2', np.nan),
            'MAE': test_metrics.get('mae', np.nan),
            'RMSE': test_metrics.get('rmse', np.nan),
            'Best Epoch': data.get('best_epoch', np.nan),
            'Parameters': data.get('model_params', np.nan)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Pearson r', ascending=False)

    return df


def plot_comparison(results, output_path):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    exp_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))

    # Plot 1: Test Pearson r comparison (bar chart)
    ax = axes[0, 0]
    pearson_rs = [results[exp].get('test_metrics', {}).get('pearson_r', 0) for exp in exp_names]
    bars = ax.barh(exp_names, pearson_rs, color=colors)
    ax.set_xlabel('Pearson r')
    ax.set_title('Test Pearson r Comparison')
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, pearson_rs):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

    # Plot 2: Test R² comparison
    ax = axes[0, 1]
    r2s = [results[exp].get('test_metrics', {}).get('r2', 0) for exp in exp_names]
    bars = ax.barh(exp_names, r2s, color=colors)
    ax.set_xlabel('R²')
    ax.set_title('Test R² Comparison')
    for bar, val in zip(bars, r2s):
        ax.text(max(0, val) + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

    # Plot 3: Validation loss curves
    ax = axes[1, 0]
    for i, exp_name in enumerate(exp_names):
        history = results[exp_name].get('history', {})
        val_loss = history.get('val_loss', [])
        if val_loss:
            ax.plot(val_loss, label=exp_name, color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curves')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Validation Pearson r curves
    ax = axes[1, 1]
    for i, exp_name in enumerate(exp_names):
        history = results[exp_name].get('history', {})
        val_r = history.get('val_pearson_r', [])
        if val_r:
            ax.plot(val_r, label=exp_name, color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pearson r')
    ax.set_title('Validation Pearson r Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def plot_predictions(results, output_path):
    """Plot predicted vs true values for all experiments."""
    n_exp = len(results)
    cols = min(3, n_exp)
    rows = (n_exp + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_exp == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (exp_name, data) in enumerate(results.items()):
        ax = axes[i]
        y_true = data.get('test_y_true', [])
        y_pred = data.get('test_y_pred', [])
        test_metrics = data.get('test_metrics', {})

        if len(y_true) > 0:
            ax.scatter(y_pred, y_true, alpha=0.6, s=30, c='steelblue')

            # Perfect line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            r = test_metrics.get('pearson_r', 0)
            r2 = test_metrics.get('r2', 0)
            ax.set_title(f'{exp_name}\nr={r:.3f}, R²={r2:.3f}')
            ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to: {output_path}")


def main():
    print("=" * 60)
    print("Comparing Experiment Results")
    print("=" * 60)

    # Load results
    results = load_all_results()

    if not results:
        print("No results found!")
        return

    # Create comparison table
    df = create_comparison_table(results)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save table
    table_path = os.path.join(config.RESULT_DIR, 'comparison_table.csv')
    df.to_csv(table_path, index=False)
    print(f"\nTable saved to: {table_path}")

    # Create plots
    plot_comparison(results, os.path.join(config.RESULT_DIR, 'comparison_plot.png'))
    plot_predictions(results, os.path.join(config.RESULT_DIR, 'predictions_plot.png'))

    # Print best model
    best_exp = df.iloc[0]['Experiment']
    best_r = df.iloc[0]['Pearson r']
    print(f"\nBest model: {best_exp} (Pearson r = {best_r:.4f})")


if __name__ == "__main__":
    main()
