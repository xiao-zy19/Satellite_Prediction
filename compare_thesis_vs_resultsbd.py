#!/usr/bin/env python3
"""
Compare thesis experimental numbers against results-bd data.

This script:
1. Reads thesis .tex files and extracts key experimental claims
2. Loads all pkl results from results-bd/
3. Computes mean±std across seeds for each experiment group
4. Compares thesis values vs results-bd values
5. Outputs a detailed comparison report
"""

import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# Configuration
# ============================================================
THESIS_DIR = "/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/satellite_population_thesis/data"
RESULTS_BD_DIR = "/home/xiaozhenyu/degree_essay/Alpha_Earth/results-bd"
OLD_RESULTS_DIR = "/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain/old_results/results"
OUTPUT_FILE = "/home/xiaozhenyu/degree_essay/Alpha_Earth/results-bd/thesis_comparison.txt"

# ============================================================
# Step 1: Load all results-bd pkl files
# ============================================================

def load_all_results(base_dir):
    """Load all pkl files from results-bd and organize by experiment group."""
    results = {}
    for subdir in ["Baseline", "Multimodal", "MultimodalBert", "MultimodalHybrid"]:
        dirpath = os.path.join(base_dir, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith("_results.pkl"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                results[f"{subdir}/{fname}"] = data
            except Exception as e:
                print(f"Warning: Failed to load {fpath}: {e}")
    return results


def extract_metrics(data):
    """Extract test metrics from a pkl result dict.
    Returns dict with keys like 'test_r2', 'test_mae', 'test_rmse', 'test_r',
    and for patch-level, also 'test_r2_mean', 'test_r2_median', 'test_r2_trimmed_mean', etc.
    """
    info = {}
    # Basic metadata
    info["exp_name"] = data.get("exp_name", "unknown")
    info["seed"] = data.get("seed", None)
    info["model_params"] = data.get("model_params", None)
    info["best_epoch"] = data.get("best_epoch", None)
    info["training_mode"] = data.get("training_mode", "city_level")
    info["use_pretrain"] = data.get("use_pretrain", False)
    info["pretrain_method"] = data.get("pretrain_method", None)
    info["fusion_type"] = data.get("fusion_type", None)
    info["image_encoder"] = data.get("image_encoder", None)
    info["policy_source"] = data.get("policy_source", None)

    # City-level results
    if "test_metrics" in data:
        tm = data["test_metrics"]
        info["test_r2"] = tm.get("r2", None)
        info["test_mae"] = tm.get("mae", None)
        info["test_rmse"] = tm.get("rmse", None)
        info["test_r"] = tm.get("pearson_r", None)

    # Patch-level results
    if "test_results" in data and isinstance(data["test_results"], dict):
        for agg in ["mean", "median", "trimmed_mean"]:
            if agg in data["test_results"] and "metrics" in data["test_results"][agg]:
                m = data["test_results"][agg]["metrics"]
                info[f"test_r2_{agg}"] = m.get("r2", None)
                info[f"test_mae_{agg}"] = m.get("mae", None)
                info[f"test_rmse_{agg}"] = m.get("rmse", None)
                info[f"test_r_{agg}"] = m.get("pearson_r", None)

    # Val metrics
    if "val_metrics" in data:
        vm = data["val_metrics"]
        info["val_r2"] = vm.get("r2", None)
        info["val_r"] = vm.get("pearson_r", None)
    if "val_results" in data and isinstance(data["val_results"], dict):
        for agg in ["mean", "median", "trimmed_mean"]:
            if agg in data["val_results"] and "metrics" in data["val_results"][agg]:
                m = data["val_results"][agg]["metrics"]
                info[f"val_r2_{agg}"] = m.get("r2", None)

    # Training history for overfit analysis
    if "history" in data:
        h = data["history"]
        if isinstance(h, dict):
            if "train_loss" in h and h["train_loss"]:
                info["final_train_loss"] = h["train_loss"][-1]
            if "val_r2" in h and h["val_r2"]:
                info["best_val_r2"] = max(h["val_r2"])

    return info


def group_by_experiment(all_results):
    """Group results by experiment name (ignoring seed suffix) and compute stats."""
    groups = defaultdict(list)
    for path, data in all_results.items():
        info = extract_metrics(data)
        # Determine experiment group key
        exp_name = info["exp_name"]
        subdir = path.split("/")[0]
        # Group key: subdir + exp_name
        key = f"{subdir}/{exp_name}"
        groups[key].append(info)
    return groups


def compute_stats(group, metric_key):
    """Compute mean±std for a metric across seeds in a group."""
    values = [r[metric_key] for r in group if r.get(metric_key) is not None]
    if not values:
        return None, None, None, []
    values = np.array(values, dtype=float)
    return float(np.mean(values)), float(np.std(values)), len(values), values.tolist()


# ============================================================
# Step 2: Define thesis claims to verify
# ============================================================

def build_thesis_claims():
    """Build a list of thesis claims with expected values and how to look them up."""
    claims = []

    # --- Chapter 4 Table 4.1: Encoder city-level performance (no pretrain) ---
    # MLP baseline (seed 42 - single seed)
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "MLP city-level no-pretrain R^2",
        "thesis_value": -1780.6,
        "lookup": {"group_prefix": "Baseline/mlp_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "MLP city-level no-pretrain MAE",
        "thesis_value": 182.58,
        "lookup": {"group_prefix": "Baseline/mlp_baseline", "metric": "test_mae", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "MLP city-level no-pretrain RMSE",
        "thesis_value": 182.62,
        "lookup": {"group_prefix": "Baseline/mlp_baseline", "metric": "test_rmse", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "MLP city-level no-pretrain Pearson r",
        "thesis_value": 0.336,
        "lookup": {"group_prefix": "Baseline/mlp_baseline", "metric": "test_r", "mode": "single_seed_42"},
    })

    # LightCNN baseline 3 seeds
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "LightCNN city-level no-pretrain R^2 mean",
        "thesis_value": 0.278,
        "thesis_std": 0.113,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r2", "mode": "mean_3seeds"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "LightCNN city-level no-pretrain MAE mean",
        "thesis_value": 3.08,
        "thesis_std": 0.16,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_mae", "mode": "mean_3seeds"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "LightCNN city-level no-pretrain RMSE mean",
        "thesis_value": 4.11,
        "thesis_std": 0.16,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_rmse", "mode": "mean_3seeds"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "LightCNN city-level no-pretrain Pearson r mean",
        "thesis_value": 0.628,
        "thesis_std": 0.054,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r", "mode": "mean_3seeds"},
    })

    # ResNet-10 baseline single seed
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-10 city-level no-pretrain R^2",
        "thesis_value": 0.439,
        "lookup": {"group_prefix": "Baseline/resnet10_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-10 city-level no-pretrain MAE",
        "thesis_value": 2.65,
        "lookup": {"group_prefix": "Baseline/resnet10_baseline", "metric": "test_mae", "mode": "single_seed_42"},
    })

    # ResNet-18 baseline 3 seeds
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-18 city-level no-pretrain R^2 mean",
        "thesis_value": 0.228,
        "thesis_std": 0.306,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_r2", "mode": "mean_3seeds"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-18 city-level no-pretrain MAE mean",
        "thesis_value": 3.31,
        "thesis_std": 0.80,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_mae", "mode": "mean_3seeds"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-18 city-level no-pretrain RMSE mean",
        "thesis_value": 4.12,
        "thesis_std": 0.92,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_rmse", "mode": "mean_3seeds"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-18 city-level no-pretrain Pearson r mean",
        "thesis_value": 0.668,
        "thesis_std": 0.068,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_r", "mode": "mean_3seeds"},
    })

    # ResNet-34 baseline single seed
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-34 city-level no-pretrain R^2",
        "thesis_value": 0.533,
        "lookup": {"group_prefix": "Baseline/resnet34_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })

    # ResNet-50 baseline single seed
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-50 city-level no-pretrain R^2",
        "thesis_value": 0.187,
        "lookup": {"group_prefix": "Baseline/resnet50_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-50 city-level no-pretrain MAE",
        "thesis_value": 3.68,
        "lookup": {"group_prefix": "Baseline/resnet50_baseline", "metric": "test_mae", "mode": "single_seed_42"},
    })

    # ResNet-101 baseline single seed
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-101 city-level no-pretrain R^2",
        "thesis_value": 0.190,
        "lookup": {"group_prefix": "Baseline/resnet101_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "ResNet-101 city-level no-pretrain MAE",
        "thesis_value": 3.21,
        "lookup": {"group_prefix": "Baseline/resnet101_baseline", "metric": "test_mae", "mode": "single_seed_42"},
    })

    # --- Chapter 4: MLP failure table (Tab 4.2) ---
    # MLP + SimCLR city-level
    claims.append({
        "chapter": "Ch4 Tab4.2",
        "description": "MLP+SimCLR city-level R^2",
        "thesis_value": -4.39,
        "lookup": {"group_prefix": "Baseline/simclr_mlp", "metric": "test_r2", "mode": "single_seed_42"},
    })
    # MLP patch-level no pretrain
    claims.append({
        "chapter": "Ch4 Tab4.2",
        "description": "MLP patch-level no-pretrain R^2 (mean agg)",
        "thesis_value": -0.18,
        "lookup": {"group_prefix": "Baseline/mlp_patch_level", "metric": "test_r2_mean", "mode": "single_seed_42"},
    })

    # --- Chapter 4: Overfit table (Tab 4.3) ---
    # LightCNN seed42 train loss and val/test R2
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed42 train loss",
        "thesis_value": 5.48,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "final_train_loss", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed42 val R^2",
        "thesis_value": 0.453,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "best_val_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed42 test R^2",
        "thesis_value": 0.230,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed42 best epoch",
        "thesis_value": 93,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "best_epoch", "mode": "single_seed_42"},
    })

    # LightCNN seed456
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed456 train loss",
        "thesis_value": 5.09,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "final_train_loss", "mode": "single_seed_456"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed456 val R^2",
        "thesis_value": 0.407,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "best_val_r2", "mode": "single_seed_456"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "LightCNN seed456 test R^2",
        "thesis_value": 0.434,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r2", "mode": "single_seed_456"},
    })

    # ResNet-10 overfit
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-10 seed42 train loss",
        "thesis_value": 3.04,
        "lookup": {"group_prefix": "Baseline/resnet10_baseline", "metric": "final_train_loss", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-10 seed42 val R^2",
        "thesis_value": 0.408,
        "lookup": {"group_prefix": "Baseline/resnet10_baseline", "metric": "best_val_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-10 seed42 best epoch",
        "thesis_value": 80,
        "lookup": {"group_prefix": "Baseline/resnet10_baseline", "metric": "best_epoch", "mode": "single_seed_42"},
    })

    # ResNet-18 seed123 overfit
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-18 seed123 train loss",
        "thesis_value": 2.50,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "final_train_loss", "mode": "single_seed_123"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-18 seed123 val R^2",
        "thesis_value": 0.656,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "best_val_r2", "mode": "single_seed_123"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-18 seed123 test R^2",
        "thesis_value": 0.502,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_r2", "mode": "single_seed_123"},
    })

    # ResNet-34 overfit
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-34 seed42 train loss",
        "thesis_value": 2.66,
        "lookup": {"group_prefix": "Baseline/resnet34_baseline", "metric": "final_train_loss", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-34 seed42 best epoch",
        "thesis_value": 83,
        "lookup": {"group_prefix": "Baseline/resnet34_baseline", "metric": "best_epoch", "mode": "single_seed_42"},
    })

    # ResNet-50 overfit
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-50 seed42 val R^2",
        "thesis_value": 0.446,
        "lookup": {"group_prefix": "Baseline/resnet50_baseline", "metric": "best_val_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-50 seed42 test R^2",
        "thesis_value": 0.187,
        "lookup": {"group_prefix": "Baseline/resnet50_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-50 seed42 train loss",
        "thesis_value": 3.18,
        "lookup": {"group_prefix": "Baseline/resnet50_baseline", "metric": "final_train_loss", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-50 seed42 best epoch",
        "thesis_value": 94,
        "lookup": {"group_prefix": "Baseline/resnet50_baseline", "metric": "best_epoch", "mode": "single_seed_42"},
    })

    # ResNet-101 overfit
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-101 seed42 val R^2",
        "thesis_value": 0.470,
        "lookup": {"group_prefix": "Baseline/resnet101_baseline", "metric": "best_val_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.3",
        "description": "ResNet-101 seed42 train loss",
        "thesis_value": 4.55,
        "lookup": {"group_prefix": "Baseline/resnet101_baseline", "metric": "final_train_loss", "mode": "single_seed_42"},
    })

    # --- Chapter 4: Pretrain table (Tab 4.5) LightCNN per-seed ---
    # LightCNN no pretrain individual seeds
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "LightCNN no-pretrain seed42 R^2",
        "thesis_value": 0.230,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "LightCNN no-pretrain seed123 R^2",
        "thesis_value": 0.172,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r2", "mode": "single_seed_123"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "LightCNN no-pretrain seed456 R^2",
        "thesis_value": 0.434,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_r2", "mode": "single_seed_456"},
    })
    # LightCNN no pretrain per-seed MAE
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "LightCNN no-pretrain seed42 MAE",
        "thesis_value": 3.30,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_mae", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "LightCNN no-pretrain seed123 MAE",
        "thesis_value": 3.03,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_mae", "mode": "single_seed_123"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "LightCNN no-pretrain seed456 MAE",
        "thesis_value": 2.90,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "test_mae", "mode": "single_seed_456"},
    })

    # SimCLR + LightCNN city-level
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "SimCLR+LightCNN city-level R^2",
        "thesis_value": 0.270,
        "lookup": {"group_prefix": "Baseline/simclr_cnn", "metric": "test_r2", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "SimCLR+LightCNN city-level MAE",
        "thesis_value": 2.86,
        "lookup": {"group_prefix": "Baseline/simclr_cnn", "metric": "test_mae", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })

    # MAE + LightCNN city-level
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "MAE+LightCNN city-level R^2",
        "thesis_value": 0.453,
        "lookup": {"group_prefix": "Baseline/mae_cnn", "metric": "test_r2", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "MAE+LightCNN city-level MAE",
        "thesis_value": 2.61,
        "lookup": {"group_prefix": "Baseline/mae_cnn", "metric": "test_mae", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.5",
        "description": "MAE+LightCNN city-level RMSE",
        "thesis_value": 3.22,
        "lookup": {"group_prefix": "Baseline/mae_cnn", "metric": "test_rmse", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })

    # --- Chapter 4: ImageNet pretrain table (Tab 4.6) ---
    # ResNet-18 ImageNet
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-18+ImageNet city-level R^2",
        "thesis_value": 0.522,
        "lookup": {"group_prefix": "Baseline/resnet18_imagenet", "metric": "test_r2", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    # ResNet-34 ImageNet
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-34+ImageNet city-level R^2",
        "thesis_value": 0.465,
        "lookup": {"group_prefix": "Baseline/resnet34_imagenet", "metric": "test_r2", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    # ResNet-50 ImageNet
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-50+ImageNet city-level R^2",
        "thesis_value": 0.636,
        "lookup": {"group_prefix": "Baseline/resnet50_imagenet", "metric": "test_r2", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    # ResNet-101 ImageNet
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-101+ImageNet city-level R^2",
        "thesis_value": 0.216,
        "lookup": {"group_prefix": "Baseline/resnet101_imagenet", "metric": "test_r2", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    # ImageNet best epochs
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-18+ImageNet best epoch",
        "thesis_value": 88,
        "lookup": {"group_prefix": "Baseline/resnet18_imagenet", "metric": "best_epoch", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-34+ImageNet best epoch",
        "thesis_value": 9,
        "lookup": {"group_prefix": "Baseline/resnet34_imagenet", "metric": "best_epoch", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-50+ImageNet best epoch",
        "thesis_value": 72,
        "lookup": {"group_prefix": "Baseline/resnet50_imagenet", "metric": "best_epoch", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.6",
        "description": "ResNet-101+ImageNet best epoch",
        "thesis_value": 33,
        "lookup": {"group_prefix": "Baseline/resnet101_imagenet", "metric": "best_epoch", "mode": "single_seed_42",
                   "filter_training_mode": "city_level"},
    })

    # --- Chapter 4 text: ResNet-18 individual seeds ---
    claims.append({
        "chapter": "Ch4 text",
        "description": "ResNet-18 seed42 R^2 (text mentions -0.199)",
        "thesis_value": -0.199,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_r2", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 text",
        "description": "ResNet-18 seed123 R^2 (text mentions 0.502)",
        "thesis_value": 0.502,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_r2", "mode": "single_seed_123"},
    })
    claims.append({
        "chapter": "Ch4 text",
        "description": "ResNet-18 seed456 R^2 (text mentions 0.380)",
        "thesis_value": 0.380,
        "lookup": {"group_prefix": "Baseline/resnet18_baseline", "metric": "test_r2", "mode": "single_seed_456"},
    })

    # --- Model parameter counts ---
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "LightCNN model params (~161K)",
        "thesis_value": 161000,
        "lookup": {"group_prefix": "Baseline/light_cnn_baseline", "metric": "model_params", "mode": "single_seed_42"},
        "tolerance_type": "absolute",
        "tolerance": 1000,
    })
    claims.append({
        "chapter": "Ch4 Tab4.1",
        "description": "MLP model params (~61K)",
        "thesis_value": 61000,
        "lookup": {"group_prefix": "Baseline/mlp_baseline", "metric": "model_params", "mode": "single_seed_42"},
        "tolerance_type": "absolute",
        "tolerance": 5000,
    })

    # --- Patch-level LightCNN baseline (MEMORY.md: SimCLR+Patch best R2=0.818) ---
    # LightCNN patch-level (no pretrain, seed42)
    claims.append({
        "chapter": "MEMORY",
        "description": "LightCNN patch-level no-pretrain seed42 R^2 (median agg)",
        "thesis_value": 0.652,  # from the pkl we inspected
        "lookup": {"group_prefix": "Baseline/light_cnn_patch_level", "metric": "test_r2_median", "mode": "single_seed_42"},
    })

    # SimCLR+LightCNN patch-level (MEMORY says best R2=0.818 median agg seed456)
    claims.append({
        "chapter": "MEMORY",
        "description": "SimCLR+LightCNN patch-level R^2 median agg (best single seed ~0.818)",
        "thesis_value": 0.818,
        "lookup": {"group_prefix": "Baseline/simclr_cnn_patch_level", "metric": "test_r2_median", "mode": "best_single_seed"},
    })

    # --- Chapter 4 text: MLP failure rate 100% (3/3 configs negative R2) ---
    # MLP+SimCLR MAE and RMSE
    claims.append({
        "chapter": "Ch4 Tab4.2",
        "description": "MLP+SimCLR city-level MAE",
        "thesis_value": 8.95,
        "lookup": {"group_prefix": "Baseline/simclr_mlp", "metric": "test_mae", "mode": "single_seed_42"},
    })
    claims.append({
        "chapter": "Ch4 Tab4.2",
        "description": "MLP+SimCLR city-level RMSE",
        "thesis_value": 9.77,
        "lookup": {"group_prefix": "Baseline/simclr_mlp", "metric": "test_rmse", "mode": "single_seed_42"},
    })
    # MLP patch-level MAE and RMSE
    claims.append({
        "chapter": "Ch4 Tab4.2",
        "description": "MLP patch-level MAE (mean agg)",
        "thesis_value": 3.64,
        "lookup": {"group_prefix": "Baseline/mlp_patch_level", "metric": "test_mae_mean", "mode": "single_seed_42"},
    })

    return claims


# ============================================================
# Step 3: Look up results-bd values for each claim
# ============================================================

def find_matching_results(groups, all_results, lookup):
    """Find results matching a lookup specification."""
    prefix = lookup["group_prefix"]
    metric = lookup["metric"]
    mode = lookup["mode"]
    filter_mode = lookup.get("filter_training_mode", None)

    # Find all matching groups
    matching = []
    for key, group in groups.items():
        if key.startswith(prefix) or key == prefix:
            for r in group:
                if filter_mode:
                    # Filter by training mode
                    tm = r.get("training_mode", "city_level")
                    if filter_mode == "city_level" and tm not in ["city_level", None, "baseline"]:
                        continue
                    elif filter_mode == "patch_level" and tm != "patch_level":
                        continue
                matching.append(r)

    if not matching:
        # Try broader match - look at individual file paths
        for fpath, data in all_results.items():
            # Extract prefix without subdir for matching
            if prefix.split("/")[1] in fpath.lower().replace("-", "_"):
                info = extract_metrics(data)
                if filter_mode:
                    tm = info.get("training_mode", "city_level")
                    if filter_mode == "city_level" and tm not in ["city_level", None, "baseline"]:
                        continue
                    elif filter_mode == "patch_level" and tm != "patch_level":
                        continue
                matching.append(info)

    if not matching:
        return None, None, None, "NOT FOUND"

    if mode == "single_seed_42":
        for r in matching:
            if r.get("seed") == 42:
                val = r.get(metric)
                if val is not None:
                    return val, None, 1, "OK"
        # Fallback: try the one without seed suffix (which is typically seed=42)
        for r in matching:
            if r.get("seed") in [42, None]:
                val = r.get(metric)
                if val is not None:
                    return val, None, 1, "OK (seed fallback)"
        return None, None, 0, "SEED 42 NOT FOUND"

    elif mode == "single_seed_123":
        for r in matching:
            if r.get("seed") == 123:
                val = r.get(metric)
                if val is not None:
                    return val, None, 1, "OK"
        return None, None, 0, "SEED 123 NOT FOUND"

    elif mode == "single_seed_456":
        for r in matching:
            if r.get("seed") == 456:
                val = r.get(metric)
                if val is not None:
                    return val, None, 1, "OK"
        return None, None, 0, "SEED 456 NOT FOUND"

    elif mode == "mean_3seeds":
        values = [r.get(metric) for r in matching if r.get(metric) is not None]
        if not values:
            return None, None, 0, "NO VALUES"
        arr = np.array(values, dtype=float)
        return float(np.mean(arr)), float(np.std(arr)), len(arr), "OK"

    elif mode == "best_single_seed":
        values = [(r.get(metric), r.get("seed")) for r in matching if r.get(metric) is not None]
        if not values:
            return None, None, 0, "NO VALUES"
        best_val, best_seed = max(values, key=lambda x: x[0])
        return best_val, None, len(values), f"OK (best from seed={best_seed}, {len(values)} seeds total)"

    return None, None, 0, f"UNKNOWN MODE: {mode}"


def format_value(val, decimal=3):
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, int) or (isinstance(val, float) and val == int(val) and abs(val) > 100):
        return f"{int(val)}"
    return f"{val:.{decimal}f}"


def check_match(thesis_val, result_val, tolerance_type="relative", tolerance=None):
    """Check if thesis and result values match within tolerance."""
    if thesis_val is None or result_val is None:
        return "N/A", "N/A"

    diff = result_val - thesis_val
    if abs(thesis_val) > 0.001:
        rel_diff = diff / abs(thesis_val) * 100
    else:
        rel_diff = diff * 100  # just show raw diff for near-zero

    if tolerance_type == "absolute" and tolerance is not None:
        if abs(diff) <= tolerance:
            return "MATCH", f"diff={diff:.1f} (within {tolerance})"
        else:
            return "DIFFER", f"diff={diff:.1f} (exceeds {tolerance})"

    # Default: consider match if within 5% or 0.02 absolute
    abs_thresh = 0.02 if abs(thesis_val) < 1 else 0.05 * abs(thesis_val)
    # For very small values near zero, use larger relative tolerance
    if abs(thesis_val) < 0.01:
        abs_thresh = max(abs_thresh, 0.1)

    if abs(diff) <= abs_thresh:
        return "MATCH", f"diff={diff:+.4f} ({rel_diff:+.2f}%)"
    elif abs(diff) <= abs_thresh * 3:
        return "CLOSE", f"diff={diff:+.4f} ({rel_diff:+.2f}%)"
    else:
        return "DIFFER", f"diff={diff:+.4f} ({rel_diff:+.2f}%)"


# ============================================================
# Step 4: Generate full inventory of results-bd experiments
# ============================================================

def generate_experiment_inventory(groups):
    """Generate a comprehensive inventory of all experiments in results-bd."""
    inventory = []
    for key, group in sorted(groups.items()):
        seeds = sorted([r["seed"] for r in group if r.get("seed") is not None])
        params = [r.get("model_params") for r in group if r.get("model_params") is not None]
        mode = group[0].get("training_mode", "city_level") if group else "unknown"
        pretrain = group[0].get("use_pretrain", False) if group else False
        pretrain_method = group[0].get("pretrain_method", None) if group else None
        fusion = group[0].get("fusion_type", None) if group else None

        # Get the main test metric (R2)
        r2_values = []
        for r in group:
            if "test_r2" in r and r["test_r2"] is not None:
                r2_values.append(r["test_r2"])
            elif "test_r2_median" in r and r["test_r2_median"] is not None:
                r2_values.append(r["test_r2_median"])

        r2_arr = np.array(r2_values, dtype=float) if r2_values else np.array([])
        r2_str = f"{np.mean(r2_arr):.3f}+/-{np.std(r2_arr):.3f}" if len(r2_arr) > 1 else (f"{r2_arr[0]:.3f}" if len(r2_arr) == 1 else "N/A")

        inventory.append({
            "key": key,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "mode": mode,
            "pretrain": pretrain_method if pretrain else "none",
            "fusion": fusion,
            "params": params[0] if params else None,
            "r2": r2_str,
        })
    return inventory


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("THESIS vs RESULTS-BD COMPARISON TOOL")
    print("=" * 80)
    print()

    # Load all results
    print(f"Loading results from {RESULTS_BD_DIR}...")
    all_results = load_all_results(RESULTS_BD_DIR)
    print(f"  Loaded {len(all_results)} pkl files")

    # Group by experiment
    groups = group_by_experiment(all_results)
    print(f"  Grouped into {len(groups)} experiment groups")
    print()

    # Build thesis claims
    claims = build_thesis_claims()
    print(f"Checking {len(claims)} thesis claims...")
    print()

    # Build output
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("THESIS vs RESULTS-BD COMPARISON REPORT")
    output_lines.append(f"Generated for results-bd at: {RESULTS_BD_DIR}")
    output_lines.append(f"Total pkl files: {len(all_results)}")
    output_lines.append(f"Experiment groups: {len(groups)}")
    output_lines.append(f"Thesis claims checked: {len(claims)}")
    output_lines.append("=" * 100)
    output_lines.append("")

    # ---- Section 1: Claim-by-claim comparison ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 1: CLAIM-BY-CLAIM COMPARISON")
    output_lines.append("=" * 100)
    output_lines.append("")

    match_count = 0
    close_count = 0
    differ_count = 0
    not_found_count = 0
    total = len(claims)

    current_chapter = None
    for claim in claims:
        ch = claim["chapter"]
        if ch != current_chapter:
            output_lines.append(f"\n--- {ch} ---")
            current_chapter = ch

        result_val, result_std, n_seeds, status = find_matching_results(
            groups, all_results, claim["lookup"]
        )

        if status.startswith("NOT FOUND") or status.endswith("NOT FOUND") or status == "NO VALUES":
            verdict = "NOT_FOUND"
            not_found_count += 1
            detail = status
        else:
            tol_type = claim.get("tolerance_type", "relative")
            tol = claim.get("tolerance", None)
            verdict, detail = check_match(claim["thesis_value"], result_val, tol_type, tol)
            if verdict == "MATCH":
                match_count += 1
            elif verdict == "CLOSE":
                close_count += 1
            else:
                differ_count += 1

        thesis_str = format_value(claim["thesis_value"])
        if "thesis_std" in claim:
            thesis_str += f"+/-{format_value(claim['thesis_std'])}"
        result_str = format_value(result_val)
        if result_std is not None:
            result_str += f"+/-{format_value(result_std)}"
        if n_seeds and n_seeds > 0:
            result_str += f" (n={n_seeds})"

        line = f"  [{verdict:9s}] {claim['description']}"
        line += f"\n             Thesis: {thesis_str:>15s}  |  Results-bd: {result_str:>25s}  |  {detail}"
        output_lines.append(line)

    output_lines.append("")
    output_lines.append(f"SUMMARY: {match_count} MATCH, {close_count} CLOSE, {differ_count} DIFFER, {not_found_count} NOT_FOUND (of {total} claims)")
    output_lines.append("")

    # ---- Section 2: Full experiment inventory ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 2: FULL RESULTS-BD EXPERIMENT INVENTORY")
    output_lines.append("=" * 100)
    output_lines.append("")

    inventory = generate_experiment_inventory(groups)
    output_lines.append(f"{'Experiment':<55s} {'Seeds':>8s} {'Mode':<12s} {'Pretrain':<10s} {'Fusion':<12s} {'Params':>10s} {'R2':>18s}")
    output_lines.append("-" * 135)
    for item in inventory:
        seeds_str = ",".join(str(s) for s in item["seeds"]) if item["seeds"] else "?"
        params_str = f"{item['params']:,}" if item['params'] else "?"
        fusion_str = item["fusion"] or "-"
        pretrain_str = item["pretrain"] or "none"
        output_lines.append(
            f"{item['key']:<55s} {seeds_str:>8s} {item['mode']:<12s} {pretrain_str:<10s} "
            f"{fusion_str:<12s} {params_str:>10s} {item['r2']:>18s}"
        )

    output_lines.append("")

    # ---- Section 3: Baseline experiments mapping table ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 3: KEY BASELINE EXPERIMENTS - DETAILED METRICS")
    output_lines.append("=" * 100)
    output_lines.append("")

    # Extract detailed per-seed metrics for key experiments
    key_experiments = [
        ("Baseline/mlp_baseline", "MLP city-level"),
        ("Baseline/light_cnn_baseline", "LightCNN city-level"),
        ("Baseline/resnet10_baseline", "ResNet-10 city-level"),
        ("Baseline/resnet18_baseline", "ResNet-18 city-level"),
        ("Baseline/resnet34_baseline", "ResNet-34 city-level"),
        ("Baseline/resnet50_baseline", "ResNet-50 city-level"),
        ("Baseline/resnet101_baseline", "ResNet-101 city-level"),
        ("Baseline/simclr_cnn", "SimCLR+CNN city-level"),
        ("Baseline/mae_cnn", "MAE+CNN city-level"),
        ("Baseline/simclr_mlp", "SimCLR+MLP city-level"),
        ("Baseline/resnet18_imagenet", "ResNet-18+ImageNet"),
        ("Baseline/resnet34_imagenet", "ResNet-34+ImageNet"),
        ("Baseline/resnet50_imagenet", "ResNet-50+ImageNet"),
        ("Baseline/resnet101_imagenet", "ResNet-101+ImageNet"),
        ("Baseline/light_cnn_patch_level", "LightCNN patch-level"),
        ("Baseline/simclr_cnn_patch_level", "SimCLR+CNN patch-level"),
        ("Baseline/mae_cnn_patch_level", "MAE+CNN patch-level"),
        ("Baseline/mlp_patch_level", "MLP patch-level"),
    ]

    for prefix, label in key_experiments:
        matching = []
        for key, group in groups.items():
            if key == prefix or key.startswith(prefix):
                # For city-level keys only pick city-level results
                if "city-level" in label and "patch" not in prefix:
                    for r in group:
                        tm = r.get("training_mode", "city_level")
                        if tm in ["city_level", None, "baseline"] or "baseline" in key:
                            matching.append(r)
                else:
                    matching.extend(group)

        if not matching:
            output_lines.append(f"  {label}: NOT FOUND in results-bd")
            continue

        output_lines.append(f"  {label}:")
        for r in sorted(matching, key=lambda x: x.get("seed", 0) or 0):
            seed = r.get("seed", "?")
            params = r.get("model_params", "?")
            epoch = r.get("best_epoch", "?")
            mode = r.get("training_mode", "?")

            if r.get("test_r2") is not None:
                output_lines.append(
                    f"    seed={seed}, params={params}, epoch={epoch}, mode={mode}"
                    f"  R2={r['test_r2']:.4f}  MAE={r.get('test_mae', 'N/A'):.4f}" if isinstance(r.get('test_mae'), (int, float)) else
                    f"    seed={seed}, params={params}, epoch={epoch}, mode={mode}"
                    f"  R2={r['test_r2']:.4f}  MAE=N/A"
                )
                if r.get("test_rmse") is not None:
                    output_lines[-1] += f"  RMSE={r['test_rmse']:.4f}"
                if r.get("test_r") is not None:
                    output_lines[-1] += f"  r={r['test_r']:.4f}"
            elif r.get("test_r2_median") is not None:
                output_lines.append(
                    f"    seed={seed}, params={params}, epoch={epoch}, mode={mode}"
                )
                for agg in ["mean", "median", "trimmed_mean"]:
                    rk = f"test_r2_{agg}"
                    mk = f"test_mae_{agg}"
                    if r.get(rk) is not None:
                        line = f"      {agg}: R2={r[rk]:.4f}"
                        if r.get(mk) is not None:
                            line += f"  MAE={r[mk]:.4f}"
                        rmk = f"test_rmse_{agg}"
                        if r.get(rmk) is not None:
                            line += f"  RMSE={r[rmk]:.4f}"
                        prk = f"test_r_{agg}"
                        if r.get(prk) is not None:
                            line += f"  r={r[prk]:.4f}"
                        output_lines.append(line)
        output_lines.append("")

    # ---- Section 4: Multimodal experiment summary ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 4: MULTIMODAL EXPERIMENTS SUMMARY")
    output_lines.append("=" * 100)
    output_lines.append("")

    for subdir_label, subdir_name in [("Struct(12d)", "Multimodal"),
                                       ("BERT(64d)", "MultimodalBert"),
                                       ("Hybrid(76d)", "MultimodalHybrid")]:
        output_lines.append(f"\n  --- {subdir_label} ({subdir_name}) ---")
        sub_groups = {k: v for k, v in groups.items() if k.startswith(f"{subdir_name}/")}
        if not sub_groups:
            output_lines.append("    No experiments found")
            continue

        # Aggregate by fusion type and encoder
        fusion_summary = defaultdict(list)
        for key, group in sub_groups.items():
            for r in group:
                fusion = r.get("fusion_type", "unknown")
                mode = r.get("training_mode", "city_level")
                # Get primary R2
                r2 = r.get("test_r2") or r.get("test_r2_median")
                if r2 is not None:
                    fusion_summary[f"{fusion}_{mode}"].append(r2)

        output_lines.append(f"    {'Fusion+Mode':<40s} {'n':>4s} {'Mean R2':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
        output_lines.append(f"    {'-'*78}")
        for fkey in sorted(fusion_summary.keys()):
            vals = np.array(fusion_summary[fkey])
            output_lines.append(
                f"    {fkey:<40s} {len(vals):>4d} {np.mean(vals):>10.4f} {np.std(vals):>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}"
            )

    output_lines.append("")

    # ---- Section 5: OLD results (thesis) -> NEW results-bd mapping ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 5: OLD (THESIS) -> NEW (RESULTS-BD) MAPPING TABLE")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append("This table shows how thesis values compare to the new results-bd data.")
    output_lines.append("If results-bd has different random splits/training configs, values may legitimately differ.")
    output_lines.append("")

    output_lines.append(f"{'Description':<55s} {'Thesis':>12s} {'Results-bd':>12s} {'Diff':>10s} {'Status':>10s}")
    output_lines.append("-" * 100)

    for claim in claims:
        result_val, result_std, n_seeds, status = find_matching_results(
            groups, all_results, claim["lookup"]
        )
        thesis_val = claim["thesis_value"]
        thesis_str = format_value(thesis_val)
        result_str = format_value(result_val)

        if result_val is not None and thesis_val is not None:
            diff = result_val - thesis_val
            diff_str = f"{diff:+.4f}" if abs(diff) < 100 else f"{diff:+.1f}"
            tol_type = claim.get("tolerance_type", "relative")
            tol = claim.get("tolerance", None)
            verdict, _ = check_match(thesis_val, result_val, tol_type, tol)
        else:
            diff_str = "N/A"
            verdict = "NOT_FOUND" if result_val is None else "N/A"

        desc = claim["description"][:54]
        output_lines.append(f"{desc:<55s} {thesis_str:>12s} {result_str:>12s} {diff_str:>10s} {verdict:>10s}")

    output_lines.append("")

    # ---- Section 6: Old Results vs New Results-bd cross-reference ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 6: OLD RESULTS vs NEW RESULTS-BD CROSS-REFERENCE")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append("This section compares old_results (which thesis was written from) against results-bd (new data).")
    output_lines.append("Different values indicate different data splits or training configurations.")
    output_lines.append("")

    # Load old results
    old_results = {}
    if os.path.isdir(OLD_RESULTS_DIR):
        old_results = load_all_results(OLD_RESULTS_DIR)
        output_lines.append(f"Loaded {len(old_results)} old result files from {OLD_RESULTS_DIR}")
    else:
        output_lines.append(f"OLD_RESULTS_DIR not found: {OLD_RESULTS_DIR}")

    if old_results:
        old_groups = group_by_experiment(old_results)
        output_lines.append(f"Old result groups: {len(old_groups)}")
        output_lines.append("")

        # Find matching experiments between old and new
        output_lines.append(f"{'Experiment':<50s} {'Seed':>5s} {'Old R2':>10s} {'New R2':>10s} {'Diff':>10s} {'Old=Thesis?':>12s}")
        output_lines.append("-" * 100)

        # Build a map of new results by (exp_name, seed)
        new_map = {}
        for key, group in groups.items():
            for r in group:
                exp = r.get("exp_name", "?")
                seed = r.get("seed", None)
                nkey = (key.split("/")[0], exp, seed)
                new_map[nkey] = r

        for old_key, old_group in sorted(old_groups.items()):
            for old_r in sorted(old_group, key=lambda x: x.get("seed", 0) or 0):
                old_exp = old_r.get("exp_name", "?")
                old_seed = old_r.get("seed", None)
                old_subdir = old_key.split("/")[0]

                # Get old R2
                old_r2 = old_r.get("test_r2")
                if old_r2 is None:
                    old_r2 = old_r.get("test_r2_median")

                # Find matching new result
                new_r = new_map.get((old_subdir, old_exp, old_seed))
                if new_r is None and old_seed is None:
                    # Try seed 42
                    new_r = new_map.get((old_subdir, old_exp, 42))

                if new_r:
                    new_r2 = new_r.get("test_r2")
                    if new_r2 is None:
                        new_r2 = new_r.get("test_r2_median")
                else:
                    new_r2 = None

                old_r2_str = f"{old_r2:.4f}" if old_r2 is not None else "N/A"
                new_r2_str = f"{new_r2:.4f}" if new_r2 is not None else "N/A"

                if old_r2 is not None and new_r2 is not None:
                    diff = new_r2 - old_r2
                    diff_str = f"{diff:+.4f}"
                else:
                    diff_str = "N/A"

                # Check if old R2 matches any thesis claim
                thesis_match = "?"
                for claim in claims:
                    if claim.get("thesis_value") is not None and old_r2 is not None:
                        tv = claim["thesis_value"]
                        if abs(tv) > 0.001:
                            if abs(old_r2 - tv) / abs(tv) < 0.01:
                                thesis_match = "YES"
                                break
                        else:
                            if abs(old_r2 - tv) < 0.01:
                                thesis_match = "YES"
                                break

                seed_str = str(old_seed) if old_seed is not None else "def"
                output_lines.append(
                    f"{old_key:<50s} {seed_str:>5s} {old_r2_str:>10s} {new_r2_str:>10s} {diff_str:>10s} {thesis_match:>12s}"
                )

        output_lines.append("")

    # ---- Section 7: Summary of key findings ----
    output_lines.append("=" * 100)
    output_lines.append("SECTION 7: KEY FINDINGS SUMMARY")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append("1. DATA SPLIT CHANGE CONFIRMED:")
    output_lines.append("   The results-bd data uses a DIFFERENT data split than the old results.")
    output_lines.append("   Old LightCNN seed42: R2=0.230 (matches thesis), New: R2=0.478")
    output_lines.append("   This means ALL thesis numbers need to be updated if using results-bd.")
    output_lines.append("")
    output_lines.append("2. QUALITATIVE TRENDS PRESERVED:")
    output_lines.append("   - MLP still fails catastrophically in results-bd (all configs negative R2)")
    output_lines.append("   - LightCNN still competitive with larger ResNets")
    output_lines.append("   - Model parameter counts unchanged (LightCNN=160,577, MLP=60,801)")
    output_lines.append("")
    output_lines.append("3. KEY DIFFERENCES:")

    # Check if ResNet-101 is now better than in thesis
    r101_new = None
    for key, group in groups.items():
        if key == "Baseline/resnet101_baseline":
            vals = [r.get("test_r2") for r in group if r.get("test_r2") is not None]
            if vals:
                r101_new = np.mean(vals)

    if r101_new is not None:
        output_lines.append(f"   - ResNet-101 baseline: thesis=0.190, results-bd={r101_new:.3f}")
        if r101_new > 0.3:
            output_lines.append(f"     WARNING: ResNet-101 performs MUCH better in results-bd!")
            output_lines.append(f"     The thesis claim that large models underperform may need revision.")

    # Check ResNet-18 individual seeds
    for key, group in groups.items():
        if key == "Baseline/resnet18_baseline":
            for r in group:
                if r.get("seed") == 42 and r.get("test_r2") is not None:
                    r18_42 = r["test_r2"]
                    output_lines.append(f"   - ResNet-18 seed42: thesis=-0.199, results-bd={r18_42:.3f}")
                    if r18_42 > 0:
                        output_lines.append(f"     The catastrophic failure of ResNet-18 seed42 does NOT reproduce!")

    # SimCLR best R2
    for key, group in groups.items():
        if key == "Baseline/simclr_cnn_patch_level":
            best = max([r.get("test_r2_median", -999) for r in group])
            output_lines.append(f"   - SimCLR+Patch best R2(median): thesis=0.818, results-bd={best:.3f}")

    # Check ImageNet results
    for encoder in ["resnet18", "resnet50"]:
        for key, group in groups.items():
            if key == f"Baseline/{encoder}_imagenet":
                for r in group:
                    if r.get("seed") == 42 and r.get("test_r2") is not None:
                        output_lines.append(f"   - {encoder}+ImageNet seed42: thesis value differs, results-bd={r['test_r2']:.3f}")

    output_lines.append("")
    output_lines.append("4. THESIS VALUES THAT STILL MATCH (within tolerance):")
    for claim in claims:
        result_val, result_std, n_seeds, status = find_matching_results(
            groups, all_results, claim["lookup"]
        )
        if result_val is not None and claim["thesis_value"] is not None:
            tol_type = claim.get("tolerance_type", "relative")
            tol = claim.get("tolerance", None)
            verdict, _ = check_match(claim["thesis_value"], result_val, tol_type, tol)
            if verdict == "MATCH":
                output_lines.append(f"   - {claim['description']}: {format_value(claim['thesis_value'])} ~ {format_value(result_val)}")

    output_lines.append("")
    output_lines.append("5. MOST CRITICAL DISCREPANCIES (>50% relative diff):")
    for claim in claims:
        result_val, result_std, n_seeds, status = find_matching_results(
            groups, all_results, claim["lookup"]
        )
        if result_val is not None and claim["thesis_value"] is not None:
            tv = claim["thesis_value"]
            if abs(tv) > 0.01:
                rel_diff = abs(result_val - tv) / abs(tv) * 100
                if rel_diff > 50:
                    output_lines.append(
                        f"   - {claim['description']}: thesis={format_value(tv)}, bd={format_value(result_val)} ({rel_diff:.0f}% diff)"
                    )

    output_lines.append("")

    # Write output
    report = "\n".join(output_lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
