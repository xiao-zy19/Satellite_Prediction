"""
Policy Feature Contribution Analysis for Chapter 5.

Three inference-time analyses on the best multimodal model (MAE+FiLM+Patch+Structured):
  1. Group-level counterfactual sensitivity (appendix table)
  2. Per-feature permutation importance (main table)
  3. FiLM gamma/beta weight analysis (static + dynamic + linear decomposition)

Baseline note:
  The bytedance-results pkl files store LAST-EPOCH test evaluation, not the
  BEST-EPOCH evaluation. Root cause: model.state_dict() returns tensors sharing
  storage with the live model; continued training after best epoch mutates these
  in-place, so when the trainer reloads self.best_model_state at training end,
  it actually loads last-epoch weights. The on-disk checkpoint (best_model.pth)
  was saved at best epoch before corruption and is correct.
  Verified on the 3 target seeds: pkl['test_results'] == pkl['history'][-1],
  while pkl['history'][best_epoch-1] matches checkpoint-recomputed values.
  A verify_baselines() function confirms this cross-reference.

  Reported baselines use pkl values (~0.685) to keep consistent with Ch5 main text.

Usage:
    cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain
    python analysis/policy_feature_analysis.py --gpu 0 --task all
    python analysis/policy_feature_analysis.py --gpu 0 --task permutation
    python analysis/policy_feature_analysis.py --gpu 0 --task group_sensitivity
    python analysis/policy_feature_analysis.py --gpu 0 --task film_analysis
"""

import sys
import os
import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import compute_metrics, set_seed
from config_multimodal import get_multimodal_experiment_config
from train_multimodal import create_multimodal_model
from dataset_policy import get_patch_level_policy_dataloaders
from policy_features import get_policy_extractor, CITY_TO_PROVINCE

# ── Constants ──────────────────────────────────────────────────────────────────

CKPT_BASE = "/home/xiaozhenyu/degree_essay/Alpha_Earth/bytedance-results/checkpoints"
SEED_CKPT_MAP = {
    42:  f"{CKPT_BASE}/mm_mae_cnn_film_patch/best_model.pth",
    123: f"{CKPT_BASE}/mm_mae_cnn_film_patch_seed123/best_model.pth",
    456: f"{CKPT_BASE}/mm_mae_cnn_film_patch_seed456/best_model.pth",
}
SEEDS = [42, 123, 456]
EXP_NAME = "mm_mae_cnn_film_patch"
NUM_PATCHES = 25

# PKL paths for baseline verification
PKL_BASE = "/home/xiaozhenyu/degree_essay/Alpha_Earth/bytedance-results/results/Multimodal"
SEED_PKL_MAP = {
    42:  f"{PKL_BASE}/mm_mae_cnn_film_patch_results.pkl",
    123: f"{PKL_BASE}/mm_mae_cnn_film_patch_seed123_results.pkl",
    456: f"{PKL_BASE}/mm_mae_cnn_film_patch_seed456_results.pkl",
}

PKL_BASELINES = {}  # {seed: pkl_r2}, loaded at startup by load_pkl_baselines()


def load_pkl_baselines():
    """Load baseline R² from pkl files (Ch5 paper口径)."""
    global PKL_BASELINES
    for seed, pkl_path in SEED_PKL_MAP.items():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        PKL_BASELINES[seed] = data['test_results']['median']['metrics']['r2']


FEATURE_NAMES = [
    "maternity_leave_days",
    "paternity_leave_days",
    "parental_leave_days",
    "marriage_leave_days",
    "birth_subsidy_amount",
    "housing_subsidy_amount",
    "childcare_subsidy_monthly",
    "tax_deduction_monthly",
    "has_childcare_policy",
    "has_ivf_insurance",
    "is_minority_favorable",
    "policy_phase",
]

FEATURE_NAMES_CN = [
    "产假天数",
    "陪产假天数",
    "育儿假天数",
    "婚假天数",
    "生育补贴金额",
    "住房补贴金额",
    "托育补贴月额",
    "个税抵扣月额",
    "有托育政策",
    "辅助生殖纳入医保",
    "少数民族优惠",
    "政策阶段",
]

# Standard 3-group version (matches Ch5 main text)
FEATURE_GROUPS_3 = {
    "假期制度 [0-3]": [0, 1, 2, 3],
    "经济激励 [4-7]": [4, 5, 6, 7],
    "社会保障 [8-11]": [8, 9, 10, 11],
}

# Sensitivity 4-group version (phase split out)
FEATURE_GROUPS_4 = {
    "假期制度 [0-3]": [0, 1, 2, 3],
    "经济激励 [4-7]": [4, 5, 6, 7],
    "社保(不含phase) [8-10]": [8, 9, 10],
    "政策阶段 [11]": [11],
}


# ── Audit metadata ─────────────────────────────────────────────────────────────

def build_audit_metadata():
    """Build metadata dict for result reproducibility."""
    return {
        'timestamp': datetime.now().isoformat(),
        'script': str(Path(__file__).resolve()),
        'exp_name': EXP_NAME,
        'seeds': SEEDS,
        'checkpoint_paths': {s: SEED_CKPT_MAP[s] for s in SEEDS},
        'pkl_paths': {s: SEED_PKL_MAP[s] for s in SEEDS},
        'aggregation_method': 'median',
        'num_patches': NUM_PATCHES,
        'feature_names': FEATURE_NAMES,
        'feature_names_cn': FEATURE_NAMES_CN,
        'baseline_note': (
            'Reported baselines use pkl values (~0.685) to keep consistent with Ch5 main text. '
            'Inference uses best-epoch checkpoint model. pkl_baselines and checkpoint_baselines '
            'are both stored for audit. The pkl files store last-epoch evaluation; the on-disk '
            'checkpoint stores best-epoch weights (verified via verify_baselines()).'
        ),
    }


# ── Baseline verification ─────────────────────────────────────────────────────

def verify_baselines(recomputed_baselines=None):
    """Verify that our checkpoint-recomputed baselines match pkl history at best_epoch.

    Loads each pkl file, extracts history['test_median_r2'] at best_epoch and at
    last epoch, and compares with our recomputed values.

    Args:
        recomputed_baselines: optional {seed: r2} dict. If None, only prints pkl info.

    Returns:
        verification: dict with per-seed breakdown and pass/fail status
    """
    print("\n" + "=" * 70)
    print("Baseline Verification: pkl history vs checkpoint-recomputed")
    print("=" * 70)

    verification = {}
    all_pass = True

    for seed in SEEDS:
        pkl_path = SEED_PKL_MAP[seed]
        if not os.path.exists(pkl_path):
            print(f"  WARNING: pkl not found for seed {seed}: {pkl_path}")
            verification[seed] = {'status': 'pkl_missing'}
            all_pass = False
            continue

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        best_epoch = data.get('best_epoch', None)
        history = data.get('history', {})
        test_median_history = history.get('test_median_r2', [])

        # pkl stored value (last-epoch evaluation)
        pkl_stored_r2 = data['test_results']['median']['metrics']['r2']

        # Extract from history
        last_epoch_r2 = test_median_history[-1] if test_median_history else None
        best_epoch_r2 = test_median_history[best_epoch - 1] if (best_epoch and test_median_history) else None

        entry = {
            'best_epoch': best_epoch,
            'total_epochs': len(test_median_history),
            'pkl_stored_r2': pkl_stored_r2,
            'history_last_epoch_r2': last_epoch_r2,
            'history_best_epoch_r2': best_epoch_r2,
            'pkl_matches_last': abs(pkl_stored_r2 - last_epoch_r2) < 1e-6 if last_epoch_r2 else False,
        }

        if recomputed_baselines and seed in recomputed_baselines:
            recomp = recomputed_baselines[seed]
            entry['recomputed_r2'] = recomp
            entry['recomp_matches_best'] = abs(recomp - best_epoch_r2) < 0.002 if best_epoch_r2 else False
            if not entry['recomp_matches_best']:
                all_pass = False

        verification[seed] = entry

    # Print comparison table
    print(f"\n  {'Seed':<8s} {'Best Ep':<8s} {'Total Ep':<9s} "
          f"{'PKL stored':>12s} {'Hist[-1]':>12s} {'Hist[best]':>12s}", end="")
    if recomputed_baselines:
        print(f" {'Recomputed':>12s} {'Match?':>8s}", end="")
    print()
    print("  " + "-" * 90)

    for seed in SEEDS:
        v = verification[seed]
        if v.get('status') == 'pkl_missing':
            print(f"  {seed:<8d} PKL NOT FOUND")
            continue
        print(f"  {seed:<8d} {v['best_epoch']:<8d} {v['total_epochs']:<9d} "
              f"{v['pkl_stored_r2']:12.6f} {v['history_last_epoch_r2']:12.6f} "
              f"{v['history_best_epoch_r2']:12.6f}", end="")
        if recomputed_baselines and seed in recomputed_baselines:
            match_str = "OK" if v.get('recomp_matches_best', False) else "FAIL"
            print(f" {v['recomputed_r2']:12.6f} {match_str:>8s}", end="")
        print()

    print(f"\n  PKL stores last-epoch eval (pkl == history[-1]): "
          f"{'CONFIRMED' if all(v.get('pkl_matches_last', False) for v in verification.values() if v.get('status') != 'pkl_missing') else 'FAILED'}")
    if recomputed_baselines:
        print(f"  Checkpoint matches best-epoch history: "
              f"{'ALL PASS' if all_pass else 'SOME FAILED'}")

    verification['all_pass'] = all_pass
    return verification


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(seed, device):
    """Load model from checkpoint for a given seed."""
    exp_config = get_multimodal_experiment_config(EXP_NAME)
    model = create_multimodal_model(exp_config.model_config, patch_level=True)
    ckpt_path = SEED_CKPT_MAP[seed]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_test_loader_and_info(seed, batch_size=64, num_workers=2,
                             temporal_split=False, train_years=None,
                             val_years=None, test_years=None):
    """Get test dataloader and dataset info for a given seed."""
    _, _, test_loader, dataset_info = get_patch_level_policy_dataloaders(
        batch_size=batch_size, num_workers=num_workers,
        augment_train=False, seed=seed,
        temporal_split=temporal_split,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years
    )
    return test_loader, dataset_info


def collect_patch_predictions(model, test_loader, device):
    """Run model on test set, collect per-sample patch predictions.

    Returns:
        sample_predictions: {sample_idx: {'predictions': {patch_idx: float},
                                           'label': float, 'city': str, 'year': int}}
    """
    model.eval()
    sample_predictions = {}
    with torch.no_grad():
        for patches, policy_feat, labels, info in tqdm(test_loader, desc="Inference", leave=False):
            patches = patches.to(device)
            policy_feat = policy_feat.to(device)
            outputs = model(patches, policy_feat)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            for i in range(patches.size(0)):
                sidx = info['sample_idx'][i].item() if torch.is_tensor(info['sample_idx'][i]) else info['sample_idx'][i]
                pidx = info['patch_idx'][i].item() if torch.is_tensor(info['patch_idx'][i]) else info['patch_idx'][i]
                if sidx not in sample_predictions:
                    sample_predictions[sidx] = {
                        'predictions': {},
                        'label': labels[i].item(),
                        'city': info['city'][i],
                        'year': info['year'][i].item() if torch.is_tensor(info['year'][i]) else info['year'][i],
                    }
                sample_predictions[sidx]['predictions'][pidx] = outputs[i].item()
    return sample_predictions


def aggregate_to_city(sample_predictions, method='median'):
    """Aggregate patch predictions to city-level."""
    y_true, y_pred, sample_info = [], [], []
    for sidx in sorted(sample_predictions.keys()):
        data = sample_predictions[sidx]
        if len(data['predictions']) != NUM_PATCHES:
            continue
        preds = [data['predictions'][p] for p in range(NUM_PATCHES)]
        if method == 'median':
            agg = np.median(preds)
        elif method == 'mean':
            agg = np.mean(preds)
        else:
            sorted_p = np.sort(preds)
            n_trim = max(1, int(len(sorted_p) * 0.1))
            agg = np.mean(sorted_p[n_trim:-n_trim])
        y_true.append(data['label'])
        y_pred.append(agg)
        sample_info.append({'city': data['city'], 'year': data['year']})
    return np.array(y_true), np.array(y_pred), sample_info


def cache_image_features(model, test_loader, device):
    """Cache encoder+proj features for all test patches.

    Returns:
        cached: {sample_idx: {patch_idx: tensor(64,)}}
        meta: {sample_idx: {'label': float, 'city': str, 'year': int,
                             'policy_feat': tensor(12,)}}
    """
    model.eval()
    cached = defaultdict(dict)
    meta = {}
    with torch.no_grad():
        for patches, policy_feat, labels, info in tqdm(test_loader, desc="Caching image features", leave=False):
            patches = patches.to(device)
            # Encoder
            image_feat = model.image_encoder(patches)
            if model.image_proj is not None:
                image_feat = model.image_proj(image_feat)
            # image_feat: (batch, 64)
            for i in range(patches.size(0)):
                sidx = info['sample_idx'][i].item() if torch.is_tensor(info['sample_idx'][i]) else info['sample_idx'][i]
                pidx = info['patch_idx'][i].item() if torch.is_tensor(info['patch_idx'][i]) else info['patch_idx'][i]
                cached[sidx][pidx] = image_feat[i].cpu()
                if sidx not in meta:
                    meta[sidx] = {
                        'label': labels[i].item(),
                        'city': info['city'][i],
                        'year': info['year'][i].item() if torch.is_tensor(info['year'][i]) else info['year'][i],
                        'policy_feat': policy_feat[i].cpu(),  # (12,)
                    }
    return dict(cached), meta


def predict_from_cache(model, cached, meta, policy_override=None, device='cpu'):
    """Run fusion+head on cached image features with optional policy override.

    Args:
        policy_override: if provided, {sample_idx: tensor(12,)} overrides policy per sample.
                        If a single tensor(12,), broadcast to all samples.

    Returns:
        sample_predictions in same format as collect_patch_predictions
    """
    model.eval()
    sample_predictions = {}

    # Determine override type
    broadcast_policy = None
    if policy_override is not None and isinstance(policy_override, torch.Tensor) and policy_override.dim() == 1:
        broadcast_policy = policy_override
        policy_override = None

    with torch.no_grad():
        for sidx in sorted(cached.keys()):
            if len(cached[sidx]) != NUM_PATCHES:
                continue

            # Stack patches: (25, 64)
            img_feats = torch.stack([cached[sidx][p] for p in range(NUM_PATCHES)]).to(device)

            # Policy feature
            if broadcast_policy is not None:
                pf = broadcast_policy.unsqueeze(0).expand(NUM_PATCHES, -1).to(device)
            elif policy_override is not None and sidx in policy_override:
                pf = policy_override[sidx].unsqueeze(0).expand(NUM_PATCHES, -1).to(device)
            else:
                pf = meta[sidx]['policy_feat'].unsqueeze(0).expand(NUM_PATCHES, -1).to(device)

            # Policy encoder (pass-through for this model)
            policy_encoded = model.policy_encoder(pf)
            fused = model.fusion(img_feats, policy_encoded)
            outputs = model.regression_head(fused)  # (25, 1)

            sample_predictions[sidx] = {
                'predictions': {p: outputs[p].item() for p in range(NUM_PATCHES)},
                'label': meta[sidx]['label'],
                'city': meta[sidx]['city'],
                'year': meta[sidx]['year'],
            }
    return sample_predictions


def compute_train_mean(dataset_info):
    """Compute training set mean policy features (province-year deduplicated)."""
    extractor = get_policy_extractor()
    seen = set()
    unique_policies = []
    for sample in dataset_info['train_samples']:
        city = sample['city']
        year = sample['year']
        province = CITY_TO_PROVINCE.get(city, city)
        policy_year = year - extractor.policy_lag
        key = (province, policy_year)
        if key not in seen:
            seen.add(key)
            feat = extractor.get_normalized_features(city, year)
            unique_policies.append(feat)
    train_mean = np.stack(unique_policies).mean(axis=0)
    print(f"  Train mean computed from {len(unique_policies)} unique (province, policy_year) pairs")
    return train_mean


# ── Experiment 1: Group Counterfactual Sensitivity ─────────────────────────────

def _run_group_sensitivity_one_grouping(group_dict, group_label, device, all_cached, all_meta, all_models, all_dataset_info):
    """Run sensitivity for one grouping scheme. Returns (all_results, summary)."""
    print(f"\n  --- Grouping: {group_label} ---")

    all_results = {}
    for seed in SEEDS:
        model = all_models[seed]
        cached = all_cached[seed]
        meta = all_meta[seed]
        dataset_info = all_dataset_info[seed]

        baseline_preds = predict_from_cache(model, cached, meta, device=device)
        y_true, y_pred, _ = aggregate_to_city(baseline_preds, method='median')
        checkpoint_r2 = compute_metrics(y_true, y_pred)['r2']
        pkl_r2 = PKL_BASELINES[seed]

        train_mean = torch.FloatTensor(compute_train_mean(dataset_info))

        seed_results = {'baseline_r2': pkl_r2, 'checkpoint_r2': checkpoint_r2}

        for group_name, indices in group_dict.items():
            group_results = {}
            for strategy_name, fill_value in [('zero', 0.0), ('train_mean', train_mean)]:
                policy_override = {}
                for sidx in cached.keys():
                    pf = meta[sidx]['policy_feat'].clone()
                    for idx in indices:
                        if strategy_name == 'zero':
                            pf[idx] = 0.0
                        else:
                            pf[idx] = fill_value[idx]
                    policy_override[sidx] = pf

                preds = predict_from_cache(model, cached, meta,
                                           policy_override=policy_override, device=device)
                yt, yp, _ = aggregate_to_city(preds, method='median')
                r2 = compute_metrics(yt, yp)['r2']
                group_results[strategy_name] = r2
                print(f"    Seed {seed} | {group_name} | {strategy_name}: R²={r2:.4f} (Δ={r2 - pkl_r2:+.4f})")

            seed_results[group_name] = group_results
        all_results[seed] = seed_results

    # Summary (using pkl baselines)
    baseline_vals = [PKL_BASELINES[s] for s in SEEDS]
    summary = {}
    for group_name in group_dict:
        zero_vals = [all_results[s][group_name]['zero'] for s in SEEDS]
        mean_vals = [all_results[s][group_name]['train_mean'] for s in SEEDS]
        delta_zero = [z - b for z, b in zip(zero_vals, baseline_vals)]
        delta_mean = [m - b for m, b in zip(mean_vals, baseline_vals)]
        summary[group_name] = {
            'baseline_mean': float(np.mean(baseline_vals)),
            'baseline_std': float(np.std(baseline_vals)),
            'zero_mean': float(np.mean(zero_vals)),
            'zero_std': float(np.std(zero_vals)),
            'train_mean_mean': float(np.mean(mean_vals)),
            'train_mean_std': float(np.std(mean_vals)),
            'delta_zero_mean': float(np.mean(delta_zero)),
            'delta_zero_std': float(np.std(delta_zero)),
            'delta_mean_mean': float(np.mean(delta_mean)),
            'delta_mean_std': float(np.std(delta_mean)),
        }

    return all_results, summary


def run_group_sensitivity(device, output_dir):
    """Group-level counterfactual sensitivity analysis with both 3-group and 4-group schemes."""
    print("\n" + "=" * 70)
    print("Experiment 1: Group Counterfactual Sensitivity Analysis")
    print("=" * 70)

    # Pre-load all models and cache features (shared across both groupings)
    all_models = {}
    all_cached = {}
    all_meta = {}
    all_dataset_info = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed}: loading model and caching features ---")
        set_seed(seed)
        all_models[seed] = load_model(seed, device)
        test_loader, dataset_info = get_test_loader_and_info(seed)
        cached, meta = cache_image_features(all_models[seed], test_loader, device)
        all_cached[seed] = cached
        all_meta[seed] = meta
        all_dataset_info[seed] = dataset_info

        # Print baseline (pkl口径)
        baseline_preds = predict_from_cache(all_models[seed], cached, meta, device=device)
        y_true, y_pred, _ = aggregate_to_city(baseline_preds, method='median')
        checkpoint_r2 = compute_metrics(y_true, y_pred)['r2']
        print(f"  Baseline R² (pkl口径): {PKL_BASELINES[seed]:.4f}  (checkpoint: {checkpoint_r2:.4f})")

    # Run standard 3-group
    raw_3, summary_3 = _run_group_sensitivity_one_grouping(
        FEATURE_GROUPS_3, "Standard 3-group", device,
        all_cached, all_meta, all_models, all_dataset_info)

    # Run sensitivity 4-group (phase split out)
    raw_4, summary_4 = _run_group_sensitivity_one_grouping(
        FEATURE_GROUPS_4, "Sensitivity 4-group (phase split)", device,
        all_cached, all_meta, all_models, all_dataset_info)

    # Print combined summary
    for label, summary, groups in [("Standard 3-Group", summary_3, FEATURE_GROUPS_3),
                                    ("Sensitivity 4-Group", summary_4, FEATURE_GROUPS_4)]:
        print(f"\n{'=' * 70}")
        print(f"{label} Summary (3-seed mean ± std)")
        print(f"{'=' * 70}")
        print(f"{'Group':<28s} {'Baseline R²':>14s} {'Zero R²':>14s} {'TrainMean R²':>14s} {'ΔZero':>14s} {'ΔMean':>14s}")
        print("-" * 98)
        for gn in groups:
            s = summary[gn]
            print(f"{gn:<28s} "
                  f"{s['baseline_mean']:.3f}±{s['baseline_std']:.3f}   "
                  f"{s['zero_mean']:.3f}±{s['zero_std']:.3f}   "
                  f"{s['train_mean_mean']:.3f}±{s['train_mean_std']:.3f}   "
                  f"{s['delta_zero_mean']:+.3f}±{s['delta_zero_std']:.3f}   "
                  f"{s['delta_mean_mean']:+.3f}±{s['delta_mean_std']:.3f}")

    # Verify baselines against pkl history
    checkpoint_per_seed = {s: raw_3[s]['checkpoint_r2'] for s in SEEDS}
    verification = verify_baselines(recomputed_baselines=checkpoint_per_seed)

    # Save
    save_data = {
        'metadata': build_audit_metadata(),
        'baseline_r2_per_seed': {s: PKL_BASELINES[s] for s in SEEDS},
        'checkpoint_r2_per_seed': checkpoint_per_seed,
        'baseline_verification': verification,
        'group_definitions': {
            'standard_3': {k: v for k, v in FEATURE_GROUPS_3.items()},
            'sensitivity_4': {k: v for k, v in FEATURE_GROUPS_4.items()},
        },
        'replacement_strategies': ['zero', 'train_mean'],
        'standard_3_group': {'raw': raw_3, 'summary': summary_3},
        'sensitivity_4_group': {'raw': raw_4, 'summary': summary_4},
    }
    save_path = output_dir / "group_sensitivity_results.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {save_path}")

    return save_data


# ── Experiment 2: Permutation Importance ───────────────────────────────────────

def _compute_permutability(year_groups, prov_year_policy, feat_j):
    """Compute what fraction of province-year pairs can be effectively permuted for feature j.

    Returns:
        permutable_fraction: fraction of unique (province, policy_year) pairs in groups
                            with >=2 distinct values for feature j
        unique_value_counts: {policy_year: n_unique_values}
    """
    total = 0
    permutable = 0
    unique_value_counts = {}
    for py, keys in year_groups.items():
        values = [prov_year_policy[k][feat_j] for k in keys]
        n_unique = len(set(values))
        unique_value_counts[py] = n_unique
        total += len(keys)
        if n_unique > 1:
            permutable += len(keys)
    return permutable / total if total > 0 else 0, unique_value_counts


def run_permutation_importance(device, output_dir, n_repeats=30):
    """Per-feature permutation importance with province-year semantic permutation."""
    print("\n" + "=" * 70)
    print("Experiment 2: Permutation Importance (12 features × 30 repeats × 3 seeds)")
    print("=" * 70)

    extractor = get_policy_extractor()
    all_results = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        model = load_model(seed, device)
        test_loader, dataset_info = get_test_loader_and_info(seed)

        # Cache image features
        cached, meta = cache_image_features(model, test_loader, device)

        # Baseline
        baseline_preds = predict_from_cache(model, cached, meta, device=device)
        y_true, y_pred, _ = aggregate_to_city(baseline_preds, method='median')
        checkpoint_r2 = compute_metrics(y_true, y_pred)['r2']
        pkl_r2 = PKL_BASELINES[seed]
        print(f"  Baseline R² (pkl口径): {pkl_r2:.4f}  (checkpoint: {checkpoint_r2:.4f})")

        # Build province-year mapping for test samples
        sample_to_prov_year = {}
        prov_year_policy = {}
        for sidx in sorted(cached.keys()):
            city = meta[sidx]['city']
            year = meta[sidx]['year']
            province = CITY_TO_PROVINCE.get(city, city)
            policy_year = year - extractor.policy_lag
            key = (province, policy_year)
            sample_to_prov_year[sidx] = key
            if key not in prov_year_policy:
                prov_year_policy[key] = meta[sidx]['policy_feat'].numpy().copy()

        # Group by policy_year
        year_groups = defaultdict(list)
        for key in prov_year_policy:
            year_groups[key[1]].append(key)

        print(f"  Test set: {len(cached)} samples, {len(prov_year_policy)} unique (province, policy_year)")
        for py, keys in sorted(year_groups.items()):
            print(f"    policy_year={py}: {len(keys)} provinces")

        # Permutability diagnostic per feature
        permutability = {}
        for feat_j in range(12):
            frac, uvc = _compute_permutability(year_groups, prov_year_policy, feat_j)
            permutability[feat_j] = {'permutable_fraction': frac, 'unique_per_year': uvc}
        print(f"\n  Permutability (fraction of pairs in groups with >=2 distinct values):")
        for feat_j in range(12):
            frac = permutability[feat_j]['permutable_fraction']
            flag = " *** LOW" if frac < 0.5 else ""
            print(f"    {FEATURE_NAMES[feat_j]:<25s}: {frac:.1%}{flag}")

        # Permutation for each feature
        seed_importance = {}
        for feat_j in range(12):
            repeat_r2s = []
            for rep in range(n_repeats):
                # Permute feature j within each policy_year group
                permuted_prov_year = {}
                rng = np.random.RandomState(seed * 1000 + feat_j * 100 + rep)
                for py, keys in year_groups.items():
                    values_j = [prov_year_policy[k][feat_j] for k in keys]
                    permuted_j = rng.permutation(values_j)
                    for k, v in zip(keys, permuted_j):
                        if k not in permuted_prov_year:
                            permuted_prov_year[k] = prov_year_policy[k].copy()
                        permuted_prov_year[k][feat_j] = v

                # Map back to samples
                policy_override = {}
                for sidx, key in sample_to_prov_year.items():
                    policy_override[sidx] = torch.FloatTensor(permuted_prov_year[key])

                preds = predict_from_cache(model, cached, meta,
                                           policy_override=policy_override, device=device)
                yt, yp, _ = aggregate_to_city(preds, method='median')
                r2 = compute_metrics(yt, yp)['r2']
                repeat_r2s.append(r2)

            importance = pkl_r2 - np.mean(repeat_r2s)
            within_std = np.std(repeat_r2s)
            seed_importance[feat_j] = {
                'importance': importance,
                'mean_permuted_r2': np.mean(repeat_r2s),
                'within_seed_std': within_std,
                'all_r2s': repeat_r2s,
                'permutable_fraction': permutability[feat_j]['permutable_fraction'],
            }
            print(f"  Feature {feat_j:2d} ({FEATURE_NAMES[feat_j]:<25s}): "
                  f"importance={importance:+.4f} (permuted R²={np.mean(repeat_r2s):.4f}±{within_std:.4f})"
                  f"  [permutable={permutability[feat_j]['permutable_fraction']:.0%}]")

        all_results[seed] = {
            'baseline_r2': pkl_r2,
            'checkpoint_r2': checkpoint_r2,
            'features': seed_importance,
            'permutability': permutability,
            'num_test_samples': len(cached),
            'num_unique_prov_year': len(prov_year_policy),
            'year_group_sizes': {py: len(keys) for py, keys in year_groups.items()},
        }

    # Cross-seed summary
    print("\n" + "=" * 70)
    print("Permutation Importance Summary (3-seed mean ± seed std)")
    print("=" * 70)

    summary = []
    for feat_j in range(12):
        imp_vals = [all_results[s]['features'][feat_j]['importance'] for s in SEEDS]
        within_stds = [all_results[s]['features'][feat_j]['within_seed_std'] for s in SEEDS]
        perm_fracs = [all_results[s]['features'][feat_j]['permutable_fraction'] for s in SEEDS]
        avg_perm_frac = float(np.mean(perm_fracs))
        identifiable = avg_perm_frac >= 0.5
        summary.append({
            'feature_idx': feat_j,
            'feature_name': FEATURE_NAMES[feat_j],
            'feature_name_cn': FEATURE_NAMES_CN[feat_j],
            'importance_mean': float(np.mean(imp_vals)),
            'importance_seed_std': float(np.std(imp_vals)),
            'avg_within_seed_std': float(np.mean(within_stds)),
            'avg_permutable_fraction': avg_perm_frac,
            'identifiable': identifiable,
        })

    # Sort: identifiable features first (by importance desc), then non-identifiable (by importance desc)
    summary.sort(key=lambda x: (x['identifiable'], x['importance_mean']), reverse=True)

    identifiable_features = [s for s in summary if s['identifiable']]
    unidentifiable_features = [s for s in summary if not s['identifiable']]

    print(f"{'Rank':<5s} {'Feature':<25s} {'Importance (mean±seed_std)':>30s} {'Avg within-seed std':>22s} {'Permutable':>12s}")
    print("-" * 100)

    rank = 1
    for item in identifiable_features:
        print(f"{rank:<5d} {item['feature_name']:<25s} "
              f"{item['importance_mean']:+.4f} ± {item['importance_seed_std']:.4f}          "
              f"{item['avg_within_seed_std']:.4f}       "
              f"{item['avg_permutable_fraction']:8.0%}")
        rank += 1

    if unidentifiable_features:
        print("-" * 100)
        print("  Below: features with <50% permutable pairs (†). Importance underestimated; use")
        print("  group sensitivity or FiLM decomposition for these features.")
        print("-" * 100)
        for item in unidentifiable_features:
            print(f"  †   {item['feature_name']:<25s} "
                  f"{item['importance_mean']:+.4f} ± {item['importance_seed_std']:.4f}          "
                  f"{item['avg_within_seed_std']:.4f}       "
                  f"{item['avg_permutable_fraction']:8.0%}")

    # Save
    checkpoint_per_seed = {s: all_results[s]['checkpoint_r2'] for s in SEEDS}
    verification = verify_baselines(recomputed_baselines=checkpoint_per_seed)
    save_data = {
        'metadata': build_audit_metadata(),
        'baseline_r2_per_seed': {s: PKL_BASELINES[s] for s in SEEDS},
        'checkpoint_r2_per_seed': checkpoint_per_seed,
        'baseline_verification': verification,
        'permutation_design': {
            'n_repeats': n_repeats,
            'permutation_unit': 'province-policy_year (within policy_year group)',
            'note_policy_phase': (
                'policy_phase is largely constant within a policy_year, so '
                'within-year permutation has very few exchangeable units. Its '
                'importance is likely underestimated by this method.'
            ),
        },
        'raw': all_results,
        'summary': summary,
    }
    save_path = output_dir / "permutation_importance_results.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {save_path}")

    return save_data


# ── Experiment 3: FiLM Weight Analysis ─────────────────────────────────────────

def run_film_analysis(device, output_dir):
    """FiLM gamma/beta weight analysis: static, dynamic, and linear decomposition."""
    print("\n" + "=" * 70)
    print("Experiment 3: FiLM Gamma/Beta Weight Analysis")
    print("=" * 70)

    extractor = get_policy_extractor()
    all_results = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        model = load_model(seed, device)
        test_loader, dataset_info = get_test_loader_and_info(seed)

        # ── 3a: Static weight analysis ──
        W_gamma = model.fusion.gamma_net.weight.detach().cpu()  # (64, 12)
        b_gamma = model.fusion.gamma_net.bias.detach().cpu()    # (64,)
        W_beta = model.fusion.beta_net.weight.detach().cpu()    # (64, 12)
        b_beta = model.fusion.beta_net.bias.detach().cpu()      # (64,)

        gamma_l2 = torch.norm(W_gamma, dim=0).numpy()  # (12,)
        beta_l2 = torch.norm(W_beta, dim=0).numpy()    # (12,)
        gamma_l2_pct = gamma_l2 / gamma_l2.sum()
        beta_l2_pct = beta_l2 / beta_l2.sum()

        print("\n  3a. Static Weight L2 Norms:")
        print(f"  {'Feature':<25s} {'Gamma L2':>10s} {'%':>7s} {'Beta L2':>10s} {'%':>7s}")
        print("  " + "-" * 62)
        for j in range(12):
            print(f"  {FEATURE_NAMES[j]:<25s} {gamma_l2[j]:10.4f} {gamma_l2_pct[j]:6.1%} "
                  f"{beta_l2[j]:10.4f} {beta_l2_pct[j]:6.1%}")

        # ── Baseline R² (for audit consistency with other experiments) ──
        cached, meta = cache_image_features(model, test_loader, device)
        baseline_preds = predict_from_cache(model, cached, meta, device=device)
        y_true, y_pred, _ = aggregate_to_city(baseline_preds, method='median')
        checkpoint_r2 = compute_metrics(y_true, y_pred)['r2']
        pkl_r2 = PKL_BASELINES[seed]
        print(f"  Baseline R² (pkl口径): {pkl_r2:.4f}  (checkpoint: {checkpoint_r2:.4f})")

        # ── 3b: Dynamic modulation analysis ──

        unique_prov_year = {}
        for sidx in cached.keys():
            city = meta[sidx]['city']
            year = meta[sidx]['year']
            province = CITY_TO_PROVINCE.get(city, city)
            policy_year = year - extractor.policy_lag
            key = (province, policy_year)
            if key not in unique_prov_year:
                unique_prov_year[key] = {
                    'policy_feat': meta[sidx]['policy_feat'],
                    'year': year,
                    'policy_year': policy_year,
                }

        print(f"\n  3b. Dynamic Modulation ({len(unique_prov_year)} unique province-year pairs):")

        all_gammas = []
        all_betas = []
        year_gammas = defaultdict(list)
        year_betas = defaultdict(list)

        for key, info in unique_prov_year.items():
            pf = info['policy_feat']
            gamma = W_gamma @ pf + b_gamma
            beta = W_beta @ pf + b_beta
            all_gammas.append(gamma.numpy())
            all_betas.append(beta.numpy())
            year_gammas[info['year']].append(gamma.numpy())
            year_betas[info['year']].append(beta.numpy())

        all_gammas = np.stack(all_gammas)
        all_betas = np.stack(all_betas)

        print(f"  Gamma: mean={all_gammas.mean():.4f}, std={all_gammas.std():.4f}, "
              f"min={all_gammas.min():.4f}, max={all_gammas.max():.4f}")
        print(f"  Beta:  mean={all_betas.mean():.4f}, std={all_betas.std():.4f}, "
              f"min={all_betas.min():.4f}, max={all_betas.max():.4f}")

        # Modulation amplitude per sample
        modulation_ratios = []
        for sidx in sorted(cached.keys()):
            if len(cached[sidx]) != NUM_PATCHES:
                continue
            pf = meta[sidx]['policy_feat']
            gamma = W_gamma @ pf + b_gamma
            beta = W_beta @ pf + b_beta
            for pidx in range(NUM_PATCHES):
                img = cached[sidx][pidx]
                modulated = gamma * img + beta
                ratio = torch.norm(modulated - img).item() / (torch.norm(img).item() + 1e-8)
                modulation_ratios.append({
                    'ratio': ratio,
                    'year': meta[sidx]['year'],
                })
        ratios_arr = np.array([r['ratio'] for r in modulation_ratios])
        print(f"\n  Modulation amplitude ||mod - img|| / ||img||:")
        print(f"    Overall: mean={ratios_arr.mean():.4f}, std={ratios_arr.std():.4f}")

        print(f"\n  {'Year':>6s} {'Gamma mean':>12s} {'Gamma std':>12s} {'Mod. ratio':>12s}")
        print("  " + "-" * 45)
        year_mod_stats = {}
        for yr in sorted(year_gammas.keys()):
            g = np.stack(year_gammas[yr])
            yr_ratios = [r['ratio'] for r in modulation_ratios if r['year'] == yr]
            year_mod_stats[yr] = {
                'gamma_mean': float(g.mean()),
                'gamma_std': float(g.std()),
                'mod_ratio_mean': float(np.mean(yr_ratios)) if yr_ratios else 0,
            }
            print(f"  {yr:>6d} {g.mean():12.4f} {g.std():12.4f} {np.mean(yr_ratios):12.4f}")

        # ── 3c: Linear contribution decomposition ──
        print(f"\n  3c. Linear Contribution Decomposition:")

        gamma_contributions = np.zeros((len(unique_prov_year), 12))
        beta_contributions = np.zeros((len(unique_prov_year), 12))
        policy_values = np.zeros((len(unique_prov_year), 12))

        for i, (key, info) in enumerate(unique_prov_year.items()):
            pf = info['policy_feat'].numpy()
            policy_values[i] = pf
            for j in range(12):
                gamma_contrib_j = pf[j] * W_gamma[:, j].numpy()
                beta_contrib_j = pf[j] * W_beta[:, j].numpy()
                gamma_contributions[i, j] = np.linalg.norm(gamma_contrib_j)
                beta_contributions[i, j] = np.linalg.norm(beta_contrib_j)

        # Verification: sum of contributions + bias should equal gamma_net output
        sample_pf = list(unique_prov_year.values())[0]['policy_feat']
        gamma_sum = sum(sample_pf[j].item() * W_gamma[:, j] for j in range(12)) + b_gamma
        gamma_direct = W_gamma @ sample_pf + b_gamma
        decomp_error = torch.norm(gamma_sum - gamma_direct).item()
        print(f"  Decomposition verification error: {decomp_error:.2e}")

        mean_gamma_contrib = gamma_contributions.mean(axis=0)
        mean_beta_contrib = beta_contributions.mean(axis=0)

        gamma_pct = mean_gamma_contrib / mean_gamma_contrib.sum()
        beta_pct = mean_beta_contrib / mean_beta_contrib.sum()

        policy_var = policy_values.var(axis=0)
        gamma_var_contrib = policy_var * gamma_l2 ** 2
        beta_var_contrib = policy_var * beta_l2 ** 2

        print(f"\n  {'Feature':<25s} {'γ Abs Contrib':>14s} {'γ %':>7s} {'β Abs Contrib':>14s} {'β %':>7s} {'γ Var Contrib':>14s}")
        print("  " + "-" * 85)
        for j in range(12):
            print(f"  {FEATURE_NAMES[j]:<25s} {mean_gamma_contrib[j]:14.4f} {gamma_pct[j]:6.1%} "
                  f"{mean_beta_contrib[j]:14.4f} {beta_pct[j]:6.1%} {gamma_var_contrib[j]:14.6f}")

        # Store results for this seed
        all_results[seed] = {
            'baseline_r2': pkl_r2,
            'checkpoint_r2': checkpoint_r2,
            'num_test_samples': len(cached),
            'num_unique_prov_year': len(unique_prov_year),
            'static': {
                'W_gamma': W_gamma.numpy(),
                'b_gamma': b_gamma.numpy(),
                'W_beta': W_beta.numpy(),
                'b_beta': b_beta.numpy(),
                'gamma_l2': gamma_l2,
                'beta_l2': beta_l2,
            },
            'dynamic': {
                'all_gammas': all_gammas,
                'all_betas': all_betas,
                'year_mod_stats': year_mod_stats,
                'modulation_ratio_mean': float(ratios_arr.mean()),
                'modulation_ratio_std': float(ratios_arr.std()),
            },
            'decomposition': {
                'mean_gamma_contrib': mean_gamma_contrib,
                'mean_beta_contrib': mean_beta_contrib,
                'gamma_pct': gamma_pct,
                'beta_pct': beta_pct,
                'gamma_var_contrib': gamma_var_contrib,
                'beta_var_contrib': beta_var_contrib,
                'policy_var': policy_var,
                'decomposition_verification_error': float(decomp_error),
            },
        }

    # ── Cross-seed summary ──
    print("\n" + "=" * 70)
    print("FiLM Analysis Cross-Seed Summary")
    print("=" * 70)

    # Table 1: Weight L2 norms
    print("\nTable 1: Weight L2 Norms (3-seed mean)")
    gamma_l2_all = np.stack([all_results[s]['static']['gamma_l2'] for s in SEEDS])
    beta_l2_all = np.stack([all_results[s]['static']['beta_l2'] for s in SEEDS])
    combined_l2 = gamma_l2_all.mean(axis=0) + beta_l2_all.mean(axis=0)
    rank_order = np.argsort(-combined_l2)

    summary_weight_l2 = []
    print(f"{'Rank':<5s} {'Feature':<25s} {'Gamma L2 (mean±std)':>22s} {'Beta L2 (mean±std)':>22s}")
    print("-" * 77)
    for rank, j in enumerate(rank_order, 1):
        entry = {
            'rank': rank,
            'feature_idx': int(j),
            'feature_name': FEATURE_NAMES[j],
            'gamma_l2_mean': float(gamma_l2_all[:, j].mean()),
            'gamma_l2_std': float(gamma_l2_all[:, j].std()),
            'beta_l2_mean': float(beta_l2_all[:, j].mean()),
            'beta_l2_std': float(beta_l2_all[:, j].std()),
        }
        summary_weight_l2.append(entry)
        print(f"{rank:<5d} {FEATURE_NAMES[j]:<25s} "
              f"{entry['gamma_l2_mean']:.4f}±{entry['gamma_l2_std']:.4f}          "
              f"{entry['beta_l2_mean']:.4f}±{entry['beta_l2_std']:.4f}")

    # Table 2: Linear contribution decomposition
    print("\nTable 2: Linear Contribution Decomposition (3-seed mean)")
    gamma_contrib_all = np.stack([all_results[s]['decomposition']['mean_gamma_contrib'] for s in SEEDS])
    beta_contrib_all = np.stack([all_results[s]['decomposition']['mean_beta_contrib'] for s in SEEDS])
    gamma_pct_all = np.stack([all_results[s]['decomposition']['gamma_pct'] for s in SEEDS])
    beta_pct_all = np.stack([all_results[s]['decomposition']['beta_pct'] for s in SEEDS])

    summary_decomposition = []
    print(f"{'Feature':<25s} {'γ Abs (mean)':>12s} {'γ % (mean)':>10s} {'β Abs (mean)':>12s} {'β % (mean)':>10s}")
    print("-" * 72)
    for j in range(12):
        entry = {
            'feature_idx': j,
            'feature_name': FEATURE_NAMES[j],
            'gamma_abs_mean': float(gamma_contrib_all[:, j].mean()),
            'gamma_abs_std': float(gamma_contrib_all[:, j].std()),
            'gamma_pct_mean': float(gamma_pct_all[:, j].mean()),
            'beta_abs_mean': float(beta_contrib_all[:, j].mean()),
            'beta_abs_std': float(beta_contrib_all[:, j].std()),
            'beta_pct_mean': float(beta_pct_all[:, j].mean()),
        }
        summary_decomposition.append(entry)
        print(f"{FEATURE_NAMES[j]:<25s} {entry['gamma_abs_mean']:12.4f} {entry['gamma_pct_mean']:9.1%} "
              f"{entry['beta_abs_mean']:12.4f} {entry['beta_pct_mean']:9.1%}")

    # Table 3: Dynamic modulation by year
    print("\nTable 3: Dynamic Modulation Statistics (3-seed mean)")
    all_years = sorted(all_results[SEEDS[0]]['dynamic']['year_mod_stats'].keys())
    summary_dynamic = []
    print(f"{'Year':>6s} {'Gamma mean':>12s} {'Gamma std':>12s} {'Mod. ratio':>12s}")
    print("-" * 45)
    for yr in all_years:
        gm = float(np.mean([all_results[s]['dynamic']['year_mod_stats'].get(yr, {}).get('gamma_mean', 0) for s in SEEDS]))
        gs = float(np.mean([all_results[s]['dynamic']['year_mod_stats'].get(yr, {}).get('gamma_std', 0) for s in SEEDS]))
        mr = float(np.mean([all_results[s]['dynamic']['year_mod_stats'].get(yr, {}).get('mod_ratio_mean', 0) for s in SEEDS]))
        summary_dynamic.append({'year': yr, 'gamma_mean': gm, 'gamma_std': gs, 'mod_ratio_mean': mr})
        print(f"{yr:>6d} {gm:12.4f} {gs:12.4f} {mr:12.4f}")

    # Bias analysis
    print("\nFiLM Bias Analysis (gamma = b_gamma when policy = 0):")
    bias_analysis = {}
    for seed in SEEDS:
        bg = all_results[seed]['static']['b_gamma']
        bb = all_results[seed]['static']['b_beta']
        bias_analysis[seed] = {
            'b_gamma_mean': float(bg.mean()), 'b_gamma_std': float(bg.std()),
            'b_beta_mean': float(bb.mean()), 'b_beta_std': float(bb.std()),
        }
        print(f"  Seed {seed}: b_gamma mean={bg.mean():.4f}, std={bg.std():.4f} | "
              f"b_beta mean={bb.mean():.4f}, std={bb.std():.4f}")

    # Verification errors
    verification_errors = {s: all_results[s]['decomposition']['decomposition_verification_error'] for s in SEEDS}
    print(f"\nDecomposition verification errors: {verification_errors}")

    # Save with full structure
    checkpoint_per_seed = {s: all_results[s]['checkpoint_r2'] for s in SEEDS}
    num_prov_year_per_seed = {s: all_results[s]['num_unique_prov_year'] for s in SEEDS}
    baseline_verification = verify_baselines(recomputed_baselines=checkpoint_per_seed)
    save_data = {
        'metadata': build_audit_metadata(),
        'baseline_r2_per_seed': {s: PKL_BASELINES[s] for s in SEEDS},
        'checkpoint_r2_per_seed': checkpoint_per_seed,
        'baseline_verification': baseline_verification,
        'num_unique_prov_year_per_seed': num_prov_year_per_seed,
        'per_seed': all_results,
        'cross_seed_summary': {
            'weight_l2_ranking': summary_weight_l2,
            'linear_decomposition': summary_decomposition,
            'dynamic_modulation_by_year': summary_dynamic,
            'bias_analysis': bias_analysis,
            'decomposition_verification_errors': verification_errors,
        },
    }
    save_path = output_dir / "film_analysis_results.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {save_path}")

    return save_data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Policy Feature Contribution Analysis")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'group_sensitivity', 'permutation', 'film_analysis'],
                        help='Which analysis to run')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint base directory (default: bytedance-results/checkpoints/)')
    parser.add_argument('--pkl_dir', type=str, default=None,
                        help='PKL results directory for baseline verification '
                             '(default: bytedance-results/results/Multimodal/)')
    parser.add_argument('--n_repeats', type=int, default=30,
                        help='Number of permutation repeats (experiment 2)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: analysis/results/)')
    args = parser.parse_args()

    # Update checkpoint paths if custom dir
    global SEED_CKPT_MAP, SEED_PKL_MAP
    if args.ckpt_dir is not None:
        ckpt_base = args.ckpt_dir
        SEED_CKPT_MAP = {
            42:  f"{ckpt_base}/mm_mae_cnn_film_patch/best_model.pth",
            123: f"{ckpt_base}/mm_mae_cnn_film_patch_seed123/best_model.pth",
            456: f"{ckpt_base}/mm_mae_cnn_film_patch_seed456/best_model.pth",
        }
    if args.pkl_dir is not None:
        pkl_base = args.pkl_dir
        SEED_PKL_MAP = {
            42:  f"{pkl_base}/mm_mae_cnn_film_patch_results.pkl",
            123: f"{pkl_base}/mm_mae_cnn_film_patch_seed123_results.pkl",
            456: f"{pkl_base}/mm_mae_cnn_film_patch_seed456_results.pkl",
        }

    # Device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Verify checkpoints
    for seed, path in SEED_CKPT_MAP.items():
        if not os.path.exists(path):
            print(f"WARNING: Checkpoint not found for seed {seed}: {path}")

    # Load pkl baselines (Ch5 paper口径)
    load_pkl_baselines()
    print(f"PKL baselines loaded: {', '.join(f'seed{s}={PKL_BASELINES[s]:.4f}' for s in SEEDS)}")

    # Verify baselines (no GPU needed, just reads pkl history)
    verify_baselines()

    # Run tasks
    if args.task in ('all', 'group_sensitivity'):
        run_group_sensitivity(device, output_dir)

    if args.task in ('all', 'permutation'):
        run_permutation_importance(device, output_dir, n_repeats=args.n_repeats)

    if args.task in ('all', 'film_analysis'):
        run_film_analysis(device, output_dir)

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
