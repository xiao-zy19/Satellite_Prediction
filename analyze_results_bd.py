#!/usr/bin/env python3
"""
Comprehensive analysis script for Alpha Earth experiment results.

Loads ALL pkl files from the results directory (Baseline, Multimodal,
MultimodalBert, MultimodalHybrid), extracts metrics, groups by configuration,
computes mean +/- std across seeds, and generates summary reports.

Usage:
    python analyze_results_bd.py [--results-dir PATH]

Default results directory: results-bd/ (falls back to results/ if not found)
"""

import os
import sys
import pickle
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = '/home/xiaozhenyu/degree_essay/Alpha_Earth'
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results-bd')
FALLBACK_RESULTS_DIR = os.path.join(BASE_DIR, 'AEF_Data', 'Baseline_Pretrain', 'results')

SUBDIRS = ['Baseline', 'Multimodal', 'MultimodalBert', 'MultimodalHybrid']
METRIC_KEYS = ['r2', 'pearson_r', 'rmse', 'mae']
AGG_STRATEGIES = ['mean', 'median', 'trimmed_mean']


# ============================================================================
# Helper functions
# ============================================================================

def infer_encoder_from_expname(exp_name):
    """Infer encoder type from experiment name for Baseline files."""
    if exp_name is None:
        return 'unknown'
    name = exp_name.lower()
    if 'mlp' in name:
        return 'mlp'
    if 'light_cnn' in name or 'cnn' in name:
        # Check if it's a pretrained CNN variant
        if 'resnet' not in name:
            return 'light_cnn'
    if 'resnet101' in name:
        return 'resnet101'
    if 'resnet50' in name:
        return 'resnet50'
    if 'resnet34' in name:
        return 'resnet34'
    if 'resnet18' in name:
        return 'resnet18'
    if 'resnet10' in name:
        return 'resnet10'
    # Generic "resnet" without number => resnet18 (the default in this codebase)
    if 'resnet' in name:
        return 'resnet18'
    return 'unknown'


def infer_pretrain_from_expname(exp_name, use_pretrain=None, pretrain_method=None):
    """Infer pretrain method from experiment name and explicit fields."""
    # Explicit fields take priority
    if pretrain_method is not None and pretrain_method != 'None':
        return str(pretrain_method)
    if use_pretrain is False or (use_pretrain is None and pretrain_method is None):
        pass  # Fall through to name-based inference

    if exp_name is None:
        return 'none'
    name = exp_name.lower()
    if 'simclr' in name:
        return 'simclr'
    if 'mae' in name:
        return 'mae'
    if 'imagenet' in name:
        return 'imagenet'
    return 'none'


def extract_metrics(metrics_dict):
    """Extract standard metrics from a metrics dictionary."""
    if not isinstance(metrics_dict, dict):
        return {}
    result = {}
    for key in METRIC_KEYS:
        val = metrics_dict.get(key)
        if val is not None:
            result[key] = float(val)
    return result


def extract_best_epoch(data):
    """Extract best epoch from data."""
    return data.get('best_epoch')


def extract_val_best_r2_from_history(data):
    """Extract best validation R2 from training history."""
    history = data.get('history')
    if history and isinstance(history, dict):
        val_r2 = history.get('val_r2')
        if val_r2 and len(val_r2) > 0:
            return float(max(val_r2))
    return None


def extract_total_epochs(data):
    """Extract total number of training epochs."""
    history = data.get('history')
    if history and isinstance(history, dict):
        train_loss = history.get('train_loss')
        if train_loss:
            return len(train_loss)
    return None


def normalize_year_list(years):
    """Normalize year-like values to a sorted list of ints."""
    if years is None:
        return None
    if isinstance(years, (list, tuple, set, np.ndarray, pd.Series)):
        values = [int(y) for y in years]
    else:
        values = [int(years)]
    return sorted(values)


def format_years_compact(years):
    """Format sorted years into compact ranges like 2018-2021,2023."""
    if not years:
        return 'NA'

    ranges = []
    start = years[0]
    prev = years[0]

    for year in years[1:]:
        if year == prev + 1:
            prev = year
            continue
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f'{start}-{prev}')
        start = year
        prev = year

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f'{start}-{prev}')

    return ','.join(ranges)


def extract_split_info(data):
    """Extract split metadata in a backward-compatible way."""
    temporal_split = bool(data.get('temporal_split', False))
    train_years = normalize_year_list(data.get('train_years'))
    val_years = normalize_year_list(data.get('val_years'))
    test_years = normalize_year_list(data.get('test_years'))

    if temporal_split:
        train_str = format_years_compact(train_years)
        val_str = format_years_compact(val_years)
        test_str = format_years_compact(test_years)
        if train_years is None and val_years is None and test_years is None:
            split_tag = 'temporal_unspecified'
        else:
            split_tag = f'T:{train_str}|V:{val_str}|E:{test_str}'
    else:
        train_str = ''
        val_str = ''
        test_str = ''
        split_tag = 'random'

    return {
        'temporal_split': temporal_split,
        'train_years': train_years,
        'val_years': val_years,
        'test_years': test_years,
        'train_years_str': train_str,
        'val_years_str': val_str,
        'test_years_str': test_str,
        'split_tag': split_tag,
    }


def parse_baseline_file(data, filename, subdir):
    """Parse a Baseline pkl file and return list of result dicts."""
    results = []

    exp_name = data.get('exp_name', '')
    encoder = infer_encoder_from_expname(exp_name)
    pretrain = infer_pretrain_from_expname(
        exp_name,
        data.get('use_pretrain'),
        data.get('pretrain_method')
    )
    training_mode = data.get('training_mode', 'unknown')
    seed = data.get('seed')
    if seed is None:
        seed = 42  # Default seed for older experiments
    best_epoch = extract_best_epoch(data)
    total_epochs = extract_total_epochs(data)
    model_params = data.get('model_params')
    val_best_r2 = extract_val_best_r2_from_history(data)
    split_info = extract_split_info(data)

    # Extract validation metrics (flat)
    val_metrics_raw = {}
    if 'val_metrics' in data:
        val_metrics_raw = extract_metrics(data['val_metrics'])

    if training_mode == 'patch_level' and 'test_results' in data:
        # Newer patch-level format with mean/median/trimmed_mean
        test_results = data['test_results']
        val_results = data.get('val_results', {})

        for agg in AGG_STRATEGIES:
            if agg in test_results and isinstance(test_results[agg], dict):
                test_m = extract_metrics(test_results[agg].get('metrics', {}))
                val_m = {}
                if agg in val_results and isinstance(val_results[agg], dict):
                    val_m = extract_metrics(val_results[agg].get('metrics', {}))

                row = {
                    'subdir': subdir,
                    'filename': filename,
                    'exp_name': exp_name,
                    'encoder': encoder,
                    'pretrain': pretrain,
                    'training_mode': 'patch_level',
                    'aggregation': agg,
                    'fusion_type': 'N/A',
                    'policy_source': 'N/A',
                    'policy_dim': 'N/A',
                    'seed': seed,
                    'best_epoch': best_epoch,
                    'total_epochs': total_epochs,
                    'model_params': model_params,
                    **split_info,
                }
                for mk in METRIC_KEYS:
                    row[f'test_{mk}'] = test_m.get(mk)
                    row[f'val_{mk}'] = val_m.get(mk)
                row['val_best_r2_history'] = val_best_r2
                results.append(row)
    else:
        # City-level or older patch-level with flat test_metrics
        test_m = extract_metrics(data.get('test_metrics', {}))
        model_agg = data.get('model_aggregation', 'mean')
        if model_agg is None:
            model_agg = 'mean'

        agg_label = model_agg if training_mode == 'city_level' else 'mean'
        # For patch-level old format: this was the only aggregation available
        if training_mode == 'patch_level':
            agg_label = 'mean'  # Old patch-level used mean by default

        row = {
            'subdir': subdir,
            'filename': filename,
            'exp_name': exp_name,
            'encoder': encoder,
            'pretrain': pretrain,
            'training_mode': training_mode,
            'aggregation': agg_label,
            'fusion_type': 'N/A',
            'policy_source': 'N/A',
            'policy_dim': 'N/A',
            'seed': seed,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'model_params': model_params,
            **split_info,
        }
        for mk in METRIC_KEYS:
            row[f'test_{mk}'] = test_m.get(mk)
            row[f'val_{mk}'] = val_metrics_raw.get(mk)
        row['val_best_r2_history'] = val_best_r2
        results.append(row)

    return results


def parse_multimodal_file(data, filename, subdir):
    """Parse a Multimodal/MultimodalBert/MultimodalHybrid pkl file."""
    results = []

    exp_name = data.get('exp_name', '')
    encoder_raw = data.get('image_encoder', 'unknown')
    # Normalize encoder name
    if encoder_raw == 'light_cnn':
        encoder = 'light_cnn'
    elif encoder_raw == 'mlp':
        encoder = 'mlp'
    elif encoder_raw == 'resnet':
        encoder = 'resnet18'  # Default resnet is resnet18
    else:
        encoder = encoder_raw

    # Infer pretrain from exp_name (multimodal files often lack use_pretrain field)
    pretrain = infer_pretrain_from_expname(
        exp_name,
        data.get('use_pretrain'),
        data.get('pretrain_method')
    )

    training_mode = data.get('training_mode', 'unknown')
    seed = data.get('seed')
    if seed is None:
        seed = 42
    best_epoch = extract_best_epoch(data)
    total_epochs = extract_total_epochs(data)
    model_params = data.get('model_params')
    fusion_type = data.get('fusion_type', 'unknown')
    policy_source = data.get('policy_source', 'N/A')
    policy_dim = data.get('policy_dim', 'N/A')
    val_best_r2 = extract_val_best_r2_from_history(data)
    split_info = extract_split_info(data)

    # Determine policy_source from subdir if not set
    if policy_source is None or policy_source == 'N/A' or policy_source == '?':
        if subdir == 'Multimodal':
            policy_source = 'structured'
        elif subdir == 'MultimodalBert':
            policy_source = 'bert'
        elif subdir == 'MultimodalHybrid':
            policy_source = 'hybrid'

    # Extract validation metrics (flat)
    val_metrics_raw = {}
    if 'val_metrics' in data:
        val_metrics_raw = extract_metrics(data['val_metrics'])

    if training_mode == 'patch_level' and 'test_results' in data:
        # Patch-level with mean/median/trimmed_mean
        test_results = data['test_results']
        val_results = data.get('val_results', {})

        for agg in AGG_STRATEGIES:
            if agg in test_results and isinstance(test_results[agg], dict):
                test_m = extract_metrics(test_results[agg].get('metrics', {}))
                val_m = {}
                if isinstance(val_results, dict) and agg in val_results and isinstance(val_results[agg], dict):
                    val_m = extract_metrics(val_results[agg].get('metrics', {}))

                row = {
                    'subdir': subdir,
                    'filename': filename,
                    'exp_name': exp_name,
                    'encoder': encoder,
                    'pretrain': pretrain,
                    'training_mode': 'patch_level',
                    'aggregation': agg,
                    'fusion_type': fusion_type,
                    'policy_source': policy_source,
                    'policy_dim': policy_dim,
                    'seed': seed,
                    'best_epoch': best_epoch,
                    'total_epochs': total_epochs,
                    'model_params': model_params,
                    **split_info,
                }
                for mk in METRIC_KEYS:
                    row[f'test_{mk}'] = test_m.get(mk)
                    row[f'val_{mk}'] = val_m.get(mk)
                row['val_best_r2_history'] = val_best_r2
                results.append(row)
    else:
        # City-level with flat test_metrics
        test_m = extract_metrics(data.get('test_metrics', {}))
        model_agg = data.get('model_aggregation', 'mean')
        if model_agg is None:
            model_agg = 'mean'

        row = {
            'subdir': subdir,
            'filename': filename,
            'exp_name': exp_name,
            'encoder': encoder,
            'pretrain': pretrain,
            'training_mode': training_mode,
            'aggregation': model_agg,
            'fusion_type': fusion_type,
            'policy_source': policy_source,
            'policy_dim': policy_dim,
            'seed': seed,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs,
            'model_params': model_params,
            **split_info,
        }
        for mk in METRIC_KEYS:
            row[f'test_{mk}'] = test_m.get(mk)
            row[f'val_{mk}'] = val_metrics_raw.get(mk)
        row['val_best_r2_history'] = val_best_r2
        results.append(row)

    return results


# ============================================================================
# Main analysis
# ============================================================================

def load_all_results(results_dir):
    """Load all pkl files and return a list of parsed result dicts."""
    all_results = []
    warnings_list = []
    file_counts = {}

    for subdir in SUBDIRS:
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            warnings_list.append(f"WARNING: Subdirectory not found: {subdir_path}")
            file_counts[subdir] = 0
            continue

        pkl_files = [f for f in os.listdir(subdir_path) if f.endswith('.pkl')]
        file_counts[subdir] = len(pkl_files)

        for filename in sorted(pkl_files):
            filepath = os.path.join(subdir_path, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                if not isinstance(data, dict):
                    warnings_list.append(f"WARNING: {subdir}/{filename} - not a dict (type={type(data).__name__})")
                    continue

                if subdir == 'Baseline':
                    rows = parse_baseline_file(data, filename, subdir)
                else:
                    rows = parse_multimodal_file(data, filename, subdir)

                if not rows:
                    warnings_list.append(f"WARNING: {subdir}/{filename} - no results extracted")
                else:
                    all_results.extend(rows)

            except Exception as e:
                warnings_list.append(f"ERROR: {subdir}/{filename} - {str(e)}")

    return all_results, warnings_list, file_counts


def compute_grouped_summary(df, group_cols, sort_by='test_r2_mean'):
    """Group by configuration and compute mean +/- std across seeds."""
    metric_cols = [c for c in df.columns if c.startswith('test_') or c.startswith('val_')]
    extra_cols = ['best_epoch', 'total_epochs', 'model_params']

    agg_dict = {}
    for mc in metric_cols:
        agg_dict[mc] = ['mean', 'std', 'count']
    for ec in extra_cols:
        if ec in df.columns:
            agg_dict[ec] = ['mean']
    agg_dict['seed'] = ['count', lambda x: ','.join(sorted([str(s) for s in x]))]

    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)

    # Flatten column names
    flat_cols = []
    flat_data = {}
    for col in grouped.columns:
        if col[1] == '<lambda_0>' or col[1] == '<lambda>':
            flat_name = f'{col[0]}_seeds'
        elif col[0] == 'seed' and col[1] == 'count':
            flat_name = 'n_seeds'
        else:
            flat_name = f'{col[0]}_{col[1]}'
        flat_cols.append(flat_name)

    grouped.columns = flat_cols

    # Remove duplicate count columns
    count_cols = [c for c in grouped.columns if c.endswith('_count') and c != 'n_seeds']
    if count_cols:
        grouped = grouped.drop(columns=count_cols[1:])  # Keep first count
        grouped = grouped.rename(columns={count_cols[0]: 'n_runs'})

    # Sort by mean test R2 descending
    if sort_by in grouped.columns:
        grouped = grouped.sort_values(sort_by, ascending=False)

    return grouped.reset_index()


def add_split_group(base_cols):
    """Prepend split metadata to grouping columns."""
    return ['split_tag'] + list(base_cols)


def format_metric(mean_val, std_val, fmt='.3f'):
    """Format metric as mean +/- std."""
    if pd.isna(mean_val):
        return 'N/A'
    mean_str = f'{mean_val:{fmt}}'
    if pd.isna(std_val) or std_val == 0:
        return mean_str
    std_str = f'{std_val:{fmt}}'
    return f'{mean_str} +/- {std_str}'


def generate_report_section(title, df_grouped, group_cols, report_lines):
    """Generate a formatted text section for the report."""
    report_lines.append('=' * 80)
    report_lines.append(title)
    report_lines.append('=' * 80)
    report_lines.append('')

    if df_grouped.empty:
        report_lines.append('  No results found.')
        report_lines.append('')
        return

    report_lines.append(f'  Total configurations: {len(df_grouped)}')
    report_lines.append('')

    # Header
    header_parts = [f'{"#":>3}']
    for gc in group_cols:
        header_parts.append(f'{gc:>15}')
    header_parts.extend([
        f'{"n_runs":>6}',
        f'{"Test R2":>20}',
        f'{"Test r":>20}',
        f'{"Test RMSE":>20}',
        f'{"Test MAE":>20}',
        f'{"Best Epoch":>10}',
    ])
    header = ' | '.join(header_parts)
    report_lines.append(header)
    report_lines.append('-' * len(header))

    for idx, row in df_grouped.iterrows():
        parts = [f'{idx+1:>3}']
        for gc in group_cols:
            val = row.get(gc, 'N/A')
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = 'N/A'
            parts.append(f'{str(val):>15}')

        n_runs = row.get('n_runs', row.get('n_seeds', '?'))
        parts.append(f'{n_runs:>6}')

        r2_str = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
        r_str = format_metric(row.get('test_pearson_r_mean'), row.get('test_pearson_r_std'))
        rmse_str = format_metric(row.get('test_rmse_mean'), row.get('test_rmse_std'))
        mae_str = format_metric(row.get('test_mae_mean'), row.get('test_mae_std'))
        epoch_val = row.get('best_epoch_mean')
        epoch_str = f'{epoch_val:.0f}' if not pd.isna(epoch_val) else 'N/A'

        parts.extend([
            f'{r2_str:>20}',
            f'{r_str:>20}',
            f'{rmse_str:>20}',
            f'{mae_str:>20}',
            f'{epoch_str:>10}',
        ])
        report_lines.append(' | '.join(parts))

    report_lines.append('')
    report_lines.append('')


def main():
    parser = argparse.ArgumentParser(description='Analyze Alpha Earth experiment results')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to results directory')
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    elif os.path.isdir(DEFAULT_RESULTS_DIR):
        results_dir = DEFAULT_RESULTS_DIR
    elif os.path.isdir(FALLBACK_RESULTS_DIR):
        results_dir = FALLBACK_RESULTS_DIR
        print(f"NOTE: results-bd/ not found, using fallback: {FALLBACK_RESULTS_DIR}")
    else:
        print(f"ERROR: No results directory found at {DEFAULT_RESULTS_DIR} or {FALLBACK_RESULTS_DIR}")
        sys.exit(1)

    print(f"Results directory: {results_dir}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================================================
    # Step 1: Load all results
    # ========================================================================
    print("Loading all pkl files...")
    all_results, warnings_list, file_counts = load_all_results(results_dir)

    print(f"\nFile counts per subdirectory:")
    for subdir, count in file_counts.items():
        print(f"  {subdir}: {count} pkl files")
    print(f"  TOTAL pkl files: {sum(file_counts.values())}")
    print(f"  Total result rows extracted: {len(all_results)}")
    print(f"  (Patch-level files produce 3 rows each: mean/median/trimmed_mean)")

    if warnings_list:
        print(f"\nWarnings/Errors ({len(warnings_list)}):")
        for w in warnings_list:
            print(f"  {w}")
    print()

    if not all_results:
        print("ERROR: No results extracted. Exiting.")
        sys.exit(1)

    # ========================================================================
    # Step 2: Create DataFrame
    # ========================================================================
    df = pd.DataFrame(all_results)

    # Fill None policy_source for Multimodal files that didn't have it
    df['policy_source'] = df['policy_source'].fillna('structured')
    df['temporal_split'] = df['temporal_split'].fillna(False).astype(bool)
    df['split_tag'] = df['split_tag'].fillna('random')

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()

    # ========================================================================
    # Step 3: Save detailed per-experiment CSV
    # ========================================================================
    detailed_csv_path = os.path.join(results_dir, 'analysis_detailed.csv')
    df_sorted = df.sort_values(['subdir', 'split_tag', 'encoder', 'pretrain', 'training_mode',
                                 'fusion_type', 'aggregation', 'seed'])
    df_sorted.to_csv(detailed_csv_path, index=False, float_format='%.6f')
    print(f"Detailed per-experiment CSV saved: {detailed_csv_path}")
    print(f"  Rows: {len(df_sorted)}")

    # ========================================================================
    # Step 4: Compute grouped summaries
    # ========================================================================
    report_lines = []
    report_lines.append('=' * 80)
    report_lines.append('ALPHA EARTH EXPERIMENT RESULTS - COMPREHENSIVE ANALYSIS REPORT')
    report_lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_lines.append(f'Results directory: {results_dir}')
    report_lines.append('=' * 80)
    report_lines.append('')

    # File count summary
    report_lines.append('FILE COUNTS:')
    for subdir, count in file_counts.items():
        report_lines.append(f'  {subdir}: {count} pkl files')
    report_lines.append(f'  TOTAL: {sum(file_counts.values())} pkl files')
    report_lines.append(f'  Total result rows: {len(df)} (patch-level => 3 rows each)')
    report_lines.append('')

    all_summary_dfs = []

    # ---- 4a: Baseline experiments ----
    df_baseline = df[df['subdir'] == 'Baseline'].copy()
    if not df_baseline.empty:
        group_cols_bl = add_split_group(['encoder', 'pretrain', 'training_mode', 'aggregation'])
        summary_bl = compute_grouped_summary(df_baseline, group_cols_bl)
        summary_bl.insert(0, 'category', 'Baseline')
        all_summary_dfs.append(summary_bl)

        print(f"\n{'='*80}")
        print("BASELINE EXPERIMENTS SUMMARY")
        print(f"{'='*80}")
        print(f"Configurations: {len(summary_bl)}")
        print(f"Total runs: {len(df_baseline)}")
        print()

        # Print top results
        top_n = min(10, len(summary_bl))
        print(f"Top {top_n} by Test R2:")
        for i, row in summary_bl.head(top_n).iterrows():
            r2 = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
            split = row.get('split_tag', '?')
            enc = row.get('encoder', '?')
            pre = row.get('pretrain', '?')
            mode = row.get('training_mode', '?')
            agg = row.get('aggregation', '?')
            n = row.get('n_runs', '?')
            print(f"  {split:>24} | {enc:>12} | {pre:>8} | {mode:>12} | {agg:>13} | n={n} | R2={r2}")

        generate_report_section(
            'BASELINE EXPERIMENTS (sorted by Test R2)',
            summary_bl, group_cols_bl, report_lines
        )

    # ---- 4b: Multimodal experiments (structured policy) ----
    df_mm = df[df['subdir'] == 'Multimodal'].copy()
    if not df_mm.empty:
        group_cols_mm = add_split_group(['encoder', 'pretrain', 'training_mode', 'aggregation', 'fusion_type'])
        summary_mm = compute_grouped_summary(df_mm, group_cols_mm)
        summary_mm.insert(0, 'category', 'Multimodal')
        all_summary_dfs.append(summary_mm)

        print(f"\n{'='*80}")
        print("MULTIMODAL EXPERIMENTS SUMMARY (Structured Policy)")
        print(f"{'='*80}")
        print(f"Configurations: {len(summary_mm)}")
        print(f"Total runs: {len(df_mm)}")
        print()

        top_n = min(15, len(summary_mm))
        print(f"Top {top_n} by Test R2:")
        for i, row in summary_mm.head(top_n).iterrows():
            r2 = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
            split = row.get('split_tag', '?')
            enc = row.get('encoder', '?')
            pre = row.get('pretrain', '?')
            mode = row.get('training_mode', '?')
            agg = row.get('aggregation', '?')
            fus = row.get('fusion_type', '?')
            n = row.get('n_runs', '?')
            print(f"  {split:>24} | {enc:>12} | {pre:>8} | {mode:>12} | {agg:>15} | {fus:>10} | n={n} | R2={r2}")

        generate_report_section(
            'MULTIMODAL EXPERIMENTS - Structured Policy (sorted by Test R2)',
            summary_mm, group_cols_mm, report_lines
        )

    # ---- 4c: MultimodalBert experiments ----
    df_bert = df[df['subdir'] == 'MultimodalBert'].copy()
    if not df_bert.empty:
        group_cols_bert = add_split_group(['encoder', 'pretrain', 'training_mode', 'aggregation', 'fusion_type'])
        summary_bert = compute_grouped_summary(df_bert, group_cols_bert)
        summary_bert.insert(0, 'category', 'MultimodalBert')
        all_summary_dfs.append(summary_bert)

        print(f"\n{'='*80}")
        print("MULTIMODAL BERT EXPERIMENTS SUMMARY")
        print(f"{'='*80}")
        print(f"Configurations: {len(summary_bert)}")
        print(f"Total runs: {len(df_bert)}")
        print()

        top_n = min(10, len(summary_bert))
        print(f"Top {top_n} by Test R2:")
        for i, row in summary_bert.head(top_n).iterrows():
            r2 = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
            split = row.get('split_tag', '?')
            enc = row.get('encoder', '?')
            pre = row.get('pretrain', '?')
            mode = row.get('training_mode', '?')
            agg = row.get('aggregation', '?')
            fus = row.get('fusion_type', '?')
            n = row.get('n_runs', '?')
            print(f"  {split:>24} | {enc:>12} | {pre:>8} | {mode:>12} | {agg:>13} | {fus:>10} | n={n} | R2={r2}")

        generate_report_section(
            'MULTIMODAL BERT EXPERIMENTS (sorted by Test R2)',
            summary_bert, group_cols_bert, report_lines
        )

    # ---- 4d: MultimodalHybrid experiments ----
    df_hybrid = df[df['subdir'] == 'MultimodalHybrid'].copy()
    if not df_hybrid.empty:
        group_cols_hyb = add_split_group(['encoder', 'pretrain', 'training_mode', 'aggregation', 'fusion_type'])
        summary_hyb = compute_grouped_summary(df_hybrid, group_cols_hyb)
        summary_hyb.insert(0, 'category', 'MultimodalHybrid')
        all_summary_dfs.append(summary_hyb)

        print(f"\n{'='*80}")
        print("MULTIMODAL HYBRID EXPERIMENTS SUMMARY")
        print(f"{'='*80}")
        print(f"Configurations: {len(summary_hyb)}")
        print(f"Total runs: {len(df_hybrid)}")
        print()

        top_n = min(10, len(summary_hyb))
        print(f"Top {top_n} by Test R2:")
        for i, row in summary_hyb.head(top_n).iterrows():
            r2 = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
            split = row.get('split_tag', '?')
            enc = row.get('encoder', '?')
            pre = row.get('pretrain', '?')
            mode = row.get('training_mode', '?')
            agg = row.get('aggregation', '?')
            fus = row.get('fusion_type', '?')
            n = row.get('n_runs', '?')
            print(f"  {split:>24} | {enc:>12} | {pre:>8} | {mode:>12} | {agg:>13} | {fus:>10} | n={n} | R2={r2}")

        generate_report_section(
            'MULTIMODAL HYBRID EXPERIMENTS (sorted by Test R2)',
            summary_hyb, group_cols_hyb, report_lines
        )

    # ========================================================================
    # Step 5: Cross-category best results
    # ========================================================================
    print(f"\n{'='*80}")
    print("OVERALL TOP 20 CONFIGURATIONS (all categories)")
    print(f"{'='*80}")

    if all_summary_dfs:
        # Combine all summaries
        df_all_summary = pd.concat(all_summary_dfs, ignore_index=True)
        df_all_summary = df_all_summary.sort_values('test_r2_mean', ascending=False)

        top_n = min(20, len(df_all_summary))
        for i, (_, row) in enumerate(df_all_summary.head(top_n).iterrows()):
            r2 = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
            cat = row.get('category', '?')
            split = row.get('split_tag', '?')
            enc = row.get('encoder', '?')
            pre = row.get('pretrain', '?')
            mode = row.get('training_mode', '?')
            agg = row.get('aggregation', '?')
            fus = row.get('fusion_type', 'N/A')
            n = row.get('n_runs', '?')
            print(f"  {i+1:>2}. [{cat:>15}] {split:>24} | {enc:>12} | {pre:>8} | {mode:>12} | "
                  f"{agg:>15} | {fus:>10} | n={n} | R2={r2}")

        # Add to report
        report_lines.append('=' * 80)
        report_lines.append('OVERALL TOP 20 CONFIGURATIONS (all categories)')
        report_lines.append('=' * 80)
        report_lines.append('')
        for i, (_, row) in enumerate(df_all_summary.head(20).iterrows()):
            r2 = format_metric(row.get('test_r2_mean'), row.get('test_r2_std'))
            r = format_metric(row.get('test_pearson_r_mean'), row.get('test_pearson_r_std'))
            rmse = format_metric(row.get('test_rmse_mean'), row.get('test_rmse_std'))
            mae_v = format_metric(row.get('test_mae_mean'), row.get('test_mae_std'))
            cat = row.get('category', '?')
            split = row.get('split_tag', '?')
            enc = row.get('encoder', '?')
            pre = row.get('pretrain', '?')
            mode = row.get('training_mode', '?')
            agg = row.get('aggregation', '?')
            fus = row.get('fusion_type', 'N/A')
            n = row.get('n_runs', '?')
            report_lines.append(
                f"  {i+1:>2}. [{cat:>15}] {split} | {enc} | {pre} | {mode} | {agg} | {fus} | n={n}"
            )
            report_lines.append(
                f"      R2={r2}  |  r={r}  |  RMSE={rmse}  |  MAE={mae_v}"
            )
        report_lines.append('')

    # ========================================================================
    # Step 6: Additional analysis - training mode comparison
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING MODE COMPARISON")
    print(f"{'='*80}")

    for subdir_name in ['Baseline', 'Multimodal', 'MultimodalBert', 'MultimodalHybrid']:
        df_sub = df[df['subdir'] == subdir_name]
        if df_sub.empty:
            continue
        for split_tag in sorted(df_sub['split_tag'].unique()):
            df_split = df_sub[df_sub['split_tag'] == split_tag]
            for mode in ['city_level', 'patch_level']:
                df_mode = df_split[df_split['training_mode'] == mode]
                if not df_mode.empty:
                    mean_r2 = df_mode['test_r2'].mean()
                    max_r2 = df_mode['test_r2'].max()
                    n = len(df_mode)
                    print(f"  {subdir_name:>18} | {split_tag:>24} | {mode:>12} | n={n:>4} | "
                          f"mean_R2={mean_r2:.3f} | max_R2={max_r2:.3f}")

    report_lines.append('=' * 80)
    report_lines.append('TRAINING MODE COMPARISON')
    report_lines.append('=' * 80)
    report_lines.append('')
    for subdir_name in ['Baseline', 'Multimodal', 'MultimodalBert', 'MultimodalHybrid']:
        df_sub = df[df['subdir'] == subdir_name]
        if df_sub.empty:
            continue
        for split_tag in sorted(df_sub['split_tag'].unique()):
            df_split = df_sub[df_sub['split_tag'] == split_tag]
            for mode in ['city_level', 'patch_level']:
                df_mode = df_split[df_split['training_mode'] == mode]
                if not df_mode.empty:
                    mean_r2 = df_mode['test_r2'].mean()
                    std_r2 = df_mode['test_r2'].std()
                    max_r2 = df_mode['test_r2'].max()
                    min_r2 = df_mode['test_r2'].min()
                    n = len(df_mode)
                    report_lines.append(
                        f"  {subdir_name:>18} | {split_tag:>24} | {mode:>12} | n={n:>4} | "
                        f"mean={mean_r2:.3f} +/- {std_r2:.3f} | "
                        f"max={max_r2:.3f} | min={min_r2:.3f}"
                    )
    report_lines.append('')

    # ========================================================================
    # Step 7: Fusion type comparison (multimodal only)
    # ========================================================================
    df_mm_all = df[df['subdir'].isin(['Multimodal', 'MultimodalBert', 'MultimodalHybrid'])]
    if not df_mm_all.empty:
        print(f"\n{'='*80}")
        print("FUSION TYPE COMPARISON (across all multimodal)")
        print(f"{'='*80}")

        fusion_summary = df_mm_all.groupby(['split_tag', 'fusion_type']).agg(
            n=('test_r2', 'count'),
            r2_mean=('test_r2', 'mean'),
            r2_std=('test_r2', 'std'),
            r2_max=('test_r2', 'max'),
        ).sort_values('r2_mean', ascending=False)

        for (split_tag, fus), row in fusion_summary.iterrows():
            print(f"  {split_tag:>24} | {fus:>12} | n={row['n']:>4} | "
                  f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
                  f"max_R2={row['r2_max']:.3f}")

        report_lines.append('=' * 80)
        report_lines.append('FUSION TYPE COMPARISON (across all multimodal)')
        report_lines.append('=' * 80)
        report_lines.append('')
        for (split_tag, fus), row in fusion_summary.iterrows():
            report_lines.append(
                f"  {split_tag:>24} | {fus:>12} | n={row['n']:>4} | "
                f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
                f"max_R2={row['r2_max']:.3f}"
            )
        report_lines.append('')

    # ========================================================================
    # Step 8: Encoder comparison (across all experiments)
    # ========================================================================
    print(f"\n{'='*80}")
    print("ENCODER COMPARISON (across all experiments)")
    print(f"{'='*80}")

    encoder_summary = df.groupby(['split_tag', 'encoder']).agg(
        n=('test_r2', 'count'),
        r2_mean=('test_r2', 'mean'),
        r2_std=('test_r2', 'std'),
        r2_max=('test_r2', 'max'),
    ).sort_values('r2_mean', ascending=False)

    for (split_tag, enc), row in encoder_summary.iterrows():
        print(f"  {split_tag:>24} | {enc:>12} | n={row['n']:>4} | "
              f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
              f"max_R2={row['r2_max']:.3f}")

    report_lines.append('=' * 80)
    report_lines.append('ENCODER COMPARISON (across all experiments)')
    report_lines.append('=' * 80)
    report_lines.append('')
    for (split_tag, enc), row in encoder_summary.iterrows():
        report_lines.append(
            f"  {split_tag:>24} | {enc:>12} | n={row['n']:>4} | "
            f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
            f"max_R2={row['r2_max']:.3f}"
        )
    report_lines.append('')

    # ========================================================================
    # Step 9: Pretrain method comparison
    # ========================================================================
    print(f"\n{'='*80}")
    print("PRETRAIN METHOD COMPARISON (across all experiments)")
    print(f"{'='*80}")

    pretrain_summary = df.groupby(['split_tag', 'pretrain']).agg(
        n=('test_r2', 'count'),
        r2_mean=('test_r2', 'mean'),
        r2_std=('test_r2', 'std'),
        r2_max=('test_r2', 'max'),
    ).sort_values('r2_mean', ascending=False)

    for (split_tag, pre), row in pretrain_summary.iterrows():
        print(f"  {split_tag:>24} | {pre:>12} | n={row['n']:>4} | "
              f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
              f"max_R2={row['r2_max']:.3f}")

    report_lines.append('=' * 80)
    report_lines.append('PRETRAIN METHOD COMPARISON (across all experiments)')
    report_lines.append('=' * 80)
    report_lines.append('')
    for (split_tag, pre), row in pretrain_summary.iterrows():
        report_lines.append(
            f"  {split_tag:>24} | {pre:>12} | n={row['n']:>4} | "
            f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
            f"max_R2={row['r2_max']:.3f}"
        )
    report_lines.append('')

    # ========================================================================
    # Step 10: Aggregation strategy comparison for patch-level
    # ========================================================================
    df_patch = df[df['training_mode'] == 'patch_level']
    if not df_patch.empty:
        print(f"\n{'='*80}")
        print("AGGREGATION STRATEGY COMPARISON (patch-level only)")
        print(f"{'='*80}")

        agg_summary = df_patch.groupby(['split_tag', 'aggregation']).agg(
            n=('test_r2', 'count'),
            r2_mean=('test_r2', 'mean'),
            r2_std=('test_r2', 'std'),
            r2_max=('test_r2', 'max'),
        ).sort_values('r2_mean', ascending=False)

        for (split_tag, agg), row in agg_summary.iterrows():
            print(f"  {split_tag:>24} | {agg:>15} | n={row['n']:>4} | "
                  f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
                  f"max_R2={row['r2_max']:.3f}")

        report_lines.append('=' * 80)
        report_lines.append('AGGREGATION STRATEGY COMPARISON (patch-level only)')
        report_lines.append('=' * 80)
        report_lines.append('')
        for (split_tag, agg), row in agg_summary.iterrows():
            report_lines.append(
                f"  {split_tag:>24} | {agg:>15} | n={row['n']:>4} | "
                f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
                f"max_R2={row['r2_max']:.3f}"
            )
        report_lines.append('')

    # ========================================================================
    # Step 11: Policy source comparison
    # ========================================================================
    df_policy = df[df['subdir'].isin(['Multimodal', 'MultimodalBert', 'MultimodalHybrid'])]
    if not df_policy.empty:
        print(f"\n{'='*80}")
        print("POLICY SOURCE COMPARISON")
        print(f"{'='*80}")

        ps_summary = df_policy.groupby(['split_tag', 'policy_source']).agg(
            n=('test_r2', 'count'),
            r2_mean=('test_r2', 'mean'),
            r2_std=('test_r2', 'std'),
            r2_max=('test_r2', 'max'),
        ).sort_values('r2_mean', ascending=False)

        for (split_tag, ps), row in ps_summary.iterrows():
            print(f"  {split_tag:>24} | {ps:>12} | n={row['n']:>4} | "
                  f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
                  f"max_R2={row['r2_max']:.3f}")

        report_lines.append('=' * 80)
        report_lines.append('POLICY SOURCE COMPARISON')
        report_lines.append('=' * 80)
        report_lines.append('')
        for (split_tag, ps), row in ps_summary.iterrows():
            report_lines.append(
                f"  {split_tag:>24} | {ps:>12} | n={row['n']:>4} | "
                f"mean_R2={row['r2_mean']:.3f} +/- {row['r2_std']:.3f} | "
                f"max_R2={row['r2_max']:.3f}"
            )
        report_lines.append('')

    # ========================================================================
    # Step 12: Save grouped summary CSV
    # ========================================================================
    if all_summary_dfs:
        df_all_summary = pd.concat(all_summary_dfs, ignore_index=True)
        df_all_summary = df_all_summary.sort_values('test_r2_mean', ascending=False)

        summary_csv_path = os.path.join(results_dir, 'analysis_summary.csv')
        df_all_summary.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"\nGrouped summary CSV saved: {summary_csv_path}")
        print(f"  Rows: {len(df_all_summary)}")

    # ========================================================================
    # Step 13: Save report
    # ========================================================================
    report_path = os.path.join(results_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Analysis report saved: {report_path}")

    # ========================================================================
    # Step 14: Summary statistics
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total pkl files processed: {sum(file_counts.values())}")
    print(f"Total result rows: {len(df)}")
    print(f"Unique encoders: {sorted(df['encoder'].unique())}")
    print(f"Unique pretrain methods: {sorted(df['pretrain'].unique())}")
    print(f"Unique training modes: {sorted(df['training_mode'].unique())}")
    print(f"Unique fusion types: {sorted(df[df['fusion_type'] != 'N/A']['fusion_type'].unique())}")
    print(f"Unique aggregation methods: {sorted(df['aggregation'].unique())}")
    print(f"Unique policy sources: {sorted(df[df['policy_source'] != 'N/A']['policy_source'].unique())}")
    print(f"Unique split tags: {sorted(df['split_tag'].unique())}")
    print(f"Seed values: {sorted(df['seed'].unique())}")
    print()

    best_row = df.loc[df['test_r2'].idxmax()]
    print(f"BEST SINGLE RESULT:")
    print(f"  File: {best_row['subdir']}/{best_row['filename']}")
    print(f"  Config: encoder={best_row['encoder']}, pretrain={best_row['pretrain']}, "
          f"mode={best_row['training_mode']}, agg={best_row['aggregation']}, "
          f"fusion={best_row['fusion_type']}, split={best_row['split_tag']}")
    print(f"  Test R2={best_row['test_r2']:.4f}, "
          f"r={best_row['test_pearson_r']:.4f}, "
          f"RMSE={best_row['test_rmse']:.4f}, "
          f"MAE={best_row['test_mae']:.4f}")
    print(f"  Seed: {best_row['seed']}, Best Epoch: {best_row['best_epoch']}")

    print(f"\nAnalysis complete!")
    print(f"Output files:")
    print(f"  1. {detailed_csv_path}")
    print(f"  2. {summary_csv_path if all_summary_dfs else 'N/A'}")
    print(f"  3. {report_path}")


if __name__ == '__main__':
    main()
