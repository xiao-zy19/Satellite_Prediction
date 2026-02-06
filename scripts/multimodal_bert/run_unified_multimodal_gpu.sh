#!/bin/bash
# ============================================================================
# Unified Multimodal Experiment Runner (GPU Normalization)
#
# Supports ALL policy sources with COMPLETE experiment coverage:
#   - structured: Original 12-dim policy features (mm_* experiments)
#   - bert: BERT embeddings 64-dim (bert_* experiments)
#   - hybrid: BERT + Structured 76-dim (hybrid_* experiments)
#
# Each policy source has 56 identical experiment configurations.
# Total: 168 experiments
#
# This script uses train_multimodal_bert.py which handles all policy sources.
# ============================================================================

set -e

# Project configuration
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
# Logs and results are saved to dedicated subfolders based on policy source:
#   mm_* (structured) -> logs/Multimodal/, results/Multimodal/
#   bert_* (bert)     -> logs/MultimodalBert/, results/MultimodalBert/
#   hybrid_* (hybrid) -> logs/MultimodalHybrid/, results/MultimodalHybrid/
LOG_BASE_DIR="${PROJECT_DIR}/logs"
RESULT_BASE_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_unified"
SESSION_NAME="unified_mm_gpu"

# Conda environment
CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth"

# Create directories
mkdir -p "${LOG_BASE_DIR}/Multimodal"          # logs for mm_* (structured)
mkdir -p "${LOG_BASE_DIR}/MultimodalBert"      # logs for bert_*
mkdir -p "${LOG_BASE_DIR}/MultimodalHybrid"    # logs for hybrid_*
mkdir -p "${RESULT_BASE_DIR}/Multimodal"       # results for mm_* (structured)
mkdir -p "${RESULT_BASE_DIR}/MultimodalBert"   # results for bert_*
mkdir -p "${RESULT_BASE_DIR}/MultimodalHybrid" # results for hybrid_*
mkdir -p "$STATUS_DIR"

# Helper function to get result dir based on experiment name
get_result_dir() {
    local exp=$1
    if [[ "$exp" == mm_* ]]; then
        echo "${RESULT_BASE_DIR}/Multimodal"
    elif [[ "$exp" == bert_* ]]; then
        echo "${RESULT_BASE_DIR}/MultimodalBert"
    elif [[ "$exp" == hybrid_* ]]; then
        echo "${RESULT_BASE_DIR}/MultimodalHybrid"
    else
        echo "${RESULT_BASE_DIR}/Multimodal"  # default
    fi
}

# Helper function to get log dir based on experiment name
get_log_dir() {
    local exp=$1
    if [[ "$exp" == mm_* ]]; then
        echo "${LOG_BASE_DIR}/Multimodal"
    elif [[ "$exp" == bert_* ]]; then
        echo "${LOG_BASE_DIR}/MultimodalBert"
    elif [[ "$exp" == hybrid_* ]]; then
        echo "${LOG_BASE_DIR}/MultimodalHybrid"
    else
        echo "${LOG_BASE_DIR}/Multimodal"  # default
    fi
}

# ============================================================================
# Experiment definitions for each policy source
# Each group is defined once, then prefixed with mm_/bert_/hybrid_
# ============================================================================

# --- LightCNN Baselines ---
_CNN_BASELINE=("cnn_concat" "cnn_concat_median" "cnn_concat_trimmed")

# --- LightCNN Fusion Variants ---
_CNN_GATED=("cnn_gated" "cnn_gated_trimmed")
_CNN_ATTENTION=("cnn_attention" "cnn_attention_trimmed")
_CNN_FILM=("cnn_film" "cnn_film_trimmed")
_CNN_FUSION=("${_CNN_GATED[@]}" "${_CNN_ATTENTION[@]}" "${_CNN_FILM[@]}")

# --- MLP Models ---
_MLP=("mlp_concat" "mlp_gated")

# --- Position-Aware Aggregation (LightCNN) ---
_CNN_AGG=("cnn_concat_attn_agg" "cnn_concat_pos_attn" "cnn_concat_spatial_attn" "cnn_transformer" "cnn_transformer_2d")
_CNN_GATED_AGG=("cnn_gated_transformer" "cnn_gated_transformer_2d")
_CNN_FILM_AGG=("cnn_film_transformer" "cnn_film_transformer_2d")
_ALL_AGG=("${_CNN_AGG[@]}" "${_CNN_GATED_AGG[@]}" "${_CNN_FILM_AGG[@]}")

# --- Custom LightCNN ---
_CNN_CUSTOM=("cnn_small_concat")

# --- ResNet Models ---
_RESNET10=("resnet10_concat" "resnet10_gated")
_RESNET18=("resnet18_concat" "resnet18_gated" "resnet18_film" "resnet18_concat_transformer" "resnet18_concat_transformer_2d")
_RESNET34=("resnet34_concat" "resnet34_gated" "resnet34_film" "resnet34_pretrained")
_RESNET50=("resnet50_concat" "resnet50_gated" "resnet50_imagenet")
_RESNET101=("resnet101_concat" "resnet101_imagenet")
_ALL_RESNET=("${_RESNET10[@]}" "${_RESNET18[@]}" "${_RESNET34[@]}" "${_RESNET50[@]}" "${_RESNET101[@]}")

# --- Patch-level ---
_PATCH=("cnn_concat_patch" "cnn_gated_patch" "cnn_film_patch" "resnet18_concat_patch")

# --- SimCLR Pretrained ---
_SIMCLR_CNN=("simclr_cnn_concat" "simclr_cnn_concat_trimmed" "simclr_cnn_gated" "simclr_cnn_film")
_SIMCLR_AGG=("simclr_cnn_transformer" "simclr_cnn_transformer_2d")
_SIMCLR_MLP=("simclr_mlp_concat")
_SIMCLR_PATCH=("simclr_cnn_concat_patch")
_ALL_SIMCLR=("${_SIMCLR_CNN[@]}" "${_SIMCLR_AGG[@]}" "${_SIMCLR_MLP[@]}" "${_SIMCLR_PATCH[@]}")

# --- MAE Pretrained ---
_MAE_CNN=("mae_cnn_concat" "mae_cnn_concat_trimmed" "mae_cnn_gated" "mae_cnn_film")
_MAE_AGG=("mae_cnn_transformer" "mae_cnn_transformer_2d")
_MAE_PATCH=("mae_cnn_concat_patch")
_ALL_MAE=("${_MAE_CNN[@]}" "${_MAE_AGG[@]}" "${_MAE_PATCH[@]}")

# --- All experiments (without prefix) ---
_ALL_EXPS=(
    "${_CNN_BASELINE[@]}"
    "${_CNN_FUSION[@]}"
    "${_MLP[@]}"
    "${_ALL_AGG[@]}"
    "${_CNN_CUSTOM[@]}"
    "${_ALL_RESNET[@]}"
    "${_PATCH[@]}"
    "${_ALL_SIMCLR[@]}"
    "${_ALL_MAE[@]}"
)

# ============================================================================
# Generate prefixed experiment arrays
# ============================================================================
prefix_array() {
    local prefix=$1
    shift
    local result=()
    for exp in "$@"; do
        result+=("${prefix}${exp}")
    done
    echo "${result[@]}"
}

# STRUCTURED (mm_* prefix)
STRUCT_CNN_BASELINE=($(prefix_array "mm_" "${_CNN_BASELINE[@]}"))
STRUCT_CNN_FUSION=($(prefix_array "mm_" "${_CNN_FUSION[@]}"))
STRUCT_MLP=($(prefix_array "mm_" "${_MLP[@]}"))
STRUCT_AGG=($(prefix_array "mm_" "${_ALL_AGG[@]}"))
STRUCT_CNN_CUSTOM=($(prefix_array "mm_" "${_CNN_CUSTOM[@]}"))
STRUCT_RESNET10=($(prefix_array "mm_" "${_RESNET10[@]}"))
STRUCT_RESNET18=($(prefix_array "mm_" "${_RESNET18[@]}"))
STRUCT_RESNET34=($(prefix_array "mm_" "${_RESNET34[@]}"))
STRUCT_RESNET50=($(prefix_array "mm_" "${_RESNET50[@]}"))
STRUCT_RESNET101=($(prefix_array "mm_" "${_RESNET101[@]}"))
STRUCT_RESNET=($(prefix_array "mm_" "${_ALL_RESNET[@]}"))
STRUCT_PATCH=($(prefix_array "mm_" "${_PATCH[@]}"))
STRUCT_SIMCLR=($(prefix_array "mm_" "${_ALL_SIMCLR[@]}"))
STRUCT_MAE=($(prefix_array "mm_" "${_ALL_MAE[@]}"))
ALL_STRUCTURED=($(prefix_array "mm_" "${_ALL_EXPS[@]}"))

# BERT (bert_* prefix)
BERT_CNN_BASELINE=($(prefix_array "bert_" "${_CNN_BASELINE[@]}"))
BERT_CNN_FUSION=($(prefix_array "bert_" "${_CNN_FUSION[@]}"))
BERT_MLP=($(prefix_array "bert_" "${_MLP[@]}"))
BERT_AGG=($(prefix_array "bert_" "${_ALL_AGG[@]}"))
BERT_CNN_CUSTOM=($(prefix_array "bert_" "${_CNN_CUSTOM[@]}"))
BERT_RESNET10=($(prefix_array "bert_" "${_RESNET10[@]}"))
BERT_RESNET18=($(prefix_array "bert_" "${_RESNET18[@]}"))
BERT_RESNET34=($(prefix_array "bert_" "${_RESNET34[@]}"))
BERT_RESNET50=($(prefix_array "bert_" "${_RESNET50[@]}"))
BERT_RESNET101=($(prefix_array "bert_" "${_RESNET101[@]}"))
BERT_RESNET=($(prefix_array "bert_" "${_ALL_RESNET[@]}"))
BERT_PATCH=($(prefix_array "bert_" "${_PATCH[@]}"))
BERT_SIMCLR=($(prefix_array "bert_" "${_ALL_SIMCLR[@]}"))
BERT_MAE=($(prefix_array "bert_" "${_ALL_MAE[@]}"))
ALL_BERT=($(prefix_array "bert_" "${_ALL_EXPS[@]}"))

# HYBRID (hybrid_* prefix)
HYBRID_CNN_BASELINE=($(prefix_array "hybrid_" "${_CNN_BASELINE[@]}"))
HYBRID_CNN_FUSION=($(prefix_array "hybrid_" "${_CNN_FUSION[@]}"))
HYBRID_MLP=($(prefix_array "hybrid_" "${_MLP[@]}"))
HYBRID_AGG=($(prefix_array "hybrid_" "${_ALL_AGG[@]}"))
HYBRID_CNN_CUSTOM=($(prefix_array "hybrid_" "${_CNN_CUSTOM[@]}"))
HYBRID_RESNET10=($(prefix_array "hybrid_" "${_RESNET10[@]}"))
HYBRID_RESNET18=($(prefix_array "hybrid_" "${_RESNET18[@]}"))
HYBRID_RESNET34=($(prefix_array "hybrid_" "${_RESNET34[@]}"))
HYBRID_RESNET50=($(prefix_array "hybrid_" "${_RESNET50[@]}"))
HYBRID_RESNET101=($(prefix_array "hybrid_" "${_RESNET101[@]}"))
HYBRID_RESNET=($(prefix_array "hybrid_" "${_ALL_RESNET[@]}"))
HYBRID_PATCH=($(prefix_array "hybrid_" "${_PATCH[@]}"))
HYBRID_SIMCLR=($(prefix_array "hybrid_" "${_ALL_SIMCLR[@]}"))
HYBRID_MAE=($(prefix_array "hybrid_" "${_ALL_MAE[@]}"))
ALL_HYBRID=($(prefix_array "hybrid_" "${_ALL_EXPS[@]}"))

# ============================================================================
# COMBINED CATEGORIES
# ============================================================================

# All experiments
ALL_EXPERIMENTS=("${ALL_STRUCTURED[@]}" "${ALL_BERT[@]}" "${ALL_HYBRID[@]}")

# Self-supervised (across all policy sources)
ALL_SSL=("${STRUCT_SIMCLR[@]}" "${STRUCT_MAE[@]}" "${BERT_SIMCLR[@]}" "${BERT_MAE[@]}" "${HYBRID_SIMCLR[@]}" "${HYBRID_MAE[@]}")

# All Patch-level (all policy sources)
ALL_PATCH=("${STRUCT_PATCH[@]}" "${BERT_PATCH[@]}" "${HYBRID_PATCH[@]}")

# All ResNet (all policy sources)
ALL_RESNET_ALL=("${STRUCT_RESNET[@]}" "${BERT_RESNET[@]}" "${HYBRID_RESNET[@]}")

# All Position-Aware Aggregation (all policy sources)
ALL_AGG_ALL=("${STRUCT_AGG[@]}" "${BERT_AGG[@]}" "${HYBRID_AGG[@]}")

# Baseline comparisons (one from each policy source)
BASELINE_COMPARE=("mm_cnn_concat" "bert_cnn_concat" "hybrid_cnn_concat")

# Quick test (3 experiments for quick verification)
QUICK_TEST=("mm_cnn_concat" "bert_cnn_concat" "hybrid_cnn_concat")

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# Help
# ============================================================================
show_help() {
    echo "Usage: bash $0 [options]"
    echo ""
    echo "** Unified Multimodal Runner with GPU Normalization **"
    echo "   Supports structured (12-dim), BERT (64-dim), and hybrid (76-dim) policy"
    echo "   Each policy source has 56 experiments, total: 168 experiments"
    echo ""
    echo "Options:"
    echo "  --help, -h        Show help"
    echo "  --list, -l        List all experiments"
    echo "  --gpus GPUS       Specify GPUs (e.g., 0,1,2,3)"
    echo "  --parallel N      Parallel count"
    echo "  --category CAT    Experiment category (see below)"
    echo "  --exp EXP1,EXP2   Specify experiment names"
    echo "  --seed SEED       Override default seed list, run single seed"
    echo "  --seeds S1,S2     Override default seed list (default: ${DEFAULT_SEEDS[*]})"
    echo "  --dry-run         Preview mode"
    echo "  --resume          Skip completed experiments"
    echo ""
    echo "Available categories (--category):"
    echo ""
    echo "  === By Policy Source (56 each) ==="
    echo "  structured        All structured (mm_*) experiments (${#ALL_STRUCTURED[@]})"
    echo "  bert              All BERT experiments (${#ALL_BERT[@]})"
    echo "  hybrid            All Hybrid experiments (${#ALL_HYBRID[@]})"
    echo "  all               All experiments (${#ALL_EXPERIMENTS[@]})"
    echo ""
    echo "  === Sub-categories (same structure for all policy sources) ==="
    echo "  {struct|bert|hybrid}_baseline   LightCNN Concat baselines (3 each)"
    echo "  {struct|bert|hybrid}_fusion     LightCNN fusion variants (6 each)"
    echo "  {struct|bert|hybrid}_mlp        MLP models (2 each)"
    echo "  {struct|bert|hybrid}_agg        Position-aware aggregation (9 each)"
    echo "  {struct|bert|hybrid}_resnet     All ResNet models (16 each)"
    echo "  {struct|bert|hybrid}_patch      Patch-level (4 each)"
    echo "  {struct|bert|hybrid}_simclr     SimCLR pretrained (8 each)"
    echo "  {struct|bert|hybrid}_mae        MAE pretrained (7 each)"
    echo ""
    echo "  === Cross-Policy Categories ==="
    echo "  ssl               All SSL pretrained (${#ALL_SSL[@]})"
    echo "  patch             All patch-level (${#ALL_PATCH[@]})"
    echo "  resnet_all        All ResNet (${#ALL_RESNET_ALL[@]})"
    echo "  agg_all           All position-aware aggregation (${#ALL_AGG_ALL[@]})"
    echo "  compare           Baseline comparison (${#BASELINE_COMPARE[@]})"
    echo "  quick             Quick test (${#QUICK_TEST[@]})"
}

list_experiments() {
    echo "All Unified Multimodal Experiments (168 total)"
    echo "================================================"
    echo ""
    echo "Each policy source has 56 identical experiment configurations:"
    echo "  - LightCNN baselines:     3 (concat, concat_median, concat_trimmed)"
    echo "  - LightCNN fusion:        6 (gated, attention, film × 2 aggregations)"
    echo "  - MLP:                    2"
    echo "  - Position-Aware Agg:     9"
    echo "  - Custom LightCNN:        1"
    echo "  - ResNet (10/18/34/50/101): 16"
    echo "  - Patch-level:            4"
    echo "  - SimCLR pretrained:      8"
    echo "  - MAE pretrained:         7"
    echo ""
    echo "=== STRUCTURED POLICY (mm_*, 12-dim) ==="
    for exp in "${ALL_STRUCTURED[@]}"; do echo "  $exp"; done
    echo ""
    echo "=== BERT POLICY (bert_*, 64-dim) ==="
    for exp in "${ALL_BERT[@]}"; do echo "  $exp"; done
    echo ""
    echo "=== HYBRID POLICY (hybrid_*, 76-dim) ==="
    for exp in "${ALL_HYBRID[@]}"; do echo "  $exp"; done
    echo ""
    echo "=== SUMMARY ==="
    echo "  Structured: ${#ALL_STRUCTURED[@]} experiments"
    echo "  BERT:       ${#ALL_BERT[@]} experiments"
    echo "  Hybrid:     ${#ALL_HYBRID[@]} experiments"
    echo "  TOTAL:      ${#ALL_EXPERIMENTS[@]} experiments"
}

# ============================================================================
# Parse arguments
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
CATEGORY="all"
CUSTOM_EXPS_STR=""
DRY_RUN=false
RESUME=false
SEEDS=("${DEFAULT_SEEDS[@]}")

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --category) CATEGORY=$2; shift 2 ;;
        --exp) CUSTOM_EXPS_STR=$2; shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Set parallel count
if [ "$PARALLEL_COUNT" -eq 0 ]; then
    PARALLEL_COUNT=${#GPUS[@]}
fi
if [ "$PARALLEL_COUNT" -gt "${#GPUS[@]}" ]; then
    PARALLEL_COUNT=${#GPUS[@]}
fi

# ============================================================================
# Select experiments
# ============================================================================
if [ -n "$CUSTOM_EXPS_STR" ]; then
    IFS=',' read -ra SELECTED_EXPS <<< "$CUSTOM_EXPS_STR"
else
    case $CATEGORY in
        # By policy source
        structured) SELECTED_EXPS=("${ALL_STRUCTURED[@]}") ;;
        bert) SELECTED_EXPS=("${ALL_BERT[@]}") ;;
        hybrid) SELECTED_EXPS=("${ALL_HYBRID[@]}") ;;
        all) SELECTED_EXPS=("${ALL_EXPERIMENTS[@]}") ;;

        # Structured sub-categories
        struct_baseline) SELECTED_EXPS=("${STRUCT_CNN_BASELINE[@]}") ;;
        struct_fusion) SELECTED_EXPS=("${STRUCT_CNN_FUSION[@]}") ;;
        struct_mlp) SELECTED_EXPS=("${STRUCT_MLP[@]}") ;;
        struct_agg) SELECTED_EXPS=("${STRUCT_AGG[@]}") ;;
        struct_resnet) SELECTED_EXPS=("${STRUCT_RESNET[@]}") ;;
        struct_patch) SELECTED_EXPS=("${STRUCT_PATCH[@]}") ;;
        struct_simclr) SELECTED_EXPS=("${STRUCT_SIMCLR[@]}") ;;
        struct_mae) SELECTED_EXPS=("${STRUCT_MAE[@]}") ;;

        # BERT sub-categories
        bert_baseline) SELECTED_EXPS=("${BERT_CNN_BASELINE[@]}") ;;
        bert_fusion) SELECTED_EXPS=("${BERT_CNN_FUSION[@]}") ;;
        bert_mlp) SELECTED_EXPS=("${BERT_MLP[@]}") ;;
        bert_agg) SELECTED_EXPS=("${BERT_AGG[@]}") ;;
        bert_resnet) SELECTED_EXPS=("${BERT_RESNET[@]}") ;;
        bert_patch) SELECTED_EXPS=("${BERT_PATCH[@]}") ;;
        bert_simclr) SELECTED_EXPS=("${BERT_SIMCLR[@]}") ;;
        bert_mae) SELECTED_EXPS=("${BERT_MAE[@]}") ;;

        # Hybrid sub-categories
        hybrid_baseline) SELECTED_EXPS=("${HYBRID_CNN_BASELINE[@]}") ;;
        hybrid_fusion) SELECTED_EXPS=("${HYBRID_CNN_FUSION[@]}") ;;
        hybrid_mlp) SELECTED_EXPS=("${HYBRID_MLP[@]}") ;;
        hybrid_agg) SELECTED_EXPS=("${HYBRID_AGG[@]}") ;;
        hybrid_resnet) SELECTED_EXPS=("${HYBRID_RESNET[@]}") ;;
        hybrid_patch) SELECTED_EXPS=("${HYBRID_PATCH[@]}") ;;
        hybrid_simclr) SELECTED_EXPS=("${HYBRID_SIMCLR[@]}") ;;
        hybrid_mae) SELECTED_EXPS=("${HYBRID_MAE[@]}") ;;

        # Cross-policy categories
        ssl) SELECTED_EXPS=("${ALL_SSL[@]}") ;;
        patch) SELECTED_EXPS=("${ALL_PATCH[@]}") ;;
        resnet_all) SELECTED_EXPS=("${ALL_RESNET_ALL[@]}") ;;
        agg_all) SELECTED_EXPS=("${ALL_AGG_ALL[@]}") ;;
        compare) SELECTED_EXPS=("${BASELINE_COMPARE[@]}") ;;
        quick) SELECTED_EXPS=("${QUICK_TEST[@]}") ;;

        *) echo "Unknown category: $CATEGORY"; show_help; exit 1 ;;
    esac
fi

# Generate task list (experiment + seed)
TASK_LIST=()
for exp in "${SELECTED_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp}:${seed}")
    done
done

# Resume: filter completed experiments
if [ "$RESUME" = true ]; then
    FILTERED=()
    for task in "${TASK_LIST[@]}"; do
        exp="${task%%:*}"
        seed="${task##*:}"

        # Build run_id
        if [ "$seed" == "42" ]; then
            run_id="${exp}"
        else
            run_id="${exp}_seed${seed}"
        fi

        # Get result directory based on experiment prefix
        exp_result_dir=$(get_result_dir "$exp")
        result_file="${exp_result_dir}/${run_id}_results.pkl"

        if [ -f "$result_file" ]; then
            echo "[Skip] $exp (seed=$seed, already done)"
        else
            FILTERED+=("$task")
        fi
    done
    TASK_LIST=("${FILTERED[@]}")
fi

TOTAL=${#TASK_LIST[@]}
NUM_EXPS=${#SELECTED_EXPS[@]}
NUM_SEEDS=${#SEEDS[@]}

echo "=============================================="
echo "  Unified Multimodal Runner (GPU Normalize)"
echo "=============================================="
echo "GPU list: ${GPUS[*]}"
echo "Parallel: $PARALLEL_COUNT"
echo "Experiments: $NUM_EXPS"
echo "Seeds: ${SEEDS[*]}"
echo "Total tasks: $TOTAL (${NUM_EXPS} exps x ${NUM_SEEDS} seeds)"
echo "Category: $CATEGORY"
echo "Result dirs: Multimodal/ | MultimodalBert/ | MultimodalHybrid/"
echo ""
echo "** Using --normalize_on_gpu for faster training **"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No experiments to run"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "Will run the following experiments:"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        exp="${task%%:*}"
        seed="${task##*:}"
        echo "  $((i+1)). $exp (seed=$seed) [--normalize_on_gpu]"
    done
    exit 0
fi

# ============================================================================
# Clean status files
# ============================================================================
rm -f "${STATUS_DIR}"/gpu_*.lock
rm -f "${STATUS_DIR}"/exp_*.status

# ============================================================================
# GPU management functions
# ============================================================================
get_free_gpu() {
    for gpu in "${GPUS[@]}"; do
        local lock_file="${STATUS_DIR}/gpu_${gpu}.lock"
        if [ ! -f "$lock_file" ]; then
            echo "$gpu"
            return 0
        fi
    done
    return 1
}

lock_gpu() {
    local gpu=$1
    local exp=$2
    echo "$exp" > "${STATUS_DIR}/gpu_${gpu}.lock"
}

get_done_count() {
    local count=0
    for f in "${STATUS_DIR}"/exp_*.status; do
        if [ -f "$f" ]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# ============================================================================
# Create tmux session
# ============================================================================
echo "Creating tmux session: $SESSION_NAME"

# Kill old session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new session
tmux new-session -d -s "$SESSION_NAME" -n "monitor"

# Monitor window
tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== Unified Multimodal Monitor (GPU Normalize) ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Total: $TOTAL | Parallel: $PARALLEL_COUNT | Category: $CATEGORY'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Ctrl+B w to view windows | Ctrl+B d to detach'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== Running ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== Done: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# Start experiment in tmux
# ============================================================================
start_experiment() {
    local exp=$1
    local gpu=$2
    local seed=$3

    # Task ID
    local task_id="${exp}_seed${seed}"
    # Get log directory based on experiment prefix
    local exp_log_dir=$(get_log_dir "$exp")
    local log_file="${exp_log_dir}/${task_id}.log"

    # Build run_id (consistent with train_multimodal_bert.py)
    if [ "$seed" == "42" ]; then
        run_id="${exp}"
    else
        run_id="${exp}_seed${seed}"
    fi

    # tmux window name
    local window_name="${exp}_s${seed}"

    echo "[$(date '+%H:%M:%S')] Starting: $exp (seed=$seed) -> GPU $gpu [--normalize_on_gpu]"

    # Lock GPU
    lock_gpu "$gpu" "$task_id"

    # Create new window
    tmux new-window -t "$SESSION_NAME" -n "$window_name"

    # Send commands
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH && echo 'Env activated: '${CONDA_ENV}" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== Experiment: $exp | GPU: $gpu | Seed: $seed | GPU Normalize: ON ==='" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Start time:' \$(date)" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" Enter

    # Run experiment (using train_multimodal_bert.py with --normalize_on_gpu)
    local cmd="python train_multimodal_bert.py --exp $exp --gpu $gpu --seed $seed --normalize_on_gpu 2>&1 | tee $log_file"
    cmd="$cmd; if [ \$? -eq 0 ]; then echo success > ${STATUS_DIR}/exp_${task_id}.status; else echo failed > ${STATUS_DIR}/exp_${task_id}.status; fi"
    cmd="$cmd; rm -f ${STATUS_DIR}/gpu_${gpu}.lock"
    cmd="$cmd; echo ''; echo '=== Experiment done ==='; echo 'End time:' \$(date)"

    tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" Enter
}

# ============================================================================
# Main scheduling loop
# ============================================================================
echo ""
echo "Starting experiment scheduler..."
echo ""

QUEUE_INDEX=0

# Start initial batch
i=0
while [ $i -lt $PARALLEL_COUNT ] && [ $QUEUE_INDEX -lt $TOTAL ]; do
    task="${TASK_LIST[$QUEUE_INDEX]}"
    exp="${task%%:*}"
    seed="${task##*:}"
    gpu="${GPUS[$i]}"
    start_experiment "$exp" "$gpu" "$seed"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
    i=$((i + 1))
    sleep 1
done

echo ""
echo "Started $QUEUE_INDEX tasks"
echo ""

# Schedule remaining experiments
while [ $QUEUE_INDEX -lt $TOTAL ]; do
    sleep 5

    FREE_GPU=$(get_free_gpu) || continue

    task="${TASK_LIST[$QUEUE_INDEX]}"
    exp="${task%%:*}"
    seed="${task##*:}"
    start_experiment "$exp" "$FREE_GPU" "$seed"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
done

echo "All $TOTAL tasks scheduled"
echo ""

# ============================================================================
# Wait for completion
# ============================================================================
echo "Waiting for completion... (Ctrl+C to exit script, experiments continue in tmux)"
echo ""

while true; do
    DONE=$(get_done_count)
    printf "\rProgress: $DONE / $TOTAL    "

    if [ "$DONE" -ge "$TOTAL" ]; then
        echo ""
        break
    fi

    sleep 5
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "  All experiments done! (Unified, GPU Normalize)"
echo "=============================================="

SUCCESS=0
FAILED=0

for task in "${TASK_LIST[@]}"; do
    exp="${task%%:*}"
    seed="${task##*:}"
    task_id="${exp}_seed${seed}"

    status_file="${STATUS_DIR}/exp_${task_id}.status"
    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  [FAILED] $exp (seed=$seed)"
        fi
    else
        echo "  [UNKNOWN] $exp (seed=$seed)"
    fi
done

echo ""
echo "Success: $SUCCESS | Failed: $FAILED"
echo "Logs: ${LOG_BASE_DIR}/Multimodal | MultimodalBert | MultimodalHybrid"
echo ""

read -p "Attach to tmux session? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
