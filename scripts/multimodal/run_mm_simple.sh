#!/bin/bash
# =============================================================================
# Simple Multimodal Experiments Runner (Background Mode)
# =============================================================================
#
# Usage:
#   ./run_mm_simple.sh <GPU_ID> [EXPERIMENT_GROUP]
#
# Examples:
#   ./run_mm_simple.sh 0          # Run all experiments on GPU 0
#   ./run_mm_simple.sh 1 city     # Run city-level experiments on GPU 1
#   ./run_mm_simple.sh 2 patch    # Run patch-level experiments on GPU 2
#
# =============================================================================

GPU_ID=${1:-0}
GROUP=${2:-"all"}

# Configuration
WORK_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
CONDA_ENV="alphaearth"
SEEDS=(42 123 456)
LOG_DIR="${WORK_DIR}/logs/mm_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Define experiment groups
declare -A EXPERIMENT_GROUPS
EXPERIMENT_GROUPS["city"]="mm_cnn_concat mm_cnn_concat_median mm_cnn_concat_trimmed mm_cnn_gated mm_cnn_gated_trimmed mm_cnn_attention mm_cnn_attention_trimmed mm_cnn_film mm_cnn_film_trimmed mm_mlp_concat mm_mlp_gated mm_resnet18_concat mm_resnet18_gated mm_resnet18_film mm_cnn_small_concat mm_resnet34_pretrained"
EXPERIMENT_GROUPS["patch"]="mm_cnn_concat_patch mm_cnn_gated_patch mm_cnn_film_patch mm_resnet18_concat_patch"
EXPERIMENT_GROUPS["all"]="${EXPERIMENT_GROUPS[city]} ${EXPERIMENT_GROUPS[patch]}"
EXPERIMENT_GROUPS["cnn"]="mm_cnn_concat mm_cnn_concat_trimmed mm_cnn_gated mm_cnn_gated_trimmed mm_cnn_film mm_cnn_film_trimmed"
EXPERIMENT_GROUPS["quick"]="mm_cnn_concat mm_cnn_gated mm_cnn_film"

# Get experiments for selected group
if [ -z "${EXPERIMENT_GROUPS[$GROUP]}" ]; then
    echo "Unknown group: $GROUP"
    echo "Available: all, city, patch, cnn, quick"
    exit 1
fi

EXPERIMENTS=(${EXPERIMENT_GROUPS[$GROUP]})
TOTAL=$((${#EXPERIMENTS[@]} * ${#SEEDS[@]}))

echo "============================================================"
echo "Multimodal Experiments Runner"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Group: $GROUP"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $TOTAL"
echo "Log directory: $LOG_DIR"
echo "============================================================"

# Check GPU
if ! nvidia-smi -i "$GPU_ID" &> /dev/null; then
    echo "ERROR: GPU $GPU_ID not available!"
    exit 1
fi

# Main log file
MAIN_LOG="${LOG_DIR}/main.log"
echo "Starting experiments at $(date)" | tee "$MAIN_LOG"

# Counter
COUNT=0
FAILED=0

# Run experiments
for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((COUNT++))
        RUN_ID="${exp}_seed${seed}"
        LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

        echo ""
        echo "[$COUNT/$TOTAL] Running: $RUN_ID" | tee -a "$MAIN_LOG"
        echo "  Start: $(date)" | tee -a "$MAIN_LOG"

        # Run experiment
        cd "$WORK_DIR"
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV"

        CUDA_VISIBLE_DEVICES=$GPU_ID python train_multimodal.py \
            --exp "$exp" \
            --gpu 0 \
            --seed "$seed" \
            > "$LOG_FILE" 2>&1

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "  Status: SUCCESS" | tee -a "$MAIN_LOG"
            # Extract final metrics
            if grep -q "Test Set:" "$LOG_FILE"; then
                grep -A 5 "Test Set:" "$LOG_FILE" | head -6 | tee -a "$MAIN_LOG"
            fi
        else
            echo "  Status: FAILED (exit code: $EXIT_CODE)" | tee -a "$MAIN_LOG"
            ((FAILED++))
            tail -10 "$LOG_FILE" | tee -a "$MAIN_LOG"
        fi

        echo "  End: $(date)" | tee -a "$MAIN_LOG"
    done
done

# Summary
echo ""
echo "============================================================" | tee -a "$MAIN_LOG"
echo "SUMMARY" | tee -a "$MAIN_LOG"
echo "============================================================" | tee -a "$MAIN_LOG"
echo "Completed: $COUNT/$TOTAL" | tee -a "$MAIN_LOG"
echo "Failed: $FAILED" | tee -a "$MAIN_LOG"
echo "Finished at: $(date)" | tee -a "$MAIN_LOG"
echo "Logs saved to: $LOG_DIR" | tee -a "$MAIN_LOG"
