#!/bin/bash
# =============================================================================
# Multimodal Experiments Runner with Tmux
# =============================================================================
#
# Runs all multimodal experiments sequentially in tmux windows.
# Each experiment runs with 3 seeds (42, 123, 456).
# Waits for each experiment to complete before starting the next.
#
# Usage:
#   ./run_mm_tmux.sh <GPU_ID> [GROUP]
#
# Arguments:
#   GPU_ID - GPU to use (0, 1, 2, ...)
#   GROUP  - Experiment group: all, city, patch, cnn, quick (default: all)
#
# Examples:
#   ./run_mm_tmux.sh 0           # All experiments on GPU 0
#   ./run_mm_tmux.sh 1 city      # City-level only on GPU 1
#   ./run_mm_tmux.sh 0 quick     # Quick test (3 experiments) on GPU 0
#
# To monitor:
#   tmux attach -t mm_exp        # Attach to session
#   Ctrl+b n                      # Next window
#   Ctrl+b p                      # Previous window
#   Ctrl+b d                      # Detach
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

GPU_ID=${1:-0}
GROUP=${2:-"all"}

WORK_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
CONDA_ENV="alphaearth"
TMUX_SESSION="mm_exp_gpu${GPU_ID}"
SEEDS=(42 123 456)

# Log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${WORK_DIR}/logs/mm_${GROUP}_gpu${GPU_ID}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# =============================================================================
# Experiment Groups
# =============================================================================

# City-level experiments (16 total)
CITY_EXPERIMENTS=(
    mm_cnn_concat
    mm_cnn_concat_median
    mm_cnn_concat_trimmed
    mm_cnn_gated
    mm_cnn_gated_trimmed
    mm_cnn_attention
    mm_cnn_attention_trimmed
    mm_cnn_film
    mm_cnn_film_trimmed
    mm_mlp_concat
    mm_mlp_gated
    mm_resnet18_concat
    mm_resnet18_gated
    mm_resnet18_film
    mm_cnn_small_concat
    mm_resnet34_pretrained
)

# Patch-level experiments (4 total)
PATCH_EXPERIMENTS=(
    mm_cnn_concat_patch
    mm_cnn_gated_patch
    mm_cnn_film_patch
    mm_resnet18_concat_patch
)

# Quick test (3 experiments for testing)
QUICK_EXPERIMENTS=(
    mm_cnn_concat
    mm_cnn_gated
    mm_cnn_film
)

# CNN only experiments
CNN_EXPERIMENTS=(
    mm_cnn_concat
    mm_cnn_concat_trimmed
    mm_cnn_gated
    mm_cnn_gated_trimmed
    mm_cnn_film
    mm_cnn_film_trimmed
)

# Select experiments
case "$GROUP" in
    all)
        EXPERIMENTS=("${CITY_EXPERIMENTS[@]}" "${PATCH_EXPERIMENTS[@]}")
        ;;
    city)
        EXPERIMENTS=("${CITY_EXPERIMENTS[@]}")
        ;;
    patch)
        EXPERIMENTS=("${PATCH_EXPERIMENTS[@]}")
        ;;
    cnn)
        EXPERIMENTS=("${CNN_EXPERIMENTS[@]}")
        ;;
    quick)
        EXPERIMENTS=("${QUICK_EXPERIMENTS[@]}")
        ;;
    *)
        echo "Unknown group: $GROUP"
        echo "Available: all, city, patch, cnn, quick"
        exit 1
        ;;
esac

TOTAL=$((${#EXPERIMENTS[@]} * ${#SEEDS[@]}))

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "============================================================"
echo "Multimodal Experiments Runner (Tmux Version)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  GPU ID:        $GPU_ID"
echo "  Group:         $GROUP"
echo "  Experiments:   ${#EXPERIMENTS[@]}"
echo "  Seeds:         ${SEEDS[*]}"
echo "  Total runs:    $TOTAL"
echo "  Tmux session:  $TMUX_SESSION"
echo "  Log directory: $LOG_DIR"
echo ""
echo "Experiments to run:"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo ""

# Check GPU
if ! nvidia-smi -i "$GPU_ID" &> /dev/null; then
    echo "ERROR: GPU $GPU_ID not available!"
    nvidia-smi
    exit 1
fi

echo "GPU $GPU_ID status:"
nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check conda env
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "ERROR: Conda environment '$CONDA_ENV' not found!"
    conda env list
    exit 1
fi

# Confirm
read -p "Press Enter to start, Ctrl+C to cancel..."

# =============================================================================
# Create Tmux Session
# =============================================================================

# Kill existing session if exists
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# Create new session
tmux new-session -d -s "$TMUX_SESSION" -n "control"
echo "Created tmux session: $TMUX_SESSION"

# Create status file
STATUS_FILE="${LOG_DIR}/status.txt"
echo "Multimodal Experiments Status" > "$STATUS_FILE"
echo "Started: $(date)" >> "$STATUS_FILE"
echo "GPU: $GPU_ID, Group: $GROUP" >> "$STATUS_FILE"
echo "Total: $TOTAL experiments" >> "$STATUS_FILE"
echo "---" >> "$STATUS_FILE"

# =============================================================================
# Run Experiments
# =============================================================================

COUNT=0
FAILED=0

for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((COUNT++))

        RUN_ID="${exp}_seed${seed}"
        WINDOW_NAME="exp_${COUNT}"
        LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

        echo ""
        echo "============================================================"
        echo "[$COUNT/$TOTAL] Starting: $RUN_ID"
        echo "============================================================"
        echo "$(date): [$COUNT/$TOTAL] $RUN_ID - STARTED" >> "$STATUS_FILE"

        # Create new tmux window
        tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME"

        # Build the command
        CMD="cd ${WORK_DIR} && \
source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=online && \
echo '========================================' && \
echo 'Experiment: ${exp}' && \
echo 'Seed: ${seed}' && \
echo 'GPU: ${GPU_ID}' && \
echo 'Log: ${LOG_FILE}' && \
echo '========================================' && \
python train_multimodal.py --exp ${exp} --gpu 0 --seed ${seed} 2>&1 | tee ${LOG_FILE} && \
echo '' && \
echo '========================================' && \
echo 'EXPERIMENT COMPLETED SUCCESSFULLY' && \
echo '========================================' || \
echo 'EXPERIMENT FAILED'"

        # Send command to window
        tmux send-keys -t "${TMUX_SESSION}:${WINDOW_NAME}" "$CMD" Enter

        echo "  Window: ${TMUX_SESSION}:${WINDOW_NAME}"
        echo "  Log: $LOG_FILE"
        echo ""
        echo "Waiting for completion..."

        # Wait for log file to appear
        WAIT=0
        while [ ! -f "$LOG_FILE" ] && [ $WAIT -lt 120 ]; do
            sleep 1
            ((WAIT++))
        done

        if [ ! -f "$LOG_FILE" ]; then
            echo "WARNING: Log file not created after 120s"
        fi

        # Monitor for completion
        COMPLETED=false
        while true; do
            # Check if window still exists
            if ! tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q "$WINDOW_NAME"; then
                COMPLETED=true
                break
            fi

            # Check log for completion markers
            if [ -f "$LOG_FILE" ]; then
                if grep -q "Results saved to" "$LOG_FILE" 2>/dev/null; then
                    COMPLETED=true
                    sleep 2  # Wait for final output
                    break
                fi

                if grep -q "EXPERIMENT COMPLETED" "$LOG_FILE" 2>/dev/null; then
                    COMPLETED=true
                    sleep 2
                    break
                fi

                # Check for fatal errors
                if grep -q "CUDA out of memory\|RuntimeError:" "$LOG_FILE" 2>/dev/null; then
                    echo "ERROR detected in experiment!"
                    break
                fi
            fi

            # Show progress periodically
            if [ $((SECONDS % 30)) -eq 0 ] && [ -f "$LOG_FILE" ]; then
                LAST_EPOCH=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_FILE" 2>/dev/null | tail -1)
                if [ -n "$LAST_EPOCH" ]; then
                    echo "  Progress: $LAST_EPOCH"
                fi
            fi

            sleep 5
        done

        # Close tmux window
        tmux kill-window -t "${TMUX_SESSION}:${WINDOW_NAME}" 2>/dev/null || true

        # Check result
        if [ -f "$LOG_FILE" ] && grep -q "Results saved" "$LOG_FILE" 2>/dev/null; then
            echo "SUCCESS: $RUN_ID completed"
            echo "$(date): [$COUNT/$TOTAL] $RUN_ID - SUCCESS" >> "$STATUS_FILE"

            # Show metrics
            if grep -q "Test Set:" "$LOG_FILE"; then
                echo "  Test Results:"
                grep -A 4 "Test Set:" "$LOG_FILE" | tail -4 | sed 's/^/    /'
            fi
        else
            echo "FAILED: $RUN_ID"
            echo "$(date): [$COUNT/$TOTAL] $RUN_ID - FAILED" >> "$STATUS_FILE"
            ((FAILED++))

            if [ -f "$LOG_FILE" ]; then
                echo "  Last 5 lines of log:"
                tail -5 "$LOG_FILE" | sed 's/^/    /'
            fi

            # Ask whether to continue
            read -p "Continue with remaining experiments? (y/n): " CONT
            if [ "$CONT" != "y" ] && [ "$CONT" != "Y" ]; then
                echo "Stopping experiments."
                break 2
            fi
        fi

        # Brief pause between experiments
        sleep 3
    done
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================"
echo ""
echo "Summary:"
echo "  Total: $TOTAL"
echo "  Completed: $((COUNT))"
echo "  Failed: $FAILED"
echo "  Success rate: $(( (COUNT - FAILED) * 100 / COUNT ))%"
echo ""
echo "Logs saved to: $LOG_DIR"
echo "Status file: $STATUS_FILE"
echo ""

# Final status
echo "---" >> "$STATUS_FILE"
echo "Finished: $(date)" >> "$STATUS_FILE"
echo "Total: $COUNT, Failed: $FAILED" >> "$STATUS_FILE"

# Clean up tmux session
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

echo "Done!"
