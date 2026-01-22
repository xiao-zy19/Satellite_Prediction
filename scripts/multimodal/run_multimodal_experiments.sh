#!/bin/bash
# =============================================================================
# Multimodal Experiments Runner
# =============================================================================
#
# This script runs all multimodal experiments sequentially on a specified GPU.
# Each experiment is run with 3 different seeds (42, 123, 456).
# Each experiment runs in a tmux window and waits for completion before starting the next.
#
# Usage:
#   ./run_multimodal_experiments.sh [GPU_ID] [EXPERIMENT_GROUP]
#
# Arguments:
#   GPU_ID           - GPU ID to use (default: 0)
#   EXPERIMENT_GROUP - Which experiments to run:
#                      "all"      - All experiments (default)
#                      "city"     - City-level experiments only
#                      "patch"    - Patch-level experiments only
#                      "cnn"      - LightCNN experiments only
#                      "resnet"   - ResNet experiments only
#                      "single"   - Run a single experiment (specify with -e flag)
#
# Examples:
#   ./run_multimodal_experiments.sh 0 all       # Run all experiments on GPU 0
#   ./run_multimodal_experiments.sh 1 city      # Run city-level experiments on GPU 1
#   ./run_multimodal_experiments.sh 2 patch     # Run patch-level experiments on GPU 2
#   ./run_multimodal_experiments.sh 0 single -e mm_cnn_concat  # Run single experiment
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default values
GPU_ID=${1:-0}
EXPERIMENT_GROUP=${2:-"all"}
SINGLE_EXPERIMENT=""

# Parse additional flags
shift 2 2>/dev/null || true
while getopts "e:" opt; do
    case $opt in
        e) SINGLE_EXPERIMENT="$OPTARG" ;;
        *) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Seeds to run
SEEDS=(42 123 456)

# Conda environment
CONDA_ENV="alphaearth"

# Working directory
WORK_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"

# Tmux session name
TMUX_SESSION="multimodal_exp"

# Log directory
LOG_DIR="${WORK_DIR}/logs/multimodal_runs"
mkdir -p "$LOG_DIR"

# =============================================================================
# Experiment Lists
# =============================================================================

# City-level experiments
CITY_LEVEL_EXPERIMENTS=(
    "mm_cnn_concat"
    "mm_cnn_concat_median"
    "mm_cnn_concat_trimmed"
    "mm_cnn_gated"
    "mm_cnn_gated_trimmed"
    "mm_cnn_attention"
    "mm_cnn_attention_trimmed"
    "mm_cnn_film"
    "mm_cnn_film_trimmed"
    "mm_mlp_concat"
    "mm_mlp_gated"
    "mm_resnet18_concat"
    "mm_resnet18_gated"
    "mm_resnet18_film"
    "mm_cnn_small_concat"
    "mm_resnet34_pretrained"
)

# Patch-level experiments
PATCH_LEVEL_EXPERIMENTS=(
    "mm_cnn_concat_patch"
    "mm_cnn_gated_patch"
    "mm_cnn_film_patch"
    "mm_resnet18_concat_patch"
)

# LightCNN experiments only
CNN_EXPERIMENTS=(
    "mm_cnn_concat"
    "mm_cnn_concat_median"
    "mm_cnn_concat_trimmed"
    "mm_cnn_gated"
    "mm_cnn_gated_trimmed"
    "mm_cnn_attention"
    "mm_cnn_attention_trimmed"
    "mm_cnn_film"
    "mm_cnn_film_trimmed"
    "mm_cnn_small_concat"
    "mm_cnn_concat_patch"
    "mm_cnn_gated_patch"
    "mm_cnn_film_patch"
)

# ResNet experiments only
RESNET_EXPERIMENTS=(
    "mm_resnet18_concat"
    "mm_resnet18_gated"
    "mm_resnet18_film"
    "mm_resnet34_pretrained"
    "mm_resnet18_concat_patch"
)

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_info() {
    echo "[INFO] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Check if conda environment exists
check_conda_env() {
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        print_error "Conda environment '${CONDA_ENV}' not found!"
        echo "Available environments:"
        conda env list
        exit 1
    fi
}

# Check if tmux is installed
check_tmux() {
    if ! command -v tmux &> /dev/null; then
        print_error "tmux is not installed. Please install it first."
        exit 1
    fi
}

# Check GPU availability
check_gpu() {
    if ! nvidia-smi -i "$GPU_ID" &> /dev/null; then
        print_error "GPU $GPU_ID is not available!"
        nvidia-smi
        exit 1
    fi
    print_info "Using GPU $GPU_ID"
    nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.total,memory.free --format=csv
}

# Run a single experiment with a specific seed
run_experiment() {
    local exp_name=$1
    local seed=$2
    local run_id="${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${run_id}_$(date +%Y%m%d_%H%M%S).log"

    print_header "Running: ${run_id}"
    print_info "Log file: ${log_file}"
    print_info "Start time: $(get_timestamp)"

    # Create the command to run
    local cmd="cd ${WORK_DIR} && \
source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate ${CONDA_ENV} && \
export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
export WANDB_MODE=online && \
python train_multimodal.py --exp ${exp_name} --gpu 0 --seed ${seed} 2>&1 | tee ${log_file}"

    # Check if tmux session exists, create if not
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux new-session -d -s "$TMUX_SESSION" -n "main"
        print_info "Created new tmux session: $TMUX_SESSION"
    fi

    # Create a new window for this experiment
    local window_name="${exp_name}_s${seed}"
    tmux new-window -t "$TMUX_SESSION" -n "$window_name"

    # Send the command to the window
    tmux send-keys -t "${TMUX_SESSION}:${window_name}" "$cmd" Enter

    print_info "Experiment started in tmux window: ${TMUX_SESSION}:${window_name}"
    print_info "To attach: tmux attach -t ${TMUX_SESSION}"

    # Wait for the experiment to complete by monitoring the log file
    print_info "Waiting for experiment to complete..."

    # Wait for log file to be created
    local wait_count=0
    while [ ! -f "$log_file" ] && [ $wait_count -lt 60 ]; do
        sleep 1
        ((wait_count++))
    done

    if [ ! -f "$log_file" ]; then
        print_error "Log file not created after 60 seconds. Check tmux window."
        return 1
    fi

    # Monitor the log file for completion indicators
    local completed=false
    local error_occurred=false

    while true; do
        # Check if the process is still running in tmux
        if ! tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q "$window_name"; then
            # Window closed, check if experiment completed successfully
            if grep -q "Training completed" "$log_file" || grep -q "Results saved" "$log_file"; then
                completed=true
            fi
            break
        fi

        # Check log file for completion or error
        if grep -q "Training completed" "$log_file" 2>/dev/null; then
            completed=true
            break
        fi

        if grep -q "Results saved" "$log_file" 2>/dev/null; then
            completed=true
            break
        fi

        if grep -q "Error\|Exception\|Traceback" "$log_file" 2>/dev/null; then
            # Check if it's a fatal error (not just a warning)
            if grep -q "CUDA out of memory\|RuntimeError\|ValueError\|KeyError" "$log_file" 2>/dev/null; then
                error_occurred=true
                break
            fi
        fi

        # Print progress every 60 seconds
        if [ $((SECONDS % 60)) -eq 0 ]; then
            local last_line=$(tail -1 "$log_file" 2>/dev/null || echo "Starting...")
            echo "  [$(get_timestamp)] Still running... Last output: ${last_line:0:80}"
        fi

        sleep 5
    done

    # Clean up tmux window if still exists
    tmux kill-window -t "${TMUX_SESSION}:${window_name}" 2>/dev/null || true

    if [ "$completed" = true ]; then
        print_success "Experiment ${run_id} completed successfully!"
        print_info "End time: $(get_timestamp)"

        # Extract final metrics from log
        if grep -q "Final Results" "$log_file"; then
            echo ""
            echo "  Final Results:"
            grep -A 10 "Final Results" "$log_file" | head -15 | sed 's/^/  /'
        fi
        return 0
    elif [ "$error_occurred" = true ]; then
        print_error "Experiment ${run_id} failed with error!"
        print_info "Check log file: ${log_file}"
        echo ""
        echo "  Last 20 lines of log:"
        tail -20 "$log_file" | sed 's/^/  /'
        return 1
    else
        print_error "Experiment ${run_id} ended unexpectedly!"
        return 1
    fi
}

# =============================================================================
# Main Script
# =============================================================================

print_header "Multimodal Experiments Runner"
echo "GPU ID: ${GPU_ID}"
echo "Experiment Group: ${EXPERIMENT_GROUP}"
echo "Seeds: ${SEEDS[*]}"
echo "Conda Environment: ${CONDA_ENV}"
echo "Working Directory: ${WORK_DIR}"
echo "Log Directory: ${LOG_DIR}"

# Pre-flight checks
print_header "Pre-flight Checks"
check_tmux
check_conda_env
check_gpu

# Select experiments based on group
case "$EXPERIMENT_GROUP" in
    "all")
        EXPERIMENTS=("${CITY_LEVEL_EXPERIMENTS[@]}" "${PATCH_LEVEL_EXPERIMENTS[@]}")
        ;;
    "city")
        EXPERIMENTS=("${CITY_LEVEL_EXPERIMENTS[@]}")
        ;;
    "patch")
        EXPERIMENTS=("${PATCH_LEVEL_EXPERIMENTS[@]}")
        ;;
    "cnn")
        EXPERIMENTS=("${CNN_EXPERIMENTS[@]}")
        ;;
    "resnet")
        EXPERIMENTS=("${RESNET_EXPERIMENTS[@]}")
        ;;
    "single")
        if [ -z "$SINGLE_EXPERIMENT" ]; then
            print_error "Single experiment mode requires -e flag with experiment name"
            exit 1
        fi
        EXPERIMENTS=("$SINGLE_EXPERIMENT")
        ;;
    *)
        print_error "Unknown experiment group: ${EXPERIMENT_GROUP}"
        echo "Available groups: all, city, patch, cnn, resnet, single"
        exit 1
        ;;
esac

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#EXPERIMENTS[@]} * ${#SEEDS[@]}))
print_info "Total experiments to run: ${TOTAL_EXPERIMENTS}"
print_info "Experiments: ${EXPERIMENTS[*]}"

# Confirm before starting
echo ""
read -p "Press Enter to start experiments, or Ctrl+C to cancel..."

# Create summary log
SUMMARY_LOG="${LOG_DIR}/summary_$(date +%Y%m%d_%H%M%S).log"
echo "Multimodal Experiments Summary" > "$SUMMARY_LOG"
echo "Started: $(get_timestamp)" >> "$SUMMARY_LOG"
echo "GPU: ${GPU_ID}" >> "$SUMMARY_LOG"
echo "Experiments: ${EXPERIMENTS[*]}" >> "$SUMMARY_LOG"
echo "Seeds: ${SEEDS[*]}" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Run experiments
CURRENT=0
FAILED_EXPERIMENTS=()
SUCCESSFUL_EXPERIMENTS=()

START_TIME=$(date +%s)

for exp_name in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((CURRENT++))
        run_id="${exp_name}_seed${seed}"

        print_header "Progress: ${CURRENT}/${TOTAL_EXPERIMENTS}"
        print_info "Experiment: ${exp_name}"
        print_info "Seed: ${seed}"

        if run_experiment "$exp_name" "$seed"; then
            SUCCESSFUL_EXPERIMENTS+=("$run_id")
            echo "[SUCCESS] ${run_id}" >> "$SUMMARY_LOG"
        else
            FAILED_EXPERIMENTS+=("$run_id")
            echo "[FAILED] ${run_id}" >> "$SUMMARY_LOG"

            # Ask user whether to continue
            echo ""
            read -p "Experiment failed. Continue with remaining experiments? (y/n): " choice
            if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
                print_info "Stopping experiments."
                break 2
            fi
        fi

        # Short delay between experiments
        sleep 5
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

# Final summary
print_header "Experiments Completed"
echo ""
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo "Successful: ${#SUCCESSFUL_EXPERIMENTS[@]}/${TOTAL_EXPERIMENTS}"
echo "Failed: ${#FAILED_EXPERIMENTS[@]}/${TOTAL_EXPERIMENTS}"
echo ""

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo "Failed experiments:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
fi

echo ""
echo "Summary log: ${SUMMARY_LOG}"
echo "Individual logs: ${LOG_DIR}/"

# Write final summary
echo "" >> "$SUMMARY_LOG"
echo "Completed: $(get_timestamp)" >> "$SUMMARY_LOG"
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m" >> "$SUMMARY_LOG"
echo "Successful: ${#SUCCESSFUL_EXPERIMENTS[@]}/${TOTAL_EXPERIMENTS}" >> "$SUMMARY_LOG"
echo "Failed: ${#FAILED_EXPERIMENTS[@]}/${TOTAL_EXPERIMENTS}" >> "$SUMMARY_LOG"

print_success "All done!"
