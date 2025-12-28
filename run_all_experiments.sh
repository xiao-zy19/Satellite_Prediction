#!/bin/bash
# Run all 11 experiments on GPU 3 with tmux panes for real-time monitoring
# Each batch runs 3 experiments in parallel, displayed in 3 panes

export CUDA_VISIBLE_DEVICES=3
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate alphaearth 2>/dev/null

mkdir -p logs/experiment_runs

# All experiments
BATCH1=("mlp_baseline" "light_cnn_baseline" "mlp_patch_level")
BATCH2=("resnet_baseline" "resnet_imagenet" "resnet_patch_level")
BATCH3=("simclr_mlp" "simclr_cnn" "simclr_cnn_patch_level")
BATCH4=("mae_cnn" "light_cnn_patch_level")

run_batch() {
    local batch_name=$1
    shift
    local exps=("$@")
    local num=${#exps[@]}

    echo "=========================================="
    echo "$batch_name - $(date)"
    echo "=========================================="

    # Run experiments in parallel, output to both terminal and log
    local pids=()
    for exp in "${exps[@]}"; do
        echo ""
        echo ">>> Starting: $exp"
        (python train.py --exp "$exp" 2>&1 | tee "logs/experiment_runs/${exp}.log") &
        pids+=($!)
    done

    # Wait for all to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo ""
    echo "$batch_name completed at $(date)"
    echo ""
}

echo "=========================================="
echo "Starting All 11 Experiments on GPU 3"
echo "Time: $(date)"
echo "=========================================="

run_batch "Batch 1/4" "${BATCH1[@]}"
run_batch "Batch 2/4" "${BATCH2[@]}"
run_batch "Batch 3/4" "${BATCH3[@]}"
run_batch "Batch 4/4" "${BATCH4[@]}"

echo ""
echo "=========================================="
echo "All 11 experiments completed!"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -lh results/*.pkl 2>/dev/null
