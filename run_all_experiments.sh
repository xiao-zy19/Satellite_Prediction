#!/bin/bash
# Run all pretrain experiments
# Usage: ./run_all_experiments.sh [GPU_ID]

GPU_ID=${1:-3}  # Default to GPU 3

cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain

echo "=============================================="
echo "Running all pretrain experiments on GPU $GPU_ID"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# List of experiments to run
EXPERIMENTS=(
    "mlp_baseline"
    "light_cnn_baseline"
    "resnet_baseline"
    "resnet_imagenet"
    "simclr_mlp"
    "simclr_cnn"
    "mae_cnn"
)

# Run each experiment
for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Experiment: $exp"
    echo "=============================================="
    echo "Start time: $(date)"

    python train.py --exp "$exp" --gpu "$GPU_ID"

    echo "Completed: $exp at $(date)"
    echo ""
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved in: results/"
echo ""

# Generate summary
python compare_results.py
