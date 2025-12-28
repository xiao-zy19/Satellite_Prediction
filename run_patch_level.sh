#!/bin/bash
# Run patch-level experiments on GPU 3

export CUDA_VISIBLE_DEVICES=3
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alphaearth

echo "=========================================="
echo "Running Patch-Level Experiments on GPU 3"
echo "=========================================="

# Run experiments serially
experiments=(
    "mlp_patch_level"
    "light_cnn_patch_level"
    "resnet_patch_level"
    "simclr_cnn_patch_level"
)

for exp in "${experiments[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting experiment: $exp"
    echo "Time: $(date)"
    echo "=========================================="

    python train.py --exp "$exp" 2>&1 | tee "logs/experiment_runs/${exp}.log"

    echo ""
    echo "Finished: $exp at $(date)"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All patch-level experiments completed!"
echo "Time: $(date)"
echo "=========================================="
