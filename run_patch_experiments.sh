#!/bin/bash
# Run all patch-level experiments on GPU 3

export CUDA_VISIBLE_DEVICES=3
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain

echo "=========================================="
echo "Starting Patch-Level Experiments on GPU 3"
echo "Time: $(date)"
echo "=========================================="

# Experiment 1: mlp_patch_level
echo ""
echo "[1/4] Starting mlp_patch_level at $(date)"
python train.py --exp mlp_patch_level
echo "[1/4] Finished mlp_patch_level at $(date)"

# Experiment 2: light_cnn_patch_level
echo ""
echo "[2/4] Starting light_cnn_patch_level at $(date)"
python train.py --exp light_cnn_patch_level
echo "[2/4] Finished light_cnn_patch_level at $(date)"

# Experiment 3: resnet_patch_level
echo ""
echo "[3/4] Starting resnet_patch_level at $(date)"
python train.py --exp resnet_patch_level
echo "[3/4] Finished resnet_patch_level at $(date)"

# Experiment 4: simclr_cnn_patch_level
echo ""
echo "[4/4] Starting simclr_cnn_patch_level at $(date)"
python train.py --exp simclr_cnn_patch_level
echo "[4/4] Finished simclr_cnn_patch_level at $(date)"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Time: $(date)"
echo "=========================================="
