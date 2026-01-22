#!/bin/bash
# Multimodal 实验训练队列
# GPU: 2
# Conda: alpha-earth

set -e

# 设置环境
export CUDA_VISIBLE_DEVICES=2
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain

echo "=============================================="
echo "Multimodal Training Queue"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=============================================="

# 高优先级 - 基线实验
echo ""
echo "[1/20] Running mm_cnn_concat..."
python train_multimodal.py --exp mm_cnn_concat --gpu 2

echo ""
echo "[2/20] Running mm_cnn_gated..."
python train_multimodal.py --exp mm_cnn_gated --gpu 2

echo ""
echo "[3/20] Running mm_cnn_film..."
python train_multimodal.py --exp mm_cnn_film --gpu 2

echo ""
echo "[4/20] Running mm_cnn_attention..."
python train_multimodal.py --exp mm_cnn_attention --gpu 2

# 中优先级 - 不同聚合方式
echo ""
echo "[5/20] Running mm_cnn_concat_trimmed..."
python train_multimodal.py --exp mm_cnn_concat_trimmed --gpu 2

echo ""
echo "[6/20] Running mm_cnn_concat_median..."
python train_multimodal.py --exp mm_cnn_concat_median --gpu 2

echo ""
echo "[7/20] Running mm_cnn_gated_trimmed..."
python train_multimodal.py --exp mm_cnn_gated_trimmed --gpu 2

echo ""
echo "[8/20] Running mm_cnn_film_trimmed..."
python train_multimodal.py --exp mm_cnn_film_trimmed --gpu 2

echo ""
echo "[9/20] Running mm_cnn_attention_trimmed..."
python train_multimodal.py --exp mm_cnn_attention_trimmed --gpu 2

# MLP 实验
echo ""
echo "[10/20] Running mm_mlp_concat..."
python train_multimodal.py --exp mm_mlp_concat --gpu 2

echo ""
echo "[11/20] Running mm_mlp_gated..."
python train_multimodal.py --exp mm_mlp_gated --gpu 2

# 自定义实验
echo ""
echo "[12/20] Running mm_cnn_small_concat..."
python train_multimodal.py --exp mm_cnn_small_concat --gpu 2

# ResNet 实验 (较大模型)
echo ""
echo "[13/20] Running mm_resnet18_concat..."
python train_multimodal.py --exp mm_resnet18_concat --gpu 2

echo ""
echo "[14/20] Running mm_resnet18_gated..."
python train_multimodal.py --exp mm_resnet18_gated --gpu 2

echo ""
echo "[15/20] Running mm_resnet18_film..."
python train_multimodal.py --exp mm_resnet18_film --gpu 2

echo ""
echo "[16/20] Running mm_resnet34_pretrained..."
python train_multimodal.py --exp mm_resnet34_pretrained --gpu 2

# Patch-level 实验 (训练时间较长)
echo ""
echo "[17/20] Running mm_cnn_concat_patch..."
python train_multimodal.py --exp mm_cnn_concat_patch --gpu 2

echo ""
echo "[18/20] Running mm_cnn_gated_patch..."
python train_multimodal.py --exp mm_cnn_gated_patch --gpu 2

echo ""
echo "[19/20] Running mm_cnn_film_patch..."
python train_multimodal.py --exp mm_cnn_film_patch --gpu 2

echo ""
echo "[20/20] Running mm_resnet18_concat_patch..."
python train_multimodal.py --exp mm_resnet18_concat_patch --gpu 2

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "End Time: $(date)"
echo "=============================================="
