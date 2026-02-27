#!/bin/bash
# Run all MoE experiments: 6 configs × 3 seeds = 18 runs
# Base: MAE + LightCNN + Gated + Patch-level

set -e

SEEDS=(42 123 456)
MOE_EXPS=(
    mm_mae_cnn_gated_patch_moe2
    mm_mae_cnn_gated_patch_moe4
    mm_mae_cnn_gated_patch_moe4_gpol
    mm_mae_cnn_gated_patch_moe4_nobal
    mm_mae_cnn_gated_patch_moe6
    mm_mae_cnn_gated_patch_moe2_gpol
)

GPU=${1:-0}  # Default GPU 0, override with first argument

echo "============================================"
echo "MoE Experiment Matrix"
echo "  Configs: ${#MOE_EXPS[@]}"
echo "  Seeds:   ${#SEEDS[@]}"
echo "  Total:   $(( ${#MOE_EXPS[@]} * ${#SEEDS[@]} )) runs"
echo "  GPU:     ${GPU}"
echo "============================================"

cd "$(dirname "$0")/../.."

COUNT=0
TOTAL=$(( ${#MOE_EXPS[@]} * ${#SEEDS[@]} ))

for exp in "${MOE_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL] Running: $exp (seed=$seed)"
        echo "--------------------------------------------"
        python train_multimodal.py --exp "$exp" --gpu "$GPU" --seed "$seed" --normalize_on_gpu
    done
done

echo ""
echo "============================================"
echo "All $TOTAL MoE experiments completed!"
echo "============================================"
