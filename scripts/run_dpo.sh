#!/bin/bash
# DPO Training Script

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="mllm-satellite-rl"

# Paths
DATA_DIR="/path/to/npy_cache"  # Update this
PREFERENCE_FILE="data/preferences.json"
SFT_CKPT="checkpoints/sft/best.pt"
OUTPUT_DIR="checkpoints/rl"
CONFIG="configs/rl_config.yaml"

# Training parameters
BATCH_SIZE=2
NUM_EPOCHS=5
LR=1e-6
BETA=0.1

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
deepspeed --num_gpus=8 rl.py \
    --config $CONFIG \
    --data-dir $DATA_DIR \
    --preference-file $PREFERENCE_FILE \
    --sft-checkpoint $SFT_CKPT \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --learning-rate $LR \
    --beta $BETA \
    --use-wandb

echo "DPO training complete!"
