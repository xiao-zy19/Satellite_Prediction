#!/bin/bash
# SFT Training Script

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="mllm-satellite-sft"

# Paths
DATA_DIR="/path/to/npy_cache"  # Update this
LABELS_FILE="/path/to/population_data.xlsx"  # Update this
PRETRAIN_CKPT="checkpoints/pretrain/final.pt"
OUTPUT_DIR="checkpoints/sft"
CONFIG="configs/sft_config.yaml"

# Training parameters
BATCH_SIZE=4
NUM_EPOCHS=20
LR=1e-5

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
deepspeed --num_gpus=8 sft.py \
    --config $CONFIG \
    --data-dir $DATA_DIR \
    --labels-file $LABELS_FILE \
    --pretrain-checkpoint $PRETRAIN_CKPT \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $NUM_EPOCHS \
    --learning-rate $LR \
    --use-wandb

echo "SFT training complete!"
