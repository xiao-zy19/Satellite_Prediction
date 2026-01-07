#!/bin/bash
# ============================================================================
# MAE Pretraining Script for 64-channel Satellite Embeddings
#
# 使用方式:
#   bash scripts/run_pretrain.sh                    # 使用默认参数
#   bash scripts/run_pretrain.sh --debug            # 调试模式
#   bash scripts/run_pretrain.sh --multi-gpu        # 多卡训练
# ============================================================================

set -e  # 遇到错误立即退出

# 默认参数
DEBUG_MODE=false
MULTI_GPU=false
USE_WANDB=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# 配置
# ============================================================================

# 数据路径
DATA_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data_local/city_satellite_tiles"
OUTPUT_DIR="checkpoints/pretrain"
CONFIG="configs/pretrain_config.yaml"

# 训练参数
BATCH_SIZE=8
NUM_EPOCHS=10
LEARNING_RATE=1e-4
SAMPLES_PER_FILE=50

# 调试模式参数
if [ "$DEBUG_MODE" = true ]; then
    echo "=========================================="
    echo "Running in DEBUG mode"
    echo "=========================================="
    BATCH_SIZE=2
    NUM_EPOCHS=2
    SAMPLES_PER_FILE=5
    DEBUG_FLAG="--debug"
else
    DEBUG_FLAG=""
fi

# Wandb
if [ "$USE_WANDB" = true ]; then
    WANDB_FLAG="--use-wandb"
    export WANDB_PROJECT="mllm-satellite-pretrain"
else
    WANDB_FLAG=""
fi

# ============================================================================
# 环境检查
# ============================================================================

echo "=========================================="
echo "MAE Pretraining for 64-channel Satellite Data"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Samples per file: $SAMPLES_PER_FILE"
echo "  Debug mode: $DEBUG_MODE"
echo "  Multi-GPU: $MULTI_GPU"
echo "  Wandb: $USE_WANDB"
echo ""

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# 统计数据文件
NUM_FILES=$(find "$DATA_DIR" -name "*.tiff" -o -name "*.tif" | wc -l)
echo "Found $NUM_FILES data files"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 切换到项目目录
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/MLLM_Training

# ============================================================================
# 运行训练
# ============================================================================

if [ "$MULTI_GPU" = true ]; then
    # 多卡训练 (使用DeepSpeed)
    echo "Starting multi-GPU training with DeepSpeed..."

    # 设置GPU
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    NUM_GPUS=8

    deepspeed --num_gpus=$NUM_GPUS pretrain.py \
        --config $CONFIG \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $NUM_EPOCHS \
        --learning-rate $LEARNING_RATE \
        --samples-per-file $SAMPLES_PER_FILE \
        $WANDB_FLAG \
        $DEBUG_FLAG
else
    # 单卡训练
    echo "Starting single-GPU training..."

    # 设置GPU (默认使用GPU 0，可以通过环境变量修改)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"

    python pretrain.py \
        --config $CONFIG \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $NUM_EPOCHS \
        --learning-rate $LEARNING_RATE \
        --samples-per-file $SAMPLES_PER_FILE \
        $WANDB_FLAG \
        $DEBUG_FLAG
fi

echo ""
echo "=========================================="
echo "Pretraining complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
