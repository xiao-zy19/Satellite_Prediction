#!/bin/bash
# ============================================================================
# 实验运行脚本 - tmux窗口管理版
# ============================================================================

set -e

# 项目配置
PROJECT_DIR="/mnt/bn/code-generation-100t-ckpt/xzy/gulab/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/experiment_runs"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status"
SESSION_NAME="satellite_exp"

# Conda 环境
CONDA_BASE="/mnt/bn/code-generation-100t-ckpt/xzy/miniconda3"
CONDA_ENV="xzy"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验配置 (与 config.py 中的 EXPERIMENTS 完全对应)
# ============================================================================

# --- City-level Baseline Models (无预训练) ---
MLP_BASELINE_EXPS=("mlp_baseline" "mlp_median" "mlp_trimmed_mean")
CNN_BASELINE_EXPS=("light_cnn_baseline" "light_cnn_median" "light_cnn_trimmed_mean")

# --- Self-supervised Pretraining ---
SIMCLR_MLP_EXPS=("simclr_mlp" "simclr_mlp_median" "simclr_mlp_trimmed_mean")
SIMCLR_CNN_EXPS=("simclr_cnn" "simclr_cnn_median" "simclr_cnn_trimmed_mean")
MAE_CNN_EXPS=("mae_cnn" "mae_cnn_median" "mae_cnn_trimmed_mean")

# --- Patch-level Training ---
PATCH_EXPS=("mlp_patch_level" "light_cnn_patch_level" "simclr_cnn_patch_level" "mae_cnn_patch_level")

# --- ResNet10 ---
RESNET10_EXPS=("resnet10_baseline" "resnet10_median" "resnet10_trimmed_mean" "resnet10_patch_level")

# --- ResNet18 ---
RESNET18_BASELINE_EXPS=("resnet18_baseline" "resnet18_median" "resnet18_trimmed_mean")
RESNET18_IMAGENET_EXPS=("resnet18_imagenet" "resnet18_imagenet_median" "resnet18_imagenet_trimmed_mean")
RESNET18_PATCH_EXPS=("resnet18_patch_level" "resnet18_imagenet_patch_level")

# --- ResNet34 ---
RESNET34_BASELINE_EXPS=("resnet34_baseline" "resnet34_median" "resnet34_trimmed_mean")
RESNET34_IMAGENET_EXPS=("resnet34_imagenet" "resnet34_imagenet_median" "resnet34_imagenet_trimmed_mean")
RESNET34_PATCH_EXPS=("resnet34_patch_level" "resnet34_imagenet_patch_level")

# --- ResNet50 ---
RESNET50_BASELINE_EXPS=("resnet50_baseline" "resnet50_median" "resnet50_trimmed_mean")
RESNET50_IMAGENET_EXPS=("resnet50_imagenet" "resnet50_imagenet_median" "resnet50_imagenet_trimmed_mean")
RESNET50_PATCH_EXPS=("resnet50_patch_level" "resnet50_imagenet_patch_level")

# --- ResNet101 ---
RESNET101_BASELINE_EXPS=("resnet101_baseline" "resnet101_median" "resnet101_trimmed_mean")
RESNET101_IMAGENET_EXPS=("resnet101_imagenet" "resnet101_imagenet_median" "resnet101_imagenet_trimmed_mean")
RESNET101_PATCH_EXPS=("resnet101_patch_level" "resnet101_imagenet_patch_level")

# --- Position-Aware Aggregation (MLP) ---
MLP_AGG_EXPS=("mlp_attention" "mlp_pos_attention" "mlp_spatial_attention" "mlp_transformer" "mlp_transformer_2d")

# --- Position-Aware Aggregation (LightCNN) ---
CNN_AGG_EXPS=("light_cnn_attention" "light_cnn_pos_attention" "light_cnn_spatial_attention" "light_cnn_transformer" "light_cnn_transformer_2d")

# --- Position-Aware Aggregation (ResNet18) ---
RESNET18_AGG_EXPS=("resnet18_attention" "resnet18_pos_attention" "resnet18_spatial_attention" "resnet18_transformer" "resnet18_transformer_2d")

# --- SimCLR + Position-Aware Aggregation ---
SIMCLR_AGG_EXPS=("simclr_cnn_attention" "simclr_cnn_pos_attention" "simclr_cnn_spatial_attention" "simclr_cnn_transformer" "simclr_cnn_transformer_2d")

# --- MAE + Position-Aware Aggregation ---
MAE_AGG_EXPS=("mae_cnn_attention" "mae_cnn_pos_attention" "mae_cnn_spatial_attention" "mae_cnn_transformer" "mae_cnn_transformer_2d")

# ============================================================================
# 组合分类 (方便按类别运行)
# ============================================================================

# 基础实验 (无预训练)
BASELINE_EXPS=("${MLP_BASELINE_EXPS[@]}" "${CNN_BASELINE_EXPS[@]}")

# 自监督预训练实验
SSL_EXPS=("${SIMCLR_MLP_EXPS[@]}" "${SIMCLR_CNN_EXPS[@]}" "${MAE_CNN_EXPS[@]}")

# 所有 ResNet 实验
RESNET_ALL_EXPS=(
    "${RESNET10_EXPS[@]}"
    "${RESNET18_BASELINE_EXPS[@]}" "${RESNET18_IMAGENET_EXPS[@]}" "${RESNET18_PATCH_EXPS[@]}"
    "${RESNET34_BASELINE_EXPS[@]}" "${RESNET34_IMAGENET_EXPS[@]}" "${RESNET34_PATCH_EXPS[@]}"
    "${RESNET50_BASELINE_EXPS[@]}" "${RESNET50_IMAGENET_EXPS[@]}" "${RESNET50_PATCH_EXPS[@]}"
    "${RESNET101_BASELINE_EXPS[@]}" "${RESNET101_IMAGENET_EXPS[@]}" "${RESNET101_PATCH_EXPS[@]}"
)

# 所有 Position-Aware Aggregation 实验
AGG_EXPS=("${MLP_AGG_EXPS[@]}" "${CNN_AGG_EXPS[@]}" "${RESNET18_AGG_EXPS[@]}" "${SIMCLR_AGG_EXPS[@]}" "${MAE_AGG_EXPS[@]}")

# 所有实验 (共80个)
ALL_EXPERIMENTS=(
    "${BASELINE_EXPS[@]}"
    "${SSL_EXPS[@]}"
    "${PATCH_EXPS[@]}"
    "${RESNET_ALL_EXPS[@]}"
    "${AGG_EXPS[@]}"
)

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEED=42

# ============================================================================
# 帮助
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h        显示帮助"
    echo "  --list, -l        列出所有实验"
    echo "  --gpus GPUS       指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N      并行数量"
    echo "  --category CAT    实验类别 (见下方)"
    echo "  --exp EXP1,EXP2   指定实验名称"
    echo "  --seed SEED       随机种子 (默认: $DEFAULT_SEED)"
    echo "  --seeds S1,S2,S3  多个种子运行 (例如: 42,123,456)"
    echo "  --dry-run         预览模式"
    echo "  --resume          跳过已完成实验"
    echo ""
    echo "可用类别 (--category):"
    echo "  all               所有实验 (共 ${#ALL_EXPERIMENTS[@]} 个)"
    echo "  baseline          基础实验 (MLP/CNN无预训练, 共 ${#BASELINE_EXPS[@]} 个)"
    echo "  ssl               自监督预训练 (SimCLR/MAE, 共 ${#SSL_EXPS[@]} 个)"
    echo "  patch             Patch-level训练 (共 ${#PATCH_EXPS[@]} 个)"
    echo "  resnet            所有ResNet实验 (共 ${#RESNET_ALL_EXPS[@]} 个)"
    echo "  resnet10          ResNet10实验 (共 ${#RESNET10_EXPS[@]} 个)"
    echo "  resnet18          ResNet18实验 (共 $((${#RESNET18_BASELINE_EXPS[@]} + ${#RESNET18_IMAGENET_EXPS[@]} + ${#RESNET18_PATCH_EXPS[@]})) 个)"
    echo "  resnet34          ResNet34实验 (共 $((${#RESNET34_BASELINE_EXPS[@]} + ${#RESNET34_IMAGENET_EXPS[@]} + ${#RESNET34_PATCH_EXPS[@]})) 个)"
    echo "  resnet50          ResNet50实验 (共 $((${#RESNET50_BASELINE_EXPS[@]} + ${#RESNET50_IMAGENET_EXPS[@]} + ${#RESNET50_PATCH_EXPS[@]})) 个)"
    echo "  resnet101         ResNet101实验 (共 $((${#RESNET101_BASELINE_EXPS[@]} + ${#RESNET101_IMAGENET_EXPS[@]} + ${#RESNET101_PATCH_EXPS[@]})) 个)"
    echo "  agg               Position-Aware Aggregation实验 (共 ${#AGG_EXPS[@]} 个)"
    echo "  mlp_agg           MLP + Aggregation (共 ${#MLP_AGG_EXPS[@]} 个)"
    echo "  cnn_agg           LightCNN + Aggregation (共 ${#CNN_AGG_EXPS[@]} 个)"
    echo "  simclr_agg        SimCLR + Aggregation (共 ${#SIMCLR_AGG_EXPS[@]} 个)"
    echo "  mae_agg           MAE + Aggregation (共 ${#MAE_AGG_EXPS[@]} 个)"
}

list_experiments() {
    echo "所有实验 (共 ${#ALL_EXPERIMENTS[@]} 个):"
    echo ""
    echo "=== Baseline (${#BASELINE_EXPS[@]}个) ==="
    for exp in "${BASELINE_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== Self-supervised Pretraining (${#SSL_EXPS[@]}个) ==="
    for exp in "${SSL_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== Patch-level (${#PATCH_EXPS[@]}个) ==="
    for exp in "${PATCH_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== ResNet (${#RESNET_ALL_EXPS[@]}个) ==="
    for exp in "${RESNET_ALL_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== Position-Aware Aggregation (${#AGG_EXPS[@]}个) ==="
    for exp in "${AGG_EXPS[@]}"; do echo "  - $exp"; done
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
CATEGORY="all"
CUSTOM_EXPS=""
DRY_RUN=false
RESUME=false
SEED=$DEFAULT_SEED
SEEDS=()  # 多个种子运行

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --category) CATEGORY=$2; shift 2 ;;
        --exp) CUSTOM_EXPS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

# 如果指定了 --seeds，使用多种子模式；否则使用单个 seed
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=("$SEED")
fi

# 设置并行数
if [ "$PARALLEL_COUNT" -eq 0 ]; then
    PARALLEL_COUNT=${#GPUS[@]}
fi
if [ "$PARALLEL_COUNT" -gt "${#GPUS[@]}" ]; then
    PARALLEL_COUNT=${#GPUS[@]}
fi

# ============================================================================
# 选择实验
# ============================================================================
if [ -n "$CUSTOM_EXPS" ]; then
    IFS=',' read -ra SELECTED_EXPS <<< "$CUSTOM_EXPS"
else
    case $CATEGORY in
        # 基础分类
        baseline) SELECTED_EXPS=("${BASELINE_EXPS[@]}") ;;
        ssl) SELECTED_EXPS=("${SSL_EXPS[@]}") ;;
        patch) SELECTED_EXPS=("${PATCH_EXPS[@]}") ;;

        # ResNet 分类
        resnet) SELECTED_EXPS=("${RESNET_ALL_EXPS[@]}") ;;
        resnet10) SELECTED_EXPS=("${RESNET10_EXPS[@]}") ;;
        resnet18) SELECTED_EXPS=("${RESNET18_BASELINE_EXPS[@]}" "${RESNET18_IMAGENET_EXPS[@]}" "${RESNET18_PATCH_EXPS[@]}") ;;
        resnet34) SELECTED_EXPS=("${RESNET34_BASELINE_EXPS[@]}" "${RESNET34_IMAGENET_EXPS[@]}" "${RESNET34_PATCH_EXPS[@]}") ;;
        resnet50) SELECTED_EXPS=("${RESNET50_BASELINE_EXPS[@]}" "${RESNET50_IMAGENET_EXPS[@]}" "${RESNET50_PATCH_EXPS[@]}") ;;
        resnet101) SELECTED_EXPS=("${RESNET101_BASELINE_EXPS[@]}" "${RESNET101_IMAGENET_EXPS[@]}" "${RESNET101_PATCH_EXPS[@]}") ;;

        # Aggregation 分类
        agg) SELECTED_EXPS=("${AGG_EXPS[@]}") ;;
        mlp_agg) SELECTED_EXPS=("${MLP_AGG_EXPS[@]}") ;;
        cnn_agg) SELECTED_EXPS=("${CNN_AGG_EXPS[@]}") ;;
        resnet18_agg) SELECTED_EXPS=("${RESNET18_AGG_EXPS[@]}") ;;
        simclr_agg) SELECTED_EXPS=("${SIMCLR_AGG_EXPS[@]}") ;;
        mae_agg) SELECTED_EXPS=("${MAE_AGG_EXPS[@]}") ;;

        # 全部
        all) SELECTED_EXPS=("${ALL_EXPERIMENTS[@]}") ;;
        *) echo "未知类别: $CATEGORY"; show_help; exit 1 ;;
    esac
fi

# 生成实验任务列表 (实验名 + 种子)
# 格式: "exp_name:seed"
TASK_LIST=()
for exp in "${SELECTED_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp}:${seed}")
    done
done

# Resume: 过滤已完成的实验
# train.py 的文件命名规则:
#   - seed == 42 (默认): {exp}_results.pkl
#   - seed != 42: {exp}_seed{seed}_results.pkl
if [ "$RESUME" = true ]; then
    FILTERED=()
    for task in "${TASK_LIST[@]}"; do
        exp="${task%%:*}"
        seed="${task##*:}"
        # 根据 seed 确定结果文件名 (与 train.py 逻辑一致)
        if [ "$seed" = "$DEFAULT_SEED" ]; then
            result_file="${RESULT_DIR}/${exp}_results.pkl"
        else
            result_file="${RESULT_DIR}/${exp}_seed${seed}_results.pkl"
        fi
        if [ -f "$result_file" ]; then
            echo "[跳过] $exp (seed=$seed, 已有结果)"
        else
            FILTERED+=("$task")
        fi
    done
    TASK_LIST=("${FILTERED[@]}")
fi

TOTAL=${#TASK_LIST[@]}
NUM_EXPS=${#SELECTED_EXPS[@]}
NUM_SEEDS=${#SEEDS[@]}

echo "=============================================="
echo "  Satellite Prediction 实验配置"
echo "=============================================="
echo "GPU列表: ${GPUS[*]}"
echo "并行数: $PARALLEL_COUNT"
echo "实验数: $NUM_EXPS"
echo "种子列表: ${SEEDS[*]}"
echo "总任务数: $TOTAL (${NUM_EXPS}个实验 x ${NUM_SEEDS}个种子)"
echo "类别: $CATEGORY"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "没有要运行的实验"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "将运行以下实验任务:"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        exp="${task%%:*}"
        seed="${task##*:}"
        echo "  $((i+1)). $exp (seed=$seed)"
    done
    exit 0
fi

# ============================================================================
# 清理状态文件
# ============================================================================
rm -f "${STATUS_DIR}"/gpu_*.lock
rm -f "${STATUS_DIR}"/exp_*.status

# ============================================================================
# GPU 管理函数
# ============================================================================
get_free_gpu() {
    for gpu in "${GPUS[@]}"; do
        local lock_file="${STATUS_DIR}/gpu_${gpu}.lock"
        if [ ! -f "$lock_file" ]; then
            echo "$gpu"
            return 0
        fi
    done
    return 1
}

lock_gpu() {
    local gpu=$1
    local exp=$2
    echo "$exp" > "${STATUS_DIR}/gpu_${gpu}.lock"
}

get_done_count() {
    local count=0
    for f in "${STATUS_DIR}"/exp_*.status; do
        if [ -f "$f" ]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# ============================================================================
# 创建tmux session
# ============================================================================
echo "创建 tmux session: $SESSION_NAME"

# 杀死旧session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# 创建新session
tmux new-session -d -s "$SESSION_NAME" -n "monitor"

# monitor窗口
tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 实验监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总实验数: $TOTAL | 并行数: $PARALLEL_COUNT'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Ctrl+B w 查看窗口 | Ctrl+B d 脱离'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 在tmux中启动实验
# ============================================================================
start_experiment() {
    local exp=$1
    local gpu=$2
    local seed=$3

    # 任务ID (用于状态跟踪)
    local task_id="${exp}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    # tmux窗口名 (简化显示)
    local window_name="${exp}_s${seed}"

    echo "[$(date '+%H:%M:%S')] 启动: $exp (seed=$seed) -> GPU $gpu"

    # 锁定GPU
    lock_gpu "$gpu" "$task_id"

    # 创建新窗口
    tmux new-window -t "$SESSION_NAME" -n "$window_name"

    # 发送命令
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== 实验: $exp | GPU: $gpu | Seed: $seed ==='" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '开始时间:' \$(date)" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" Enter

    # 运行实验 (传递 --seed 参数)
    local cmd="python train.py --exp $exp --gpu $gpu --seed $seed 2>&1 | tee $log_file"
    cmd="$cmd; if [ \$? -eq 0 ]; then echo success > ${STATUS_DIR}/exp_${task_id}.status; else echo failed > ${STATUS_DIR}/exp_${task_id}.status; fi"
    cmd="$cmd; rm -f ${STATUS_DIR}/gpu_${gpu}.lock"
    cmd="$cmd; echo ''; echo '=== 实验结束 ==='; echo '结束时间:' \$(date)"

    tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" Enter
}

# ============================================================================
# 主调度循环
# ============================================================================
echo ""
echo "开始调度实验..."
echo ""

QUEUE_INDEX=0

# 启动初始批次
i=0
while [ $i -lt $PARALLEL_COUNT ] && [ $QUEUE_INDEX -lt $TOTAL ]; do
    task="${TASK_LIST[$QUEUE_INDEX]}"
    exp="${task%%:*}"
    seed="${task##*:}"
    gpu="${GPUS[$i]}"
    start_experiment "$exp" "$gpu" "$seed"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
    i=$((i + 1))
    sleep 1
done

echo ""
echo "已启动 $QUEUE_INDEX 个任务"
echo ""

# 调度剩余实验
while [ $QUEUE_INDEX -lt $TOTAL ]; do
    sleep 5

    FREE_GPU=$(get_free_gpu) || continue

    task="${TASK_LIST[$QUEUE_INDEX]}"
    exp="${task%%:*}"
    seed="${task##*:}"
    start_experiment "$exp" "$FREE_GPU" "$seed"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
done

echo "所有 $TOTAL 个任务已调度"
echo ""

# ============================================================================
# 等待完成
# ============================================================================
echo "等待实验完成... (Ctrl+C 退出脚本，实验继续在tmux中运行)"
echo ""

while true; do
    DONE=$(get_done_count)
    printf "\r进度: $DONE / $TOTAL    "

    if [ "$DONE" -ge "$TOTAL" ]; then
        echo ""
        break
    fi

    sleep 5
done

# ============================================================================
# 汇总
# ============================================================================
echo ""
echo "=============================================="
echo "  所有实验已完成!"
echo "=============================================="

SUCCESS=0
FAILED=0

for exp in "${SELECTED_EXPS[@]}"; do
    status_file="${STATUS_DIR}/exp_${exp}.status"
    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  [失败] $exp"
        fi
    fi
done

echo ""
echo "成功: $SUCCESS | 失败: $FAILED"
echo "日志: $LOG_DIR"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
