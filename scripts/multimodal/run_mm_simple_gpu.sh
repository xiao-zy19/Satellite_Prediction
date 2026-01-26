#!/bin/bash
# ============================================================================
# Multimodal 实验运行脚本 - tmux窗口管理版 (GPU归一化版本)
# 基于 run_mm_simple.sh，添加 --normalize_on_gpu 参数以提升训练速度
# ============================================================================

set -e

# 项目配置
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/multimodal_runs"
RESULT_DIR="${PROJECT_DIR}/results/multimodal_runs"
STATUS_DIR="${PROJECT_DIR}/.run_status_mm"
SESSION_NAME="mm_exp_gpu"

# Conda 环境
CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验配置 (与 config_multimodal.py 中的 MULTIMODAL_EXPERIMENTS 对应)
# ============================================================================

# --- City-level Baseline (LightCNN + Concat) ---
CNN_CONCAT_EXPS=("mm_cnn_concat" "mm_cnn_concat_median" "mm_cnn_concat_trimmed")

# --- Fusion Variants (LightCNN) ---
CNN_FUSION_EXPS=("mm_cnn_gated" "mm_cnn_gated_trimmed" "mm_cnn_attention" "mm_cnn_attention_trimmed" "mm_cnn_film" "mm_cnn_film_trimmed")

# --- MLP Models ---
MLP_EXPS=("mm_mlp_concat" "mm_mlp_gated")

# --- ResNet Models ---
RESNET_EXPS=("mm_resnet18_concat" "mm_resnet18_gated" "mm_resnet18_film" "mm_resnet34_pretrained")

# --- Patch-level Experiments ---
PATCH_EXPS=("mm_cnn_concat_patch" "mm_cnn_gated_patch" "mm_cnn_film_patch" "mm_resnet18_concat_patch")

# --- Custom/Other ---
CUSTOM_MODEL_EXPS=("mm_cnn_small_concat")

# ============================================================================
# 组合分类
# ============================================================================

# 基础对照组
BASELINE_EXPS=("${CNN_CONCAT_EXPS[@]}")

# 所有融合变体 (LightCNN)
ALL_FUSION_EXPS=("${CNN_CONCAT_EXPS[@]}" "${CNN_FUSION_EXPS[@]}")

# 所有模型架构 (MLP, CNN, ResNet)
ALL_MODELS_EXPS=("${CNN_CONCAT_EXPS[@]}" "${MLP_EXPS[@]}" "${RESNET_EXPS[@]}")

# 所有 Patch-level
ALL_PATCH_EXPS=("${PATCH_EXPS[@]}")

# 所有实验
ALL_EXPERIMENTS=(
    "${CNN_CONCAT_EXPS[@]}"
    "${CNN_FUSION_EXPS[@]}"
    "${MLP_EXPS[@]}"
    "${RESNET_EXPS[@]}"
    "${PATCH_EXPS[@]}"
    "${CUSTOM_MODEL_EXPS[@]}"
)

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
# 默认配置 42, 123, 456 三种种子
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 帮助
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** GPU归一化版本: 使用 --normalize_on_gpu 加速训练 **"
    echo ""
    echo "选项:"
    echo "  --help, -h        显示帮助"
    echo "  --list, -l        列出所有实验"
    echo "  --gpus GPUS       指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N      并行数量"
    echo "  --category CAT    实验类别 (见下方)"
    echo "  --exp EXP1,EXP2   指定实验名称"
    echo "  --seed SEED       覆盖默认种子列表，只运行单个种子"
    echo "  --seeds S1,S2     覆盖默认种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --dry-run         预览模式"
    echo "  --resume          跳过已完成实验"
    echo ""
    echo "可用类别 (--category):"
    echo "  all               所有实验 (共 ${#ALL_EXPERIMENTS[@]} 个)"
    echo "  baseline          基础实验 (LightCNN Concat, 共 ${#BASELINE_EXPS[@]} 个)"
    echo "  fusion            LightCNN 融合变体 (共 ${#ALL_FUSION_EXPS[@]} 个)"
    echo "  models            不同架构 (MLP/CNN/ResNet, 共 ${#ALL_MODELS_EXPS[@]} 个)"
    echo "  patch             Patch-level 训练 (共 ${#ALL_PATCH_EXPS[@]} 个)"
    echo "  mlp               MLP 模型 (共 ${#MLP_EXPS[@]} 个)"
    echo "  resnet            ResNet 模型 (共 ${#RESNET_EXPS[@]} 个)"
}

list_experiments() {
    echo "所有实验 (共 ${#ALL_EXPERIMENTS[@]} 个):"
    echo ""
    echo "=== Baseline (LightCNN + Concat) ==="
    for exp in "${CNN_CONCAT_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== Fusion Variants (LightCNN) ==="
    for exp in "${CNN_FUSION_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== MLP Models ==="
    for exp in "${MLP_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== ResNet Models ==="
    for exp in "${RESNET_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== Patch-level Experiments ==="
    for exp in "${PATCH_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== Custom/Other ==="
    for exp in "${CUSTOM_MODEL_EXPS[@]}"; do echo "  - $exp"; done
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
CATEGORY="all"
CUSTOM_EXPS_STR=""
DRY_RUN=false
RESUME=false
SEEDS=("${DEFAULT_SEEDS[@]}")  # 默认使用 42, 123, 456

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --category) CATEGORY=$2; shift 2 ;;
        --exp) CUSTOM_EXPS_STR=$2; shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;  # 单个种子覆盖列表
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;; # 多个种子覆盖列表
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

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
if [ -n "$CUSTOM_EXPS_STR" ]; then
    IFS=',' read -ra SELECTED_EXPS <<< "$CUSTOM_EXPS_STR"
else
    case $CATEGORY in
        baseline) SELECTED_EXPS=("${BASELINE_EXPS[@]}") ;;
        fusion) SELECTED_EXPS=("${ALL_FUSION_EXPS[@]}") ;;
        models) SELECTED_EXPS=("${ALL_MODELS_EXPS[@]}") ;;
        patch) SELECTED_EXPS=("${ALL_PATCH_EXPS[@]}") ;;
        mlp) SELECTED_EXPS=("${MLP_EXPS[@]}") ;;
        resnet) SELECTED_EXPS=("${RESNET_EXPS[@]}") ;;
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
# train_multimodal.py 的结果命名: {exp}_seed{seed}_results.pkl (注意: 即使 seed=42 也会带后缀，除非代码特殊处理)
# 根据 train_multimodal.py 代码:
# if seed != config.RANDOM_SEED: run_id = f"{exp_name}_seed{seed}" else: run_id = exp_name
# config.RANDOM_SEED 默认为 42
if [ "$RESUME" = true ]; then
    FILTERED=()
    for task in "${TASK_LIST[@]}"; do
        exp="${task%%:*}"
        seed="${task##*:}"

        # 构建 run_id
        if [ "$seed" == "42" ]; then
            run_id="${exp}"
        else
            run_id="${exp}_seed${seed}"
        fi

        result_file="${RESULT_DIR}/${run_id}_results.pkl"

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
echo "  Multimodal Experiment Runner (GPU Normalize)"
echo "=============================================="
echo "GPU列表: ${GPUS[*]}"
echo "并行数: $PARALLEL_COUNT"
echo "实验数: $NUM_EXPS"
echo "种子列表: ${SEEDS[*]}"
echo "总任务数: $TOTAL (${NUM_EXPS}个实验 x ${NUM_SEEDS}个种子)"
echo "类别: $CATEGORY"
echo "结果目录: $RESULT_DIR"
echo ""
echo "** 使用 --normalize_on_gpu 加速训练 **"
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
        echo "  $((i+1)). $exp (seed=$seed) [--normalize_on_gpu]"
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
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== Multimodal 实验监控 (GPU Normalize) ==='" Enter
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

    # 任务ID
    local task_id="${exp}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    # 构建 run_id (与 train_multimodal.py 逻辑一致用于文件名，但这里log用全名)
    if [ "$seed" == "42" ]; then
        run_id="${exp}"
    else
        run_id="${exp}_seed${seed}"
    fi

    # tmux窗口名
    local window_name="${exp}_s${seed}"

    echo "[$(date '+%H:%M:%S')] 启动: $exp (seed=$seed) -> GPU $gpu [--normalize_on_gpu]"

    # 锁定GPU
    lock_gpu "$gpu" "$task_id"

    # 创建新窗口
    tmux new-window -t "$SESSION_NAME" -n "$window_name"

    # 发送命令
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    # 直接设置 PATH 激活环境 (绕过 conda 的旧路径问题)
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH && echo '环境已激活: '${CONDA_ENV}" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== 实验: $exp | GPU: $gpu | Seed: $seed | GPU Normalize: ON ==='" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '开始时间:' \$(date)" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" Enter

    # 运行实验 (使用 train_multimodal.py，添加 --normalize_on_gpu)
    local cmd="python train_multimodal.py --exp $exp --gpu $gpu --seed $seed --normalize_on_gpu 2>&1 | tee $log_file"
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
echo "  所有实验已完成! (GPU Normalize版本)"
echo "=============================================="

SUCCESS=0
FAILED=0

for task in "${TASK_LIST[@]}"; do
    # 任务ID重构
    exp="${task%%:*}"
    seed="${task##*:}"
    task_id="${exp}_seed${seed}"

    status_file="${STATUS_DIR}/exp_${task_id}.status"
    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  [失败] $exp (seed=$seed)"
        fi
    else
        echo "  [未知] $exp (seed=$seed)"
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
