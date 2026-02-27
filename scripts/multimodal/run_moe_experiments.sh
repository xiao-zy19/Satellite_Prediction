#!/bin/bash
# ============================================================================
# MoE 实验运行脚本 - tmux窗口管理版 (GPU归一化版本)
# 基于 run_mm_simple_gpu.sh 架构，专门运行 Mixture of Experts 实验
# ============================================================================

set -e

# 项目配置
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/moe_runs"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_moe"
SESSION_NAME="moe_exp"

# Conda 环境
CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$STATUS_DIR"

# ============================================================================
# MoE 实验配置 (与 config_multimodal.py 对应)
# 全部基于 MAE + LightCNN + Gated + Patch-level
# ============================================================================

# --- K=2 (最小 MoE) ---
MOE_K2_EXPS=("mm_mae_cnn_gated_patch_moe2" "mm_mae_cnn_gated_patch_moe2_gpol")

# --- K=4 (标准 MoE) ---
MOE_K4_EXPS=("mm_mae_cnn_gated_patch_moe4" "mm_mae_cnn_gated_patch_moe4_gpol" "mm_mae_cnn_gated_patch_moe4_nobal")

# --- K=6 (更多专家) ---
MOE_K6_EXPS=("mm_mae_cnn_gated_patch_moe6")

# 组合分类
MOE_GATE_FUSED_EXPS=("mm_mae_cnn_gated_patch_moe2" "mm_mae_cnn_gated_patch_moe4" "mm_mae_cnn_gated_patch_moe6")
MOE_GATE_POLICY_EXPS=("mm_mae_cnn_gated_patch_moe2_gpol" "mm_mae_cnn_gated_patch_moe4_gpol")
MOE_ABLATION_EXPS=("mm_mae_cnn_gated_patch_moe4" "mm_mae_cnn_gated_patch_moe4_gpol" "mm_mae_cnn_gated_patch_moe4_nobal")

ALL_MOE_EXPS=(
    "${MOE_K2_EXPS[@]}"
    "${MOE_K4_EXPS[@]}"
    "${MOE_K6_EXPS[@]}"
)

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 帮助
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** MoE 实验运行脚本 (GPU归一化版本) **"
    echo "** 全部基于 MAE + LightCNN + Gated + Patch-level **"
    echo ""
    echo "选项:"
    echo "  --help, -h        显示帮助"
    echo "  --list, -l        列出所有 MoE 实验"
    echo "  --gpus GPUS       指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N      并行数量 (默认=GPU个数)"
    echo "  --category CAT    实验类别 (见下方)"
    echo "  --exp EXP1,EXP2   指定实验名称"
    echo "  --seed SEED       覆盖默认种子列表，只运行单个种子"
    echo "  --seeds S1,S2     覆盖默认种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --dry-run         预览模式"
    echo "  --resume          跳过已完成实验"
    echo ""
    echo "可用类别 (--category):"
    echo "  all               所有 MoE 实验 (共 ${#ALL_MOE_EXPS[@]} 个)"
    echo "  k2                K=2 专家 (共 ${#MOE_K2_EXPS[@]} 个)"
    echo "  k4                K=4 专家 (共 ${#MOE_K4_EXPS[@]} 个)"
    echo "  k6                K=6 专家 (共 ${#MOE_K6_EXPS[@]} 个)"
    echo "  fused             Gate=fused only (共 ${#MOE_GATE_FUSED_EXPS[@]} 个)"
    echo "  policy            Gate=fused+policy (共 ${#MOE_GATE_POLICY_EXPS[@]} 个)"
    echo "  ablation          K=4 消融组 (共 ${#MOE_ABLATION_EXPS[@]} 个)"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 0,1 --parallel 2                   # 2卡并行跑全部"
    echo "  bash $0 --gpus 0 --category k2 --seed 42          # 单卡单种子试水 K=2"
    echo "  bash $0 --gpus 0,1,2 --category ablation --resume # 3卡跑K=4消融，跳过已完成"
    echo "  bash $0 --gpus 0 --exp mm_mae_cnn_gated_patch_moe4 --seed 42  # 跑单个实验"
    echo "  bash $0 --dry-run                                  # 预览所有任务"
}

list_experiments() {
    echo "所有 MoE 实验 (共 ${#ALL_MOE_EXPS[@]} 个):"
    echo ""
    echo "=== K=2 (最小 MoE) ==="
    for exp in "${MOE_K2_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== K=4 (标准 MoE) ==="
    for exp in "${MOE_K4_EXPS[@]}"; do echo "  - $exp"; done
    echo ""
    echo "=== K=6 (更多专家) ==="
    for exp in "${MOE_K6_EXPS[@]}"; do echo "  - $exp"; done
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
SEEDS=("${DEFAULT_SEEDS[@]}")

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --category) CATEGORY=$2; shift 2 ;;
        --exp) CUSTOM_EXPS_STR=$2; shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
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
        k2) SELECTED_EXPS=("${MOE_K2_EXPS[@]}") ;;
        k4) SELECTED_EXPS=("${MOE_K4_EXPS[@]}") ;;
        k6) SELECTED_EXPS=("${MOE_K6_EXPS[@]}") ;;
        fused) SELECTED_EXPS=("${MOE_GATE_FUSED_EXPS[@]}") ;;
        policy) SELECTED_EXPS=("${MOE_GATE_POLICY_EXPS[@]}") ;;
        ablation) SELECTED_EXPS=("${MOE_ABLATION_EXPS[@]}") ;;
        all) SELECTED_EXPS=("${ALL_MOE_EXPS[@]}") ;;
        *) echo "未知类别: $CATEGORY"; show_help; exit 1 ;;
    esac
fi

# 生成实验任务列表 (实验名 + 种子)
TASK_LIST=()
for exp in "${SELECTED_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp}:${seed}")
    done
done

# Resume: 过滤已完成的实验
if [ "$RESUME" = true ]; then
    FILTERED=()
    for task in "${TASK_LIST[@]}"; do
        exp="${task%%:*}"
        seed="${task##*:}"

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
echo "  MoE Experiment Runner (GPU Normalize)"
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
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== MoE 实验监控 (GPU Normalize) ==='" Enter
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

    # 构建 run_id (与 train_multimodal.py 逻辑一致)
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
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH && echo '环境已激活: '${CONDA_ENV}" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== MoE实验: $exp | GPU: $gpu | Seed: $seed | GPU Normalize: ON ==='" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '开始时间:' \$(date)" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" Enter

    # 运行实验
    local cmd="python train_multimodal.py --exp $exp --gpu $gpu --seed $seed --normalize_on_gpu 2>&1 | tee $log_file"
    cmd="$cmd; if [ \$? -eq 0 ]; then echo success > ${STATUS_DIR}/exp_${task_id}.status; else echo failed > ${STATUS_DIR}/exp_${task_id}.status; fi"
    cmd="$cmd; rm -f ${STATUS_DIR}/gpu_${gpu}.lock"
    cmd="$cmd; echo ''; echo '=== MoE实验结束 ==='; echo '结束时间:' \$(date)"

    tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd" Enter
}

# ============================================================================
# 主调度循环
# ============================================================================
echo ""
echo "开始调度 MoE 实验..."
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

echo "所有 $TOTAL 个 MoE 任务已调度"
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
echo "  所有 MoE 实验已完成! (GPU Normalize版本)"
echo "=============================================="

SUCCESS=0
FAILED=0

for task in "${TASK_LIST[@]}"; do
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
