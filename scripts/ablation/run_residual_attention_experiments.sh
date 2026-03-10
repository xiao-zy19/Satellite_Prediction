#!/bin/bash
# ============================================================================
# Residual Attention Aggregation 实验脚本
# run_residual_attention_experiments.sh
#
# 目的: 验证 mean + gated attention correction 聚合器在 city-level 下的效果
#
# 实验配置 (9个):
#   单模态 (train.py → results/Baseline/):
#     light_cnn_residual_attention     - 核心对比: vs light_cnn_baseline(mean) / light_cnn_attention
#     mlp_residual_attention           - MLP编码器验证
#     resnet18_residual_attention      - ResNet编码器验证
#     simclr_cnn_residual_attention    - SimCLR预训练 + residual attention
#     mae_cnn_residual_attention       - MAE预训练 + residual attention
#
#   多模态 (train_multimodal.py → results/Multimodal/):
#     mm_cnn_concat_residual_attn     - Concat融合 + residual attention聚合
#     mm_cnn_film_residual_attn       - FiLM融合 + residual attention聚合
#     mm_simclr_cnn_concat_residual_attn - SimCLR + Concat + residual attention
#     mm_mae_cnn_concat_residual_attn    - MAE + Concat + residual attention
#
# 规模: 9 配置 × 3 seeds = 27 实验
#
# 使用示例:
#   bash run_residual_attention_experiments.sh --gpus 0,1,2,3 --resume
#   bash run_residual_attention_experiments.sh --category baseline --dry-run
#   bash run_residual_attention_experiments.sh --category multimodal --gpus 0,1
#   bash run_residual_attention_experiments.sh --exp light_cnn_residual_attention --seed 42
#   bash run_residual_attention_experiments.sh --list
#   bash run_residual_attention_experiments.sh --count
# ============================================================================

set -e

# ============================================================================
# 项目配置
# ============================================================================
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/residual_attention"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_residual_attn"
SESSION_NAME="residual_attn"

CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth12"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "${RESULT_DIR}/Baseline"
mkdir -p "${RESULT_DIR}/Multimodal"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验定义
# 格式: "script:exp_name:result_subdir"
# ============================================================================

# --- 单模态基线 (train.py → Baseline/) ---
BASELINE_EXPS=(
    "train.py:light_cnn_residual_attention:Baseline"
    "train.py:mlp_residual_attention:Baseline"
    "train.py:resnet18_residual_attention:Baseline"
)

# --- 单模态+预训练 (train.py → Baseline/) ---
PRETRAIN_EXPS=(
    "train.py:simclr_cnn_residual_attention:Baseline"
    "train.py:mae_cnn_residual_attention:Baseline"
)

# --- 多模态 (train_multimodal.py → Multimodal/) ---
MULTIMODAL_EXPS=(
    "train_multimodal.py:mm_cnn_concat_residual_attn:Multimodal"
    "train_multimodal.py:mm_cnn_film_residual_attn:Multimodal"
    "train_multimodal.py:mm_simclr_cnn_concat_residual_attn:Multimodal"
    "train_multimodal.py:mm_mae_cnn_concat_residual_attn:Multimodal"
)

# 组合
ALL_EXPS=("${BASELINE_EXPS[@]}" "${PRETRAIN_EXPS[@]}" "${MULTIMODAL_EXPS[@]}")

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 辅助函数
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** Residual Attention Aggregation 实验 **"
    echo "** mean + gated attention correction (gate bias=2.0, σ≈0.88 偏向mean) **"
    echo ""
    echo "选项:"
    echo "  --help, -h           显示帮助"
    echo "  --list, -l           列出所有实验配置"
    echo "  --count              统计实验数量"
    echo "  --gpus GPUS          指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N         并行数量 (默认=GPU数)"
    echo "  --category CAT       实验类别 (见下方)"
    echo "  --exp EXP1,EXP2      指定实验名"
    echo "  --seeds S1,S2,...    种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --seed SEED          单个种子"
    echo "  --dry-run            预览模式"
    echo "  --resume             跳过已完成实验"
    echo ""
    echo "可用类别 (--category):"
    echo "  all          全部 (${#ALL_EXPS[@]})"
    echo "  baseline     单模态无预训练 (${#BASELINE_EXPS[@]})"
    echo "  pretrain     单模态+预训练 (${#PRETRAIN_EXPS[@]})"
    echo "  multimodal   多模态 (${#MULTIMODAL_EXPS[@]})"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 0,1,2,3 --resume"
    echo "  bash $0 --category baseline --dry-run"
    echo "  bash $0 --exp light_cnn_residual_attention,mae_cnn_residual_attention --seed 42"
}

list_experiments() {
    echo "=============================================="
    echo "  Residual Attention 实验列表"
    echo "=============================================="
    echo ""
    printf "  %-45s %-25s %s\n" "实验名称" "训练脚本" "结果目录"
    echo "  --------------------------------------------- ------------------------- ---------------"
    for entry in "${ALL_EXPS[@]}"; do
        IFS=':' read -r script exp_name subdir <<< "$entry"
        printf "  %-45s %-25s %s\n" "$exp_name" "$script" "$subdir"
    done
    echo ""
    echo "共 ${#ALL_EXPS[@]} 个配置 × ${#DEFAULT_SEEDS[@]} seeds = $(( ${#ALL_EXPS[@]} * ${#DEFAULT_SEEDS[@]} )) 个实验"
}

count_experiments() {
    echo "=============================================="
    echo "  实验数量统计"
    echo "=============================================="
    echo ""
    printf "  %-15s %3d 配置 × %d seeds = %3d 实验\n" "baseline"    "${#BASELINE_EXPS[@]}"    "${#DEFAULT_SEEDS[@]}" "$(( ${#BASELINE_EXPS[@]} * ${#DEFAULT_SEEDS[@]} ))"
    printf "  %-15s %3d 配置 × %d seeds = %3d 实验\n" "pretrain"    "${#PRETRAIN_EXPS[@]}"    "${#DEFAULT_SEEDS[@]}" "$(( ${#PRETRAIN_EXPS[@]} * ${#DEFAULT_SEEDS[@]} ))"
    printf "  %-15s %3d 配置 × %d seeds = %3d 实验\n" "multimodal"  "${#MULTIMODAL_EXPS[@]}"  "${#DEFAULT_SEEDS[@]}" "$(( ${#MULTIMODAL_EXPS[@]} * ${#DEFAULT_SEEDS[@]} ))"
    echo "  ---"
    printf "  %-15s %3d 配置 × %d seeds = %3d 实验\n" "all"         "${#ALL_EXPS[@]}"         "${#DEFAULT_SEEDS[@]}" "$(( ${#ALL_EXPS[@]} * ${#DEFAULT_SEEDS[@]} ))"
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
CATEGORY="all"
CUSTOM_EXPS_STR=""
SEEDS=("${DEFAULT_SEEDS[@]}")
DRY_RUN=false
RESUME=false

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --count) count_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --category) CATEGORY=$2; shift 2 ;;
        --exp) CUSTOM_EXPS_STR=$2; shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
    esac
done

# 并行数
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
    IFS=',' read -ra CUSTOM_EXPS <<< "$CUSTOM_EXPS_STR"
    SELECTED_EXPS=()
    for entry in "${ALL_EXPS[@]}"; do
        IFS=':' read -r script exp_name subdir <<< "$entry"
        for exp in "${CUSTOM_EXPS[@]}"; do
            if [ "$exp_name" == "$exp" ]; then
                SELECTED_EXPS+=("$entry")
                break
            fi
        done
    done
    if [ ${#SELECTED_EXPS[@]} -eq 0 ]; then
        echo "错误: 未找到匹配的实验: $CUSTOM_EXPS_STR"
        echo "提示: 运行 --list 查看全部可用实验名"
        exit 1
    fi
else
    case $CATEGORY in
        all) SELECTED_EXPS=("${ALL_EXPS[@]}") ;;
        baseline) SELECTED_EXPS=("${BASELINE_EXPS[@]}") ;;
        pretrain) SELECTED_EXPS=("${PRETRAIN_EXPS[@]}") ;;
        multimodal) SELECTED_EXPS=("${MULTIMODAL_EXPS[@]}") ;;
        *) echo "未知类别: $CATEGORY"; show_help; exit 1 ;;
    esac
fi

# ============================================================================
# 生成任务列表 (实验 × 种子)
# ============================================================================
TASK_LIST=()
for entry in "${SELECTED_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${entry}:${seed}")
    done
done

# ============================================================================
# Resume: 跳过已完成
# ============================================================================
if [ "$RESUME" = true ]; then
    FILTERED=()
    SKIPPED=0
    for task in "${TASK_LIST[@]}"; do
        IFS=':' read -r script exp_name subdir seed <<< "$task"

        if [ "$seed" == "42" ]; then
            run_id="${exp_name}"
        else
            run_id="${exp_name}_seed${seed}"
        fi

        result_file="${RESULT_DIR}/${subdir}/${run_id}_results.pkl"

        if [ -f "$result_file" ]; then
            SKIPPED=$((SKIPPED + 1))
        else
            FILTERED+=("$task")
        fi
    done
    echo "[Resume] 跳过 $SKIPPED 个已完成实验"
    TASK_LIST=("${FILTERED[@]}")
fi

TOTAL=${#TASK_LIST[@]}

# ============================================================================
# 信息展示
# ============================================================================
echo "=============================================="
echo "  Residual Attention Aggregation 实验"
echo "=============================================="
echo "GPU列表:     ${GPUS[*]}"
echo "并行数:      $PARALLEL_COUNT"
echo "类别:        ${CATEGORY}${CUSTOM_EXPS_STR:+ (自定义: $CUSTOM_EXPS_STR)}"
echo "种子列表:    ${SEEDS[*]}"
echo "总任务数:    $TOTAL"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "没有要运行的实验"
    exit 0
fi

# ============================================================================
# Dry-run 预览
# ============================================================================
if [ "$DRY_RUN" = true ]; then
    echo "将运行以下实验任务:"
    echo ""
    printf "  %-4s %-45s %-6s %-15s %s\n" "#" "实验名称" "种子" "结果目录" "训练脚本"
    echo "  ---- --------------------------------------------- ------ --------------- -------------------------"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        IFS=':' read -r script exp_name subdir seed <<< "$task"
        printf "  %-4d %-45s %-6s %-15s %s\n" "$((i+1))" "$exp_name" "$seed" "$subdir" "$script"
    done
    echo ""
    echo "共 $TOTAL 个任务 (预览模式, 未实际运行)"
    exit 0
fi

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
# 清理旧状态
# ============================================================================
rm -f "${STATUS_DIR}"/gpu_*.lock
rm -f "${STATUS_DIR}"/exp_*.status

# ============================================================================
# 创建 tmux session
# ============================================================================
echo "创建 tmux session: $SESSION_NAME"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -n "monitor"
tmux set-option -t "$SESSION_NAME" allow-rename off
tmux set-option -t "$SESSION_NAME" automatic-rename off

tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== Residual Attention 实验监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
WINDOW_INDEX=0

start_experiment() {
    local script=$1
    local exp_name=$2
    local subdir=$3
    local gpu=$4
    local seed=$5

    local task_id="${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    WINDOW_INDEX=$((WINDOW_INDEX + 1))
    local window_name="e$(printf '%04d' $WINDOW_INDEX)"

    echo "[$(date '+%H:%M:%S')] 启动: $exp_name (seed=$seed) -> GPU $gpu [窗口: $window_name]"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== [$window_name] $exp_name | GPU: $gpu | Seed: $seed ==='" Enter

    local cmd="python ${script} --exp $exp_name --gpu $gpu --seed $seed --normalize_on_gpu 2>&1 | tee $log_file"
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
    IFS=':' read -r script exp_name subdir seed <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$script" "$exp_name" "$subdir" "$gpu" "$seed"
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
    IFS=':' read -r script exp_name subdir seed <<< "$task"
    start_experiment "$script" "$exp_name" "$subdir" "$FREE_GPU" "$seed"
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
echo "  Residual Attention 实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0
FAIL_LIST=""

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r script exp_name subdir seed <<< "$task"
    task_id="${exp_name}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            FAIL_LIST="${FAIL_LIST}  [失败] $exp_name (seed=$seed)\n"
        fi
    else
        FAIL_LIST="${FAIL_LIST}  [未知] $exp_name (seed=$seed)\n"
    fi
done

if [ -n "$FAIL_LIST" ]; then
    echo -e "$FAIL_LIST"
fi

echo ""
echo "成功: $SUCCESS | 失败: $FAILED | 总计: $TOTAL"
echo ""
echo "结果目录:"
echo "  单模态: ${RESULT_DIR}/Baseline/"
echo "  多模态: ${RESULT_DIR}/Multimodal/"
echo ""
echo "日志目录: $LOG_DIR"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
