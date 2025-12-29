#!/bin/bash
# ============================================================================
# 动态并行实验运行脚本
# ============================================================================
# 使用方法: 在tmux中运行
#   bash run_experiments.sh [并行数量] [实验列表]
#
# 示例:
#   bash run_experiments.sh 2          # 并行跑2个，运行所有实验
#   bash run_experiments.sh 3          # 并行跑3个，运行所有实验
#   bash run_experiments.sh 2 mlp_baseline light_cnn_baseline  # 只运行指定实验
#
# Ctrl+b n/p 切换窗口，Ctrl+b w 查看所有窗口
# ============================================================================

GPU_ID=3
LOG_DIR="logs/experiment_runs"
PROJECT_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"

mkdir -p "$LOG_DIR"

# 默认并行数量
PARALLEL_COUNT=${1:-2}

# 检查并行数量参数
if ! [[ "$PARALLEL_COUNT" =~ ^[0-9]+$ ]] || [ "$PARALLEL_COUNT" -lt 1 ]; then
    echo "错误: 并行数量必须是正整数"
    echo "用法: bash run_experiments.sh [并行数量] [实验列表]"
    exit 1
fi

# 所有可用的实验 (按推荐顺序)
ALL_EXPERIMENTS=(
    "mlp_baseline"
    "light_cnn_baseline"
    "resnet_baseline"
    "resnet_imagenet"
    "mlp_patch_level"
    "light_cnn_patch_level"
    "resnet_patch_level"
    "simclr_mlp"
    "simclr_cnn"
    "mae_cnn"
    "simclr_cnn_patch_level"
)

# 如果指定了实验列表，使用指定的；否则使用全部
if [ $# -gt 1 ]; then
    shift  # 移除第一个参数（并行数量）
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

# 验证实验名称
for exp in "${EXPERIMENTS[@]}"; do
    valid=false
    for all_exp in "${ALL_EXPERIMENTS[@]}"; do
        if [ "$exp" == "$all_exp" ]; then
            valid=true
            break
        fi
    done
    if [ "$valid" == "false" ]; then
        echo "错误: 未知实验 '$exp'"
        echo "可用实验: ${ALL_EXPERIMENTS[*]}"
        exit 1
    fi
done

# 检查是否在tmux中
if [[ -z "$TMUX" ]]; then
    echo "请在tmux中运行此脚本!"
    echo "  tmux new -s exp"
    echo "  bash run_experiments.sh $PARALLEL_COUNT"
    exit 1
fi

SESSION=$(tmux display-message -p '#S')
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo "=============================================="
echo "  动态并行实验运行"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Session: $SESSION"
echo "并行数量: $PARALLEL_COUNT"
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "实验列表: ${EXPERIMENTS[*]}"
echo "=============================================="
echo ""

# 记录运行中的实验 (window_name -> experiment_name)
declare -A RUNNING_EXPERIMENTS
# 实验队列索引
QUEUE_INDEX=0

# 在新窗口中运行实验的函数
run_in_window() {
    local exp=$1
    local window_name="exp_${exp}"

    echo "  [启动] $exp (窗口: $window_name)"

    # 创建新窗口并运行实验
    tmux new-window -n "$window_name"
    tmux send-keys -t "$window_name" "source ~/anaconda3/etc/profile.d/conda.sh && conda activate alphaearth" Enter
    sleep 1
    tmux send-keys -t "$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$window_name" "export CUDA_VISIBLE_DEVICES=$GPU_ID" Enter
    tmux send-keys -t "$window_name" "echo '========================================'" Enter
    tmux send-keys -t "$window_name" "echo '实验: $exp'" Enter
    tmux send-keys -t "$window_name" "echo '开始时间:' \$(date)" Enter
    tmux send-keys -t "$window_name" "echo '========================================'" Enter
    tmux send-keys -t "$window_name" "python train.py --exp $exp --gpu $GPU_ID 2>&1 | tee $LOG_DIR/${exp}.log; echo ''; echo '>>> $exp 完成! <<<'; echo '完成时间:' \$(date)" Enter

    RUNNING_EXPERIMENTS["$window_name"]="$exp"
}

# 检查窗口中的实验是否完成
is_experiment_done() {
    local window_name=$1

    # 检查窗口是否还有python进程
    if ! tmux list-panes -t "$window_name" -F '#{pane_pid}' 2>/dev/null | xargs -I{} ps --ppid {} 2>/dev/null | grep -q "python"; then
        return 0  # 完成
    fi
    return 1  # 未完成
}

# 启动初始批次的实验
start_initial_batch() {
    local count=$1
    for ((i=0; i<count && QUEUE_INDEX<TOTAL_EXPERIMENTS; i++)); do
        run_in_window "${EXPERIMENTS[$QUEUE_INDEX]}"
        QUEUE_INDEX=$((QUEUE_INDEX + 1))
        sleep 2  # 稍微错开启动时间，避免GPU内存冲突
    done
}

# 主循环：监控并维持并行数量
run_with_queue() {
    echo "========================================"
    echo "开始实验 ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "========================================"

    # 启动初始批次
    start_initial_batch "$PARALLEL_COUNT"

    completed_count=0

    # 持续监控直到所有实验完成
    while [ ${#RUNNING_EXPERIMENTS[@]} -gt 0 ]; do
        sleep 5

        # 检查每个运行中的实验
        for window_name in "${!RUNNING_EXPERIMENTS[@]}"; do
            exp="${RUNNING_EXPERIMENTS[$window_name]}"

            if is_experiment_done "$window_name"; then
                completed_count=$((completed_count + 1))
                echo "  [完成] $exp ($completed_count/$TOTAL_EXPERIMENTS) - $(date '+%H:%M:%S')"
                unset RUNNING_EXPERIMENTS["$window_name"]

                # 如果队列中还有实验，启动下一个
                if [ $QUEUE_INDEX -lt $TOTAL_EXPERIMENTS ]; then
                    sleep 2
                    run_in_window "${EXPERIMENTS[$QUEUE_INDEX]}"
                    QUEUE_INDEX=$((QUEUE_INDEX + 1))
                fi
            fi
        done
    done
}

# 运行实验
run_with_queue

echo ""
echo "========================================"
echo "所有 $TOTAL_EXPERIMENTS 个实验完成!"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "各实验窗口仍然保留，可用 Ctrl+b w 查看"
echo "日志保存在: $LOG_DIR/"
