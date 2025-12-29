#!/bin/bash
# ============================================================================
# 并行运行所有实验 (每次2个，每个实验在独立窗口)
# ============================================================================
# 使用方法: 在tmux中运行
#   bash run_all_experiments_parallel.sh
#
# 会为每个实验创建独立的tmux窗口，方便查看各实验进度
# Ctrl+b n/p 切换窗口，Ctrl+b w 查看所有窗口
# ============================================================================

GPU_ID=3
LOG_DIR="logs/experiment_runs"
PROJECT_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"

mkdir -p "$LOG_DIR"

# 检查是否在tmux中
if [[ -z "$TMUX" ]]; then
    echo "请在tmux中运行此脚本!"
    echo "  tmux new -s exp"
    echo "  bash run_all_experiments_parallel.sh"
    exit 1
fi

SESSION=$(tmux display-message -p '#S')

echo "=============================================="
echo "  并行实验运行 (每次2个)"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Session: $SESSION"
echo "=============================================="
echo ""

# 在新窗口中运行实验的函数
run_in_window() {
    local exp=$1
    local window_name=$2

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
    tmux send-keys -t "$window_name" "python train.py --exp $exp --gpu $GPU_ID 2>&1 | tee $LOG_DIR/${exp}.log; echo ''; echo '>>> $exp 完成!'" Enter
}

# 等待窗口中的实验完成
wait_for_window() {
    local window_name=$1
    echo "  等待 $window_name 完成..."

    # 等待直到窗口中没有python进程
    while true; do
        # 检查窗口是否还有python train.py进程
        if ! tmux list-panes -t "$window_name" -F '#{pane_pid}' 2>/dev/null | xargs -I{} ps --ppid {} 2>/dev/null | grep -q "python"; then
            break
        fi
        sleep 5
    done
}

# 批次定义
declare -a BATCH1=("mlp_baseline" "light_cnn_baseline")
declare -a BATCH2=("resnet_baseline" "resnet_imagenet")
declare -a BATCH3=("mlp_patch_level" "light_cnn_patch_level")
declare -a BATCH4=("resnet_patch_level" "simclr_mlp")
declare -a BATCH5=("simclr_cnn" "mae_cnn")
declare -a BATCH6=("simclr_cnn_patch_level")

run_batch() {
    local batch_num=$1
    shift
    local exps=("$@")

    echo "========================================"
    echo "批次 $batch_num/6: ${exps[*]}"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # 启动当前批次的所有实验
    for exp in "${exps[@]}"; do
        echo "  启动: $exp"
        run_in_window "$exp" "$exp"
        sleep 2  # 稍微错开启动时间
    done

    # 等待当前批次所有实验完成
    for exp in "${exps[@]}"; do
        wait_for_window "$exp"
        echo "  完成: $exp"
    done

    echo "批次 $batch_num 完成!"
    echo ""
}

# 依次运行各批次
run_batch 1 "${BATCH1[@]}"
run_batch 2 "${BATCH2[@]}"
run_batch 3 "${BATCH3[@]}"
run_batch 4 "${BATCH4[@]}"
run_batch 5 "${BATCH5[@]}"
run_batch 6 "${BATCH6[@]}"

echo "========================================"
echo "所有 11 个实验完成!"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "各实验窗口仍然保留，可用 Ctrl+b w 查看"
