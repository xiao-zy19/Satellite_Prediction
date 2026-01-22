#!/bin/bash
# ============================================================================
# 串行运行所有实验（简单可靠版）
# ============================================================================
# 使用方法: ./run_all_experiments.sh
# ============================================================================

SESSION_NAME="exp_serial"
GPU_ID=3

# 清理旧session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# 创建新session
tmux new-session -d -s "$SESSION_NAME" -n "train"

# 发送初始化命令
tmux send-keys -t "$SESSION_NAME:train" "source ~/anaconda3/etc/profile.d/conda.sh && conda activate alphaearth" Enter
sleep 2
tmux send-keys -t "$SESSION_NAME:train" "export CUDA_VISIBLE_DEVICES=$GPU_ID" Enter
tmux send-keys -t "$SESSION_NAME:train" "cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain" Enter

# 发送实验循环命令
tmux send-keys -t "$SESSION_NAME:train" 'for exp in mlp_baseline light_cnn_baseline resnet_baseline resnet_imagenet mlp_patch_level light_cnn_patch_level resnet_patch_level simclr_mlp simclr_cnn mae_cnn simclr_cnn_patch_level; do
    echo ""
    echo "========================================"
    echo "运行实验: $exp"
    echo "时间: $(date)"
    echo "========================================"
    python train.py --exp $exp --gpu 3 2>&1 | tee logs/experiment_runs/${exp}.log
done
echo "所有实验完成!"' Enter

echo "=============================================="
echo "实验已启动！(串行模式)"
echo "=============================================="
echo "查看进度: tmux attach -t $SESSION_NAME"
echo "分离session: Ctrl+b d"
