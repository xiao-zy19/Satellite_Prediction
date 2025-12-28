#!/bin/bash
# Launch experiments in tmux with separate panes for each experiment
# Usage: bash start_experiments.sh

cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain

SESSION="exp_all"
CONDA_CMD="source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate alphaearth; export CUDA_VISIBLE_DEVICES=3; cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"

# Kill existing session
tmux kill-session -t $SESSION 2>/dev/null

# All experiments in order
ALL_EXPS=(
    "mlp_baseline"
    "light_cnn_baseline"
    "mlp_patch_level"
    "resnet_baseline"
    "resnet_imagenet"
    "resnet_patch_level"
    "simclr_mlp"
    "simclr_cnn"
    "simclr_cnn_patch_level"
    "mae_cnn"
    "light_cnn_patch_level"
)

# Create session with first 3 experiments (Batch 1)
tmux new-session -d -s $SESSION -n "batch1"

# Pane 0: mlp_baseline
tmux send-keys -t $SESSION:batch1.0 "$CONDA_CMD && python train.py --exp mlp_baseline 2>&1 | tee logs/experiment_runs/mlp_baseline.log" Enter

# Split and run light_cnn_baseline
tmux split-window -h -t $SESSION:batch1
tmux send-keys -t $SESSION:batch1.1 "$CONDA_CMD && python train.py --exp light_cnn_baseline 2>&1 | tee logs/experiment_runs/light_cnn_baseline.log" Enter

# Split and run mlp_patch_level
tmux split-window -v -t $SESSION:batch1.0
tmux send-keys -t $SESSION:batch1.2 "$CONDA_CMD && python train.py --exp mlp_patch_level 2>&1 | tee logs/experiment_runs/mlp_patch_level.log" Enter

# Balance panes
tmux select-layout -t $SESSION:batch1 tiled

echo "=========================================="
echo "Batch 1 started in tmux session: $SESSION"
echo "Experiments: mlp_baseline, light_cnn_baseline, mlp_patch_level"
echo ""
echo "To view: tmux attach -t $SESSION"
echo "To detach: Ctrl+B then D"
echo "=========================================="
echo ""
echo "After Batch 1 completes, run:"
echo "  bash start_batch2.sh  # for resnet models"
echo "  bash start_batch3.sh  # for simclr models"
echo "  bash start_batch4.sh  # for mae + remaining"
