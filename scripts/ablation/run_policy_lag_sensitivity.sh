#!/bin/bash
# ============================================================================
# 政策时间滞后敏感性分析实验脚本
# run_policy_lag_sensitivity.sh
#
# 目的: 在最优多模态配置 (MAE + LightCNN + FiLM + Patch-level + Structured)
#       上检验 policy lag=0/1/2 的稳健性, 为论文第五章 §5.3 提供实验依据.
#
# 实验矩阵:
#   配置: mm_mae_cnn_film_patch (固定)
#   时滞: lag=0, 1, 2  (3种)
#   种子: 42, 123, 456 (3个)
#   总计: 9 个实验 (其中 lag=1 的 3 个如已有则跳过)
#
# 结果存放:
#   默认: bytedance-results/results/Multimodal/
#   文件命名:
#     lag=0: mm_mae_cnn_film_patch_lag0[_seed{s}]_results.pkl
#     lag=1: mm_mae_cnn_film_patch[_seed{s}]_results.pkl  (无 _lag1 后缀)
#     lag=2: mm_mae_cnn_film_patch_lag2[_seed{s}]_results.pkl
#
# 使用示例:
#   bash run_policy_lag_sensitivity.sh --gpus 2 --resume
#   bash run_policy_lag_sensitivity.sh --gpus 2,3 --resume --dry-run
#   bash run_policy_lag_sensitivity.sh --lags 0,2 --gpus 2 --resume
# ============================================================================

set -e

# ============================================================================
# 项目配置
# ============================================================================
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/policy_lag_sensitivity"
STATUS_DIR="${PROJECT_DIR}/.run_status_policy_lag_sens"
SESSION_NAME="policy_lag_sens"

# 结果直接写入 bytedance-results, 论文数据提取脚本可直接读取
RESULT_OUTPUT_DIR="/home/xiaozhenyu/degree_essay/Alpha_Earth/bytedance-results/results/Multimodal"

CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth12"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_OUTPUT_DIR"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验定义
# ============================================================================
EXP_NAME="mm_mae_cnn_film_patch"
TRAIN_SCRIPT="train_multimodal.py"

DEFAULT_GPUS=(2)
DEFAULT_SEEDS=(42 123 456)
DEFAULT_LAGS=(0 1 2)

# ============================================================================
# 辅助函数
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** 政策时间滞后敏感性分析 **"
    echo "固定配置: ${EXP_NAME} (MAE + LightCNN + FiLM + Patch + Structured)"
    echo ""
    echo "选项:"
    echo "  --help, -h       显示帮助"
    echo "  --gpus GPUS      指定GPU (例如: 2 或 2,3, 默认: ${DEFAULT_GPUS[*]})"
    echo "  --lags L1,L2,..  时滞列表 (默认: ${DEFAULT_LAGS[*]})"
    echo "  --seeds S1,S2,.. 种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --result-dir DIR 结果输出目录 (默认: ${RESULT_OUTPUT_DIR})"
    echo "  --dry-run        预览模式"
    echo "  --resume         跳过已完成实验"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 2 --resume"
    echo "  bash $0 --gpus 2,3 --resume --dry-run"
    echo "  bash $0 --lags 0,2 --gpus 2 --resume"
}

# 根据 lag 和 seed 生成 run_id
get_run_id() {
    local lag=$1
    local seed=$2
    local lag_suffix=""
    if [ "$lag" != "1" ]; then
        lag_suffix="_lag${lag}"
    fi
    if [ "$seed" != "42" ]; then
        echo "${EXP_NAME}${lag_suffix}_seed${seed}"
    else
        echo "${EXP_NAME}${lag_suffix}"
    fi
}

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
    local task_id=$2
    echo "$task_id" > "${STATUS_DIR}/gpu_${gpu}.lock"
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
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")
LAGS=("${DEFAULT_LAGS[@]}")
DRY_RUN=false
RESUME=false

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --lags) IFS=',' read -ra LAGS <<< "$2"; shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --result-dir) RESULT_OUTPUT_DIR=$2; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
    esac
done

PARALLEL_COUNT=${#GPUS[@]}

# ============================================================================
# 生成任务列表  格式: "lag:seed:run_id"
# ============================================================================
TASK_LIST=()
for lag in "${LAGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_id=$(get_run_id "$lag" "$seed")
        TASK_LIST+=("${lag}:${seed}:${run_id}")
    done
done

# Resume: 检查 RESULT_OUTPUT_DIR 中已有的 pkl
if [ "$RESUME" = true ]; then
    FILTERED=()
    SKIPPED=0
    for task in "${TASK_LIST[@]}"; do
        IFS=':' read -r lag seed run_id <<< "$task"
        result_file="${RESULT_OUTPUT_DIR}/${run_id}_results.pkl"
        if [ -f "$result_file" ]; then
            SKIPPED=$((SKIPPED + 1))
        else
            FILTERED+=("$task")
        fi
    done
    echo "[Resume] 跳过 $SKIPPED 个已完成实验 (检查目录: ${RESULT_OUTPUT_DIR})"
    TASK_LIST=("${FILTERED[@]}")
fi

TOTAL=${#TASK_LIST[@]}

# ============================================================================
# 信息展示
# ============================================================================
echo "=============================================="
echo "  政策时间滞后敏感性分析"
echo "=============================================="
echo "实验配置:    ${EXP_NAME}"
echo "训练脚本:    ${TRAIN_SCRIPT}"
echo "时滞列表:    ${LAGS[*]}"
echo "种子列表:    ${SEEDS[*]}"
echo "GPU列表:     ${GPUS[*]}"
echo "并行数:      ${PARALLEL_COUNT}"
echo "结果目录:    ${RESULT_OUTPUT_DIR}"
echo "总任务数:    ${TOTAL}"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "没有要运行的实验 (全部已完成或列表为空)"
    exit 0
fi

# ============================================================================
# Dry-run 预览
# ============================================================================
if [ "$DRY_RUN" = true ]; then
    echo "将运行以下实验任务:"
    echo ""
    printf "  %-4s %-5s %-6s %-50s %s\n" "#" "Lag" "Seed" "Run ID" "结果文件"
    echo "  ---- ----- ------ -------------------------------------------------- -------------------"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        IFS=':' read -r lag seed run_id <<< "$task"
        printf "  %-4d %-5s %-6s %-50s %s\n" "$((i+1))" "$lag" "$seed" "$run_id" "${run_id}_results.pkl"
    done
    echo ""
    echo "共 $TOTAL 个任务 (预览模式, 未实际运行)"
    exit 0
fi

# ============================================================================
# 清理旧状态 + 创建 tmux session
# ============================================================================
rm -f "${STATUS_DIR}"/gpu_*.lock
rm -f "${STATUS_DIR}"/exp_*.status

echo "创建 tmux session: $SESSION_NAME"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -n "monitor"
tmux set-option -t "$SESSION_NAME" allow-rename off
tmux set-option -t "$SESSION_NAME" automatic-rename off

tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 政策时滞敏感性分析监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT | Lags: ${LAGS[*]}'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
WINDOW_INDEX=0

start_experiment() {
    local lag=$1
    local seed=$2
    local run_id=$3
    local gpu=$4

    local task_id="lag${lag}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    WINDOW_INDEX=$((WINDOW_INDEX + 1))
    local window_name="lag$(printf '%d' $lag)s$(printf '%d' $seed)"

    echo "[$(date '+%H:%M:%S')] 启动: lag=${lag}, seed=${seed} -> GPU ${gpu} [窗口: ${window_name}]"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== [${window_name}] lag=${lag} seed=${seed} | GPU: ${gpu} ==='" Enter

    # set -o pipefail ensures $? captures python's exit code, not tee's
    local cmd="set -o pipefail; python ${TRAIN_SCRIPT} --exp ${EXP_NAME} --gpu ${gpu} --seed ${seed} --policy_lag ${lag} --normalize_on_gpu --result_dir ${RESULT_OUTPUT_DIR} 2>&1 | tee ${log_file}"
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
    IFS=':' read -r lag seed run_id <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$lag" "$seed" "$run_id" "$gpu"
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
    IFS=':' read -r lag seed run_id <<< "$task"
    start_experiment "$lag" "$seed" "$run_id" "$FREE_GPU"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
done

echo "所有 $TOTAL 个任务已调度"
echo ""

# ============================================================================
# 等待完成
# ============================================================================
echo "等待实验完成... (Ctrl+C 退出脚本, 实验继续在tmux中运行)"
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
echo "  政策时滞敏感性分析完成!"
echo "=============================================="

SUCCESS=0
FAILED=0
FAIL_LIST=""

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r lag seed run_id <<< "$task"
    task_id="lag${lag}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            FAIL_LIST="${FAIL_LIST}  [失败] lag=${lag} seed=${seed} (${run_id})\n"
        fi
    else
        FAIL_LIST="${FAIL_LIST}  [未知] lag=${lag} seed=${seed} (${run_id})\n"
    fi
done

if [ -n "$FAIL_LIST" ]; then
    echo -e "$FAIL_LIST"
fi

echo ""
echo "成功: $SUCCESS | 失败: $FAILED | 总计: $TOTAL"
echo ""
echo "结果目录: ${RESULT_OUTPUT_DIR}"
echo "日志目录: ${LOG_DIR}"
echo ""
echo "结果文件:"
for lag in "${LAGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_id=$(get_run_id "$lag" "$seed")
        result_file="${RESULT_OUTPUT_DIR}/${run_id}_results.pkl"
        if [ -f "$result_file" ]; then
            echo "  [OK] ${run_id}_results.pkl"
        else
            echo "  [--] ${run_id}_results.pkl"
        fi
    done
done
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
