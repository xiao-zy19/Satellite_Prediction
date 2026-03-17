#!/bin/bash
# ============================================================================
# BERT端到端投影层评估实验
# run_bert_e2e_evaluation.sh
#
# 目的: 验证修复后的BERT端到端投影训练效果
#       对比 structured (12-dim) vs bert (768-dim E2E) vs hybrid (780-dim E2E)
#
# 基准配置: LightCNN + MAE + FiLM + Patch级 (当前structured最优)
# 同时也跑 gated/concat 融合作为对照
#
# 实验矩阵:
#   政策: bert, hybrid (structured已有结果作为对比基准)
#   融合: film (主实验), gated, concat (对照)
#   种子: 42, 123, 456
#   总计: 2 policy × 3 fusion × 3 seeds = 18 个实验
#
# 结果目录: results/BertE2E/ (独立文件夹, 不与旧结果混淆)
#
# 使用示例:
#   bash run_bert_e2e_evaluation.sh --gpus 0,1 --resume
#   bash run_bert_e2e_evaluation.sh --gpus 3 --dry-run
#   bash run_bert_e2e_evaluation.sh --gpus 0,1,2,3 --fusion film --seeds 42
# ============================================================================

set -e

# ============================================================================
# 项目配置 (按需修改此区域)
# ============================================================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/bert_e2e_eval"
RESULT_DIR="${PROJECT_DIR}/results/BertE2E"
STATUS_DIR="${PROJECT_DIR}/.run_status_bert_e2e"
SESSION_NAME="bert_e2e_eval"

# Conda 环境 (自动检测)
if [ -d "/share_data/data101/xiaozhenyu/anaconda3" ]; then
    CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
elif [ -d "/home/xiaozhenyu/anaconda3" ]; then
    CONDA_BASE="/home/xiaozhenyu/anaconda3"
else
    CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
fi
CONDA_ENV="alphaearth12"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验定义
# ============================================================================
# 格式: "exp_config_name"
# 对应 config_multimodal_bert.py 中的键名

BERT_EXPERIMENTS=(
    "bert_mae_cnn_film_patch"
    "bert_mae_cnn_gated_patch"
    "bert_mae_cnn_concat_patch"
)

HYBRID_EXPERIMENTS=(
    "hybrid_mae_cnn_film_patch"
    "hybrid_mae_cnn_gated_patch"
    "hybrid_mae_cnn_concat_patch"
)

DEFAULT_GPUS=(0 1 2 3)
DEFAULT_SEEDS=(42 123 456)
DEFAULT_FUSIONS=(film gated concat)

# ============================================================================
# 辅助函数
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** BERT端到端投影层评估 — LightCNN + MAE + Patch级 **"
    echo ""
    echo "选项:"
    echo "  --help, -h           显示帮助"
    echo "  --gpus GPUS          指定GPU (例如: 0,1,2,3)"
    echo "  --seeds S1,S2,...    种子列表 (默认: 42,123,456)"
    echo "  --seed SEED          单个种子"
    echo "  --fusion F1,F2,...   融合方式 (film,gated,concat, 默认: 全部)"
    echo "  --policy P1,P2,...   政策模式 (bert,hybrid, 默认: 全部)"
    echo "  --dry-run            预览模式"
    echo "  --resume             跳过已完成实验"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 0,1 --resume"
    echo "  bash $0 --gpus 3 --fusion film --seeds 42 --dry-run"
    echo "  bash $0 --gpus 0,1,2,3 --policy bert"
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
    echo "$2" > "${STATUS_DIR}/gpu_${1}.lock"
}

get_done_count() {
    ls "${STATUS_DIR}"/exp_*.status 2>/dev/null | wc -l
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")
FUSIONS=("${DEFAULT_FUSIONS[@]}")
POLICIES=(bert hybrid)
DRY_RUN=false
RESUME=false

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --fusion) IFS=',' read -ra FUSIONS <<< "$2"; shift 2 ;;
        --policy) IFS=',' read -ra POLICIES <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
    esac
done

# ============================================================================
# 构建任务列表
# ============================================================================
TASK_LIST=()

for policy in "${POLICIES[@]}"; do
    for fusion in "${FUSIONS[@]}"; do
        # 构造实验名
        exp_name="${policy}_mae_cnn_${fusion}_patch"

        for seed in "${SEEDS[@]}"; do
            TASK_LIST+=("${exp_name}:${seed}")
        done
    done
done

# Resume: 跳过已完成
if [ "$RESUME" = true ]; then
    FILTERED=()
    SKIPPED=0
    for task in "${TASK_LIST[@]}"; do
        IFS=':' read -r exp_name seed <<< "$task"
        if [ "$seed" == "42" ]; then
            run_id="${exp_name}"
        else
            run_id="${exp_name}_seed${seed}"
        fi
        result_file="${RESULT_DIR}/${run_id}_results.pkl"
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
echo "  BERT端到端投影层评估实验"
echo "=============================================="
echo "基准: LightCNN + MAE + Patch级"
echo "GPU列表:     ${GPUS[*]}"
echo "政策模式:    ${POLICIES[*]}"
echo "融合方式:    ${FUSIONS[*]}"
echo "种子列表:    ${SEEDS[*]}"
echo "总任务数:    $TOTAL"
echo "结果目录:    $RESULT_DIR"
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
    echo "将运行以下实验:"
    echo ""
    printf "  %-4s %-40s %-6s %s\n" "#" "实验名称" "种子" "GPU(轮询)"
    echo "  ---- ---------------------------------------- ------ ---------"
    for i in "${!TASK_LIST[@]}"; do
        IFS=':' read -r exp_name seed <<< "${TASK_LIST[$i]}"
        gpu_idx=$((i % ${#GPUS[@]}))
        printf "  %-4d %-40s %-6s GPU %s\n" "$((i+1))" "$exp_name" "$seed" "${GPUS[$gpu_idx]}"
    done
    echo ""
    echo "共 $TOTAL 个任务 (预览模式, 未实际运行)"
    exit 0
fi

# ============================================================================
# 清理旧状态 & 创建 tmux session
# ============================================================================
rm -f "${STATUS_DIR}"/gpu_*.lock
rm -f "${STATUS_DIR}"/exp_*.status

echo "创建 tmux session: $SESSION_NAME"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -n "monitor"
tmux set-option -t "$SESSION_NAME" allow-rename off

tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== BERT E2E 评估监控 === 总任务: $TOTAL'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
WINDOW_INDEX=0

start_experiment() {
    local exp_name=$1
    local gpu=$2
    local seed=$3

    local task_id="${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    WINDOW_INDEX=$((WINDOW_INDEX + 1))
    local window_name="e$(printf '%03d' $WINDOW_INDEX)"

    echo "[$(date '+%H:%M:%S')] 启动: $exp_name (seed=$seed) -> GPU $gpu"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter

    # 训练命令: 结果存到 BertE2E/ 子目录
    # train_multimodal_bert.py 默认按 policy_source 存到 MultimodalBert/MultimodalHybrid/
    # 我们通过软链或训练后移动来统一到 BertE2E/
    local cmd="python train_multimodal_bert.py --exp $exp_name --gpu $gpu --seed $seed --normalize_on_gpu --no-wandb 2>&1 | tee $log_file"

    # 训练完成后: 把结果从默认目录复制到 BertE2E/
    if [ "$seed" == "42" ]; then
        local run_id="${exp_name}"
    else
        local run_id="${exp_name}_seed${seed}"
    fi

    # 检测 policy source 确定源目录
    local src_subdir
    if [[ "$exp_name" == bert_* ]]; then
        src_subdir="MultimodalBert"
    elif [[ "$exp_name" == hybrid_* ]]; then
        src_subdir="MultimodalHybrid"
    else
        src_subdir="Multimodal"
    fi

    cmd="$cmd; cp -f ${RESULT_DIR}/../${src_subdir}/${run_id}_results.pkl ${RESULT_DIR}/${run_id}_results.pkl 2>/dev/null || true"
    cmd="$cmd; if [ \$? -eq 0 ] || [ -f ${RESULT_DIR}/${run_id}_results.pkl ]; then echo success > ${STATUS_DIR}/exp_${task_id}.status; else echo failed > ${STATUS_DIR}/exp_${task_id}.status; fi"
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

PARALLEL_COUNT=${#GPUS[@]}
QUEUE_INDEX=0

# 启动初始批次
i=0
while [ $i -lt $PARALLEL_COUNT ] && [ $QUEUE_INDEX -lt $TOTAL ]; do
    task="${TASK_LIST[$QUEUE_INDEX]}"
    IFS=':' read -r exp_name seed <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$exp_name" "$gpu" "$seed"
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
    IFS=':' read -r exp_name seed <<< "$task"
    start_experiment "$exp_name" "$FREE_GPU" "$seed"
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
echo "  BERT E2E 评估实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0
FAIL_LIST=""

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r exp_name seed <<< "$task"
    task_id="${exp_name}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ] && [ "$(cat "$status_file")" = "success" ]; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
        FAIL_LIST="${FAIL_LIST}  [失败] $exp_name (seed=$seed)\n"
    fi
done

if [ -n "$FAIL_LIST" ]; then
    echo -e "$FAIL_LIST"
fi

echo ""
echo "成功: $SUCCESS | 失败: $FAILED | 总计: $TOTAL"
echo ""
echo "结果目录: $RESULT_DIR"
echo "日志目录: $LOG_DIR"
echo ""
echo "对比基准 (structured, 已有结果):"
echo "  mm_mae_cnn_film_patch → results/Multimodal/"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
