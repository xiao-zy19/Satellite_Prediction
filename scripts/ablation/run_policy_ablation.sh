#!/bin/bash
# ============================================================================
# 政策输入方式消融实验脚本
# run_policy_ablation.sh
#
# 目的: 对比四种不同政策特征输入方式对人口增长率预测的影响
#
# 消融维度 (政策输入方式):
#   1. none       - 单模态 (仅卫星影像, 无政策特征)
#   2. structured - 结构化政策特征 (12维, 人工提取)
#   3. bert       - BERT政策嵌入 (64维, 从政策文本自动编码)
#   4. hybrid     - 混合特征 (76维 = BERT 64维 + 结构化 12维)
#
# 模型选择 (2个代表性架构, 不含MLP):
#   - LightCNN  : 轻量级CNN (~127K参数, 整体表现最优)
#   - ResNet18  : 大模型代表 (~11.4M参数)
#
# 融合策略: FiLM (在已有实验中表现最稳定)
#
# 实验矩阵:
#   2 models × 4 policy modes × 3 seeds = 24 个实验
#
# 实验名称映射:
#   单模态:     light_cnn_baseline / resnet_baseline (via train.py)
#   structured: mm_cnn_film / mm_resnet18_film       (via train_multimodal_bert.py)
#   bert:       bert_cnn_film / bert_resnet18_film    (via train_multimodal_bert.py)
#   hybrid:     hybrid_cnn_film / hybrid_resnet18_film(via train_multimodal_bert.py)
#
# 使用示例:
#   bash run_policy_ablation.sh --gpus 0,1,2,3 --resume
#   bash run_policy_ablation.sh --model cnn --dry-run
#   bash run_policy_ablation.sh --policy structured,bert --gpus 0,1
#   bash run_policy_ablation.sh --list
# ============================================================================

set -e

# 项目配置
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/ablation_policy"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_policy_ablation"
SESSION_NAME="policy_ablation"

# Conda 环境
CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "${RESULT_DIR}/Multimodal"
mkdir -p "${RESULT_DIR}/MultimodalBert"
mkdir -p "${RESULT_DIR}/MultimodalHybrid"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验配置
#
# 格式: "policy_mode:train_script:exp_name:result_subdir:description"
#
# policy_mode: none / structured / bert / hybrid
# train_script: train.py / train_multimodal_bert.py
# exp_name: 实验预设名
# result_subdir: 结果文件存放子目录 (空=根目录)
# description: 简短描述
# ============================================================================

# --- LightCNN 组 (FiLM融合) ---
CNN_EXPERIMENTS=(
    "none:train.py:light_cnn_baseline:Baseline:LightCNN_仅影像"
    "structured:train_multimodal_bert.py:mm_cnn_film:Multimodal:LightCNN_结构化政策12d"
    "bert:train_multimodal_bert.py:bert_cnn_film:MultimodalBert:LightCNN_BERT政策64d"
    "hybrid:train_multimodal_bert.py:hybrid_cnn_film:MultimodalHybrid:LightCNN_混合政策76d"
)

# --- ResNet18 组 (FiLM融合) ---
RESNET_EXPERIMENTS=(
    "none:train.py:resnet18_baseline:Baseline:ResNet18_仅影像"
    "structured:train_multimodal_bert.py:mm_resnet18_film:Multimodal:ResNet18_结构化政策12d"
    "bert:train_multimodal_bert.py:bert_resnet18_film:MultimodalBert:ResNet18_BERT政策64d"
    "hybrid:train_multimodal_bert.py:hybrid_resnet18_film:MultimodalHybrid:ResNet18_混合政策76d"
)

# 合并
ALL_EXPERIMENTS=("${CNN_EXPERIMENTS[@]}" "${RESNET_EXPERIMENTS[@]}")

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 帮助
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** 政策输入方式消融实验 **"
    echo "   对比: 无政策 / 结构化(12d) / BERT(64d) / 混合(76d)"
    echo "   模型: LightCNN / ResNet18 (FiLM融合)"
    echo "   总计: 2 models × 4 policies × 3 seeds = 24 实验"
    echo ""
    echo "选项:"
    echo "  --help, -h        显示帮助"
    echo "  --list, -l        列出所有实验配置"
    echo "  --gpus GPUS       指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N      并行数量 (默认=GPU数)"
    echo "  --model MODEL     只运行指定模型:"
    echo "                      cnn    - 只运行LightCNN实验"
    echo "                      resnet - 只运行ResNet18实验"
    echo "                      all    - 运行所有 (默认)"
    echo "  --policy P1,P2    只运行指定政策模式 (none,structured,bert,hybrid)"
    echo "  --seeds S1,S2     种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --dry-run         预览模式，不实际运行"
    echo "  --resume          跳过已完成实验"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 0,1,2,3 --resume"
    echo "  bash $0 --model cnn --policy none,structured --dry-run"
    echo "  bash $0 --gpus 0,1 --parallel 2"
}

list_experiments() {
    echo "=============================================="
    echo "  政策输入方式消融实验对照表"
    echo "=============================================="
    echo ""
    echo "=== LightCNN 组 (FiLM融合, ~127K参数) ==="
    for exp_info in "${CNN_EXPERIMENTS[@]}"; do
        IFS=':' read -r policy script exp_name subdir desc <<< "$exp_info"
        printf "  %-12s %-35s %s\n" "[$policy]" "$exp_name" "$desc"
    done
    echo ""
    echo "=== ResNet18 组 (FiLM融合, ~11.4M参数) ==="
    for exp_info in "${RESNET_EXPERIMENTS[@]}"; do
        IFS=':' read -r policy script exp_name subdir desc <<< "$exp_info"
        printf "  %-12s %-35s %s\n" "[$policy]" "$exp_name" "$desc"
    done
    echo ""
    echo "总计: ${#ALL_EXPERIMENTS[@]} 实验配置 × ${#DEFAULT_SEEDS[@]} seeds = $(( ${#ALL_EXPERIMENTS[@]} * ${#DEFAULT_SEEDS[@]} )) 任务"
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
MODEL_FILTER="all"
POLICY_FILTER=""
SEEDS=("${DEFAULT_SEEDS[@]}")
DRY_RUN=false
RESUME=false

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --model) MODEL_FILTER=$2; shift 2 ;;
        --policy) POLICY_FILTER=$2; shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
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
# 筛选实验
# ============================================================================
SELECTED_EXPERIMENTS=()

case $MODEL_FILTER in
    cnn) SELECTED_EXPERIMENTS=("${CNN_EXPERIMENTS[@]}") ;;
    resnet) SELECTED_EXPERIMENTS=("${RESNET_EXPERIMENTS[@]}") ;;
    all) SELECTED_EXPERIMENTS=("${ALL_EXPERIMENTS[@]}") ;;
    *) echo "无效的模型选择: $MODEL_FILTER (有效: cnn, resnet, all)"; exit 1 ;;
esac

# 根据政策模式进一步筛选
if [ -n "$POLICY_FILTER" ]; then
    IFS=',' read -ra POLICY_LIST <<< "$POLICY_FILTER"
    FILTERED_EXPERIMENTS=()
    for exp_info in "${SELECTED_EXPERIMENTS[@]}"; do
        IFS=':' read -r policy script exp_name subdir desc <<< "$exp_info"
        for p in "${POLICY_LIST[@]}"; do
            if [ "$policy" == "$p" ]; then
                FILTERED_EXPERIMENTS+=("$exp_info")
                break
            fi
        done
    done
    SELECTED_EXPERIMENTS=("${FILTERED_EXPERIMENTS[@]}")
fi

# ============================================================================
# 生成任务列表
# ============================================================================
TASK_LIST=()
for exp_info in "${SELECTED_EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp_info}:${seed}")
    done
done

# Resume: 过滤已完成的实验
if [ "$RESUME" = true ]; then
    FILTERED=()
    for task in "${TASK_LIST[@]}"; do
        # 解析 policy:script:exp_name:subdir:desc:seed
        IFS=':' read -r policy script exp_name subdir desc seed <<< "$task"

        # 构建结果文件路径
        if [ "$seed" == "42" ]; then
            run_id="${exp_name}"
        else
            run_id="${exp_name}_seed${seed}"
        fi

        if [ -n "$subdir" ]; then
            result_file="${RESULT_DIR}/${subdir}/${run_id}_results.pkl"
        else
            result_file="${RESULT_DIR}/${run_id}_results.pkl"
        fi

        if [ -f "$result_file" ]; then
            echo "[跳过] $exp_name (seed=$seed, 已有结果: $result_file)"
        else
            FILTERED+=("$task")
        fi
    done
    TASK_LIST=("${FILTERED[@]}")
fi

TOTAL=${#TASK_LIST[@]}

echo "=============================================="
echo "  政策输入方式消融实验"
echo "=============================================="
echo "GPU列表: ${GPUS[*]}"
echo "并行数: $PARALLEL_COUNT"
echo "模型筛选: $MODEL_FILTER"
echo "政策筛选: ${POLICY_FILTER:-all}"
echo "种子列表: ${SEEDS[*]}"
echo "总任务数: $TOTAL"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "没有要运行的实验"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "将运行以下实验任务:"
    echo ""
    printf "  %-4s %-12s %-35s %-6s %s\n" "编号" "政策模式" "实验名称" "种子" "训练脚本"
    echo "  ---- ------------ ----------------------------------- ------ -------------------------"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        IFS=':' read -r policy script exp_name subdir desc seed <<< "$task"
        printf "  %-4d %-12s %-35s %-6s %s\n" "$((i+1))" "[$policy]" "$exp_name" "$seed" "$script"
    done
    echo ""
    echo "共 $TOTAL 个任务 (预览模式, 未实际运行)"
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

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -n "monitor"

tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 政策输入方式消融实验监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
start_experiment() {
    local policy=$1
    local script=$2
    local exp_name=$3
    local subdir=$4
    local desc=$5
    local gpu=$6
    local seed=$7

    local task_id="${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"
    local window_name="${exp_name}_s${seed}"

    echo "[$(date '+%H:%M:%S')] 启动: [$policy] $exp_name (seed=$seed) -> GPU $gpu"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== [$policy] $exp_name | GPU: $gpu | Seed: $seed ==='" Enter

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
    IFS=':' read -r policy script exp_name subdir desc seed <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$policy" "$script" "$exp_name" "$subdir" "$desc" "$gpu" "$seed"
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
    IFS=':' read -r policy script exp_name subdir desc seed <<< "$task"
    start_experiment "$policy" "$script" "$exp_name" "$subdir" "$desc" "$FREE_GPU" "$seed"
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
echo "  政策消融实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r policy script exp_name subdir desc seed <<< "$task"
    task_id="${exp_name}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  [失败] [$policy] $exp_name (seed=$seed)"
        fi
    else
        echo "  [未知] [$policy] $exp_name (seed=$seed)"
    fi
done

echo ""
echo "成功: $SUCCESS | 失败: $FAILED"
echo "日志目录: $LOG_DIR"
echo ""
echo "实验矩阵:"
echo "  ┌─────────────┬──────────────────┬──────────────────────┐"
echo "  │ 政策模式     │ LightCNN (FiLM)  │ ResNet18 (FiLM)      │"
echo "  ├─────────────┼──────────────────┼──────────────────────┤"
echo "  │ none (0d)   │ light_cnn_baseln │ resnet_baseline       │"
echo "  │ struct (12d)│ mm_cnn_film      │ mm_resnet18_film      │"
echo "  │ bert (64d)  │ bert_cnn_film    │ bert_resnet18_film    │"
echo "  │ hybrid (76d)│ hybrid_cnn_film  │ hybrid_resnet18_film  │"
echo "  └─────────────┴──────────────────┴──────────────────────┘"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
