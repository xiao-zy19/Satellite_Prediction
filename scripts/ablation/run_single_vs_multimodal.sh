#!/bin/bash
# ============================================================================
# 单模态 vs 多模态 对比实验脚本
# run_single_vs_multimodal.sh
#
# 目的: 验证政策特征对人口增长率预测的贡献
#
# 核心对比:
#   单模态 (仅卫星影像)  vs  多模态 (卫星影像 + 政策特征)
#
# 实验设计:
# 1. 选择多模态中表现较好的配置 (mm_cnn_*, mm_resnet*)
# 2. 运行对应的单模态配置 (light_cnn_*, resnet*_baseline)
# 3. 通过对比分析政策特征的贡献
#
# 注意: 单模态和多模态的后处理层结构略有不同，
#       因此这是"功能性消融"而非"严格架构消融"
#
# 使用示例:
#   bash run_single_vs_multimodal.sh --mode single --resume --gpus 0,1,2,3
#   bash run_single_vs_multimodal.sh --group 1 --dry-run
# ============================================================================

set -e

# 项目配置
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/ablation_runs"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_ablation"
SESSION_NAME="single_vs_multi"

# Conda 环境
CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 消融实验配置
#
# 格式: "单模态实验名:多模态实验名:描述"
#
# 选择依据 (基于已有实验结果):
# - 多模态 mm_cnn_attention (R²=0.64) 表现最好
# - 多模态 mm_cnn_film (R²=0.60) 稳定
# - 多模态 mm_cnn_film_transformer_2d (R²=0.61) 最佳聚合
# - 单模态 resnet_baseline (R²=0.57) 单模态最好
# - 单模态 light_cnn_patch_level (R²=0.54)
# ============================================================================

# --- 消融组1: LightCNN 基础对比 (mean聚合) ---
# 验证: concat/gated/attention/film 融合策略的效果
ABLATION_GROUP1=(
    "light_cnn_baseline:mm_cnn_concat:LightCNN_mean_concat"
    "light_cnn_baseline:mm_cnn_gated:LightCNN_mean_gated"
    "light_cnn_baseline:mm_cnn_attention:LightCNN_mean_attention"
    "light_cnn_baseline:mm_cnn_film:LightCNN_mean_film"
)

# --- 消融组2: LightCNN + trimmed_mean 聚合 ---
ABLATION_GROUP2=(
    "light_cnn_trimmed_mean:mm_cnn_concat_trimmed:LightCNN_trimmed_concat"
    "light_cnn_trimmed_mean:mm_cnn_gated_trimmed:LightCNN_trimmed_gated"
    "light_cnn_trimmed_mean:mm_cnn_attention_trimmed:LightCNN_trimmed_attention"
    "light_cnn_trimmed_mean:mm_cnn_film_trimmed:LightCNN_trimmed_film"
)

# --- 消融组3: LightCNN + Transformer聚合 ---
ABLATION_GROUP3=(
    "light_cnn_transformer:mm_cnn_concat_transformer:LightCNN_transformer_concat"
    "light_cnn_transformer_2d:mm_cnn_concat_transformer_2d:LightCNN_transformer2d_concat"
    "light_cnn_transformer_2d:mm_cnn_film_transformer_2d:LightCNN_transformer2d_film"
)

# --- 消融组4: ResNet18 对比 ---
ABLATION_GROUP4=(
    "resnet18_baseline:mm_resnet18_concat:ResNet18_mean_concat"
    "resnet18_baseline:mm_resnet18_gated:ResNet18_mean_gated"
    "resnet18_baseline:mm_resnet18_film:ResNet18_mean_film"
)

# --- 消融组5: ResNet34 对比 ---
ABLATION_GROUP5=(
    "resnet34_baseline:mm_resnet34_concat:ResNet34_mean_concat"
    "resnet34_baseline:mm_resnet34_gated:ResNet34_mean_gated"
    "resnet34_baseline:mm_resnet34_film:ResNet34_mean_film"
)

# --- 消融组6: Patch-level 对比 ---
ABLATION_GROUP6=(
    "light_cnn_patch_level:mm_cnn_concat_patch:LightCNN_patch_concat"
    "light_cnn_patch_level:mm_cnn_gated_patch:LightCNN_patch_gated"
    "light_cnn_patch_level:mm_cnn_film_patch:LightCNN_patch_film"
)

# --- 消融组7: MAE预训练对比 ---
ABLATION_GROUP7=(
    "mae_cnn:mm_mae_cnn_concat:MAE_CNN_concat"
    "mae_cnn_trimmed_mean:mm_mae_cnn_concat_trimmed:MAE_CNN_trimmed_concat"
)

# 所有消融组
ALL_ABLATION_PAIRS=(
    "${ABLATION_GROUP1[@]}"
    "${ABLATION_GROUP2[@]}"
    "${ABLATION_GROUP3[@]}"
    "${ABLATION_GROUP4[@]}"
    "${ABLATION_GROUP5[@]}"
    "${ABLATION_GROUP6[@]}"
    "${ABLATION_GROUP7[@]}"
)

# 提取所有单模态实验
declare -A SINGLE_MODAL_EXPS
declare -A MULTI_MODAL_EXPS

for pair in "${ALL_ABLATION_PAIRS[@]}"; do
    IFS=':' read -r single multi desc <<< "$pair"
    SINGLE_MODAL_EXPS["$single"]=1
    MULTI_MODAL_EXPS["$multi"]=1
done

# 转换为数组
SINGLE_MODAL_LIST=($(echo "${!SINGLE_MODAL_EXPS[@]}" | tr ' ' '\n' | sort -u))
MULTI_MODAL_LIST=($(echo "${!MULTI_MODAL_EXPS[@]}" | tr ' ' '\n' | sort -u))

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 帮助
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "消融实验脚本 - 对比单模态与多模态效果"
    echo ""
    echo "选项:"
    echo "  --help, -h        显示帮助"
    echo "  --list, -l        列出所有消融实验对"
    echo "  --gpus GPUS       指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N      并行数量"
    echo "  --mode MODE       运行模式:"
    echo "                      single  - 只运行单模态实验"
    echo "                      multi   - 只运行多模态实验"
    echo "                      all     - 运行所有实验 (默认)"
    echo "  --group N         只运行第N组消融实验 (1-7)"
    echo "  --seeds S1,S2     种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --dry-run         预览模式"
    echo "  --resume          跳过已完成实验"
    echo ""
    echo "消融组说明:"
    echo "  1: LightCNN + mean聚合 (${#ABLATION_GROUP1[@]}对)"
    echo "  2: LightCNN + trimmed_mean聚合 (${#ABLATION_GROUP2[@]}对)"
    echo "  3: LightCNN + Transformer聚合 (${#ABLATION_GROUP3[@]}对)"
    echo "  4: ResNet18 (${#ABLATION_GROUP4[@]}对)"
    echo "  5: ResNet34 (${#ABLATION_GROUP5[@]}对)"
    echo "  6: Patch-level (${#ABLATION_GROUP6[@]}对)"
    echo "  7: MAE预训练 (${#ABLATION_GROUP7[@]}对)"
    echo ""
    echo "总计: ${#SINGLE_MODAL_LIST[@]}个单模态 + ${#MULTI_MODAL_LIST[@]}个多模态 = $((${#SINGLE_MODAL_LIST[@]} + ${#MULTI_MODAL_LIST[@]}))个实验配置"
}

list_experiments() {
    echo "=============================================="
    echo "  消融实验对照表"
    echo "=============================================="
    echo ""

    echo "=== 消融组1: LightCNN + mean聚合 ==="
    for pair in "${ABLATION_GROUP1[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
    echo ""

    echo "=== 消融组2: LightCNN + trimmed_mean聚合 ==="
    for pair in "${ABLATION_GROUP2[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
    echo ""

    echo "=== 消融组3: LightCNN + Transformer聚合 ==="
    for pair in "${ABLATION_GROUP3[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
    echo ""

    echo "=== 消融组4: ResNet18 ==="
    for pair in "${ABLATION_GROUP4[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
    echo ""

    echo "=== 消融组5: ResNet34 ==="
    for pair in "${ABLATION_GROUP5[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
    echo ""

    echo "=== 消融组6: Patch-level ==="
    for pair in "${ABLATION_GROUP6[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
    echo ""

    echo "=== 消融组7: MAE预训练 ==="
    for pair in "${ABLATION_GROUP7[@]}"; do
        IFS=':' read -r single multi desc <<< "$pair"
        echo "  单模态: $single"
        echo "  多模态: $multi"
        echo "  ---"
    done
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
MODE="all"
GROUP=0
SEEDS=("${DEFAULT_SEEDS[@]}")
DRY_RUN=false
RESUME=false

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --group) GROUP=$2; shift 2 ;;
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
# 根据参数选择实验
# ============================================================================
SELECTED_PAIRS=()

if [ "$GROUP" -gt 0 ]; then
    case $GROUP in
        1) SELECTED_PAIRS=("${ABLATION_GROUP1[@]}") ;;
        2) SELECTED_PAIRS=("${ABLATION_GROUP2[@]}") ;;
        3) SELECTED_PAIRS=("${ABLATION_GROUP3[@]}") ;;
        4) SELECTED_PAIRS=("${ABLATION_GROUP4[@]}") ;;
        5) SELECTED_PAIRS=("${ABLATION_GROUP5[@]}") ;;
        6) SELECTED_PAIRS=("${ABLATION_GROUP6[@]}") ;;
        7) SELECTED_PAIRS=("${ABLATION_GROUP7[@]}") ;;
        *) echo "无效的组号: $GROUP (有效范围: 1-7)"; exit 1 ;;
    esac
else
    SELECTED_PAIRS=("${ALL_ABLATION_PAIRS[@]}")
fi

# 提取实验列表
EXPS_TO_RUN=()

for pair in "${SELECTED_PAIRS[@]}"; do
    IFS=':' read -r single multi desc <<< "$pair"

    case $MODE in
        single)
            EXPS_TO_RUN+=("single:$single")
            ;;
        multi)
            EXPS_TO_RUN+=("multi:$multi")
            ;;
        all)
            EXPS_TO_RUN+=("single:$single")
            EXPS_TO_RUN+=("multi:$multi")
            ;;
        *)
            echo "无效的模式: $MODE (有效: single, multi, all)"
            exit 1
            ;;
    esac
done

# 去重
EXPS_TO_RUN=($(printf '%s\n' "${EXPS_TO_RUN[@]}" | sort -u))

# 生成任务列表 (实验类型:实验名:种子)
TASK_LIST=()
for exp_info in "${EXPS_TO_RUN[@]}"; do
    IFS=':' read -r exp_type exp_name <<< "$exp_info"
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp_type}:${exp_name}:${seed}")
    done
done

# Resume: 过滤已完成的实验
if [ "$RESUME" = true ]; then
    FILTERED=()
    for task in "${TASK_LIST[@]}"; do
        IFS=':' read -r exp_type exp_name seed <<< "$task"

        # 构建结果文件路径
        if [ "$seed" == "42" ]; then
            run_id="${exp_name}"
        else
            run_id="${exp_name}_seed${seed}"
        fi

        # 单模态结果在 Baseline/ 子目录，多模态 mm_* 结果在 Multimodal/ 子目录
        if [ "$exp_type" == "multi" ]; then
            result_file="${RESULT_DIR}/Multimodal/${run_id}_results.pkl"
        else
            result_file="${RESULT_DIR}/Baseline/${run_id}_results.pkl"
        fi

        if [ -f "$result_file" ]; then
            echo "[跳过] $exp_name (seed=$seed, 已有结果)"
        else
            FILTERED+=("$task")
        fi
    done
    TASK_LIST=("${FILTERED[@]}")
fi

TOTAL=${#TASK_LIST[@]}

echo "=============================================="
echo "  消融实验配置"
echo "=============================================="
echo "GPU列表: ${GPUS[*]}"
echo "并行数: $PARALLEL_COUNT"
echo "模式: $MODE"
if [ "$GROUP" -gt 0 ]; then
    echo "消融组: $GROUP"
fi
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
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        IFS=':' read -r exp_type exp_name seed <<< "$task"
        echo "  $((i+1)). [$exp_type] $exp_name (seed=$seed)"
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

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -n "monitor"

tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 消融实验监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
start_experiment() {
    local exp_type=$1
    local exp_name=$2
    local gpu=$3
    local seed=$4

    local task_id="${exp_type}_${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"
    local window_name="${exp_name}_s${seed}"

    echo "[$(date '+%H:%M:%S')] 启动: [$exp_type] $exp_name (seed=$seed) -> GPU $gpu"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== [$exp_type] $exp_name | GPU: $gpu | Seed: $seed ==='" Enter

    # 根据实验类型选择训练脚本
    # 两者都使用 --normalize_on_gpu 以加速训练
    local train_script=""
    local extra_args="--normalize_on_gpu"

    if [ "$exp_type" == "single" ]; then
        train_script="train.py"
    else
        train_script="train_multimodal.py"
    fi

    local cmd="python ${train_script} --exp $exp_name --gpu $gpu --seed $seed $extra_args 2>&1 | tee $log_file"
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
    IFS=':' read -r exp_type exp_name seed <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$exp_type" "$exp_name" "$gpu" "$seed"
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
    IFS=':' read -r exp_type exp_name seed <<< "$task"
    start_experiment "$exp_type" "$exp_name" "$FREE_GPU" "$seed"
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
echo "  消融实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r exp_type exp_name seed <<< "$task"
    task_id="${exp_type}_${exp_name}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  [失败] [$exp_type] $exp_name (seed=$seed)"
        fi
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
