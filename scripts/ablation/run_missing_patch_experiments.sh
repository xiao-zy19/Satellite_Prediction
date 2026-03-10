#!/bin/bash
# ============================================================================
# 缺失 Patch-Level 实验专用脚本
# run_missing_patch_experiments.sh
#
# 目的:
#   运行 run_policy_ablation_all_experiments.sh 当前未覆盖的 7 个单模态
#   patch-level ResNet/ImageNet 实验。
#
# 说明:
#   这些实验在 config.py 中存在，但在 config_multimodal_bert.py 中没有对应的
#   structured / bert / hybrid patch-level 版本，因此这里只能运行单模态 none 路线:
#   train.py -> results/Baseline/
#
# 实验列表:
#   1. resnet10_patch_level
#   2. resnet34_patch_level
#   3. resnet50_patch_level
#   4. resnet101_patch_level
#   5. resnet34_imagenet_patch_level
#   6. resnet50_imagenet_patch_level
#   7. resnet101_imagenet_patch_level
#
# 规模:
#   7 配置 × 3 seeds = 21 个任务
#
# 使用示例:
#   bash run_missing_patch_experiments.sh --gpus 0,1,2,3 --resume
#   bash run_missing_patch_experiments.sh --category imagenet_patch --seed 42 --dry-run
#   bash run_missing_patch_experiments.sh --exp resnet50_patch_level,resnet101_patch_level
# ============================================================================

set -e

# ============================================================================
# 项目配置
# ============================================================================
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/ablation_missing_patch"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_missing_patch"
SESSION_NAME="missing_patch_exp"

CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth12"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "${RESULT_DIR}/Baseline"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 实验定义
# ============================================================================
PATCH_NO_PRETRAIN_EXPS=(
    "resnet10_patch_level:ResNet10_patch_level"
    "resnet34_patch_level:ResNet34_patch_level"
    "resnet50_patch_level:ResNet50_patch_level"
    "resnet101_patch_level:ResNet101_patch_level"
)

PATCH_IMAGENET_EXPS=(
    "resnet34_imagenet_patch_level:ResNet34_ImageNet_patch_level"
    "resnet50_imagenet_patch_level:ResNet50_ImageNet_patch_level"
    "resnet101_imagenet_patch_level:ResNet101_ImageNet_patch_level"
)

ALL_EXPERIMENTS=(
    "${PATCH_NO_PRETRAIN_EXPS[@]}"
    "${PATCH_IMAGENET_EXPS[@]}"
)

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 辅助函数
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** 缺失 Patch-Level 实验专用脚本 **"
    echo "   仅运行单模态 train.py 路线 (Baseline/)"
    echo "   环境: ${CONDA_ENV}"
    echo ""
    echo "选项:"
    echo "  --help, -h           显示帮助"
    echo "  --list, -l           列出所有实验"
    echo "  --count              统计实验数量"
    echo "  --gpus GPUS          指定 GPU (例如: 0,1,2,3)"
    echo "  --parallel N         并行数量 (默认=GPU数)"
    echo "  --category CAT       实验类别"
    echo "  --exp EXP1,EXP2      指定实验名"
    echo "  --seeds S1,S2,...    种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --seed SEED          单个种子"
    echo "  --dry-run            预览模式"
    echo "  --resume             跳过已完成实验"
    echo ""
    echo "可用类别 (--category):"
    echo "  all                  全部 (${#ALL_EXPERIMENTS[@]})"
    echo "  patch                非 ImageNet patch (${#PATCH_NO_PRETRAIN_EXPS[@]})"
    echo "  imagenet_patch       ImageNet patch (${#PATCH_IMAGENET_EXPS[@]})"
    echo "  resnet10             ResNet10 patch"
    echo "  resnet34             ResNet34 patch + ResNet34 ImageNet patch"
    echo "  resnet50             ResNet50 patch + ResNet50 ImageNet patch"
    echo "  resnet101            ResNet101 patch + ResNet101 ImageNet patch"
}

list_experiments() {
    echo "=============================================="
    echo "  缺失 Patch-Level 实验列表"
    echo "=============================================="
    echo ""
    echo "=== 非 ImageNet Patch (${#PATCH_NO_PRETRAIN_EXPS[@]}) ==="
    for exp_info in "${PATCH_NO_PRETRAIN_EXPS[@]}"; do
        IFS=':' read -r exp_name desc <<< "$exp_info"
        printf "  %-35s %s\n" "$exp_name" "$desc"
    done
    echo ""
    echo "=== ImageNet Patch (${#PATCH_IMAGENET_EXPS[@]}) ==="
    for exp_info in "${PATCH_IMAGENET_EXPS[@]}"; do
        IFS=':' read -r exp_name desc <<< "$exp_info"
        printf "  %-35s %s\n" "$exp_name" "$desc"
    done
    echo ""
    echo "总计: ${#ALL_EXPERIMENTS[@]} 个实验配置 × ${#DEFAULT_SEEDS[@]} seeds = $((${#ALL_EXPERIMENTS[@]} * ${#DEFAULT_SEEDS[@]})) 个任务"
    echo ""
    echo "注: 这些实验没有对应的多模态 patch-level policy 版本，因此不会展开为 structured / bert / hybrid。"
}

count_experiments() {
    echo "=============================================="
    echo "  实验数量统计"
    echo "=============================================="
    echo ""
    printf "  %-18s %3d 配置 → %3d 任务\n" "patch"          "${#PATCH_NO_PRETRAIN_EXPS[@]}" "$((${#PATCH_NO_PRETRAIN_EXPS[@]} * ${#DEFAULT_SEEDS[@]}))"
    printf "  %-18s %3d 配置 → %3d 任务\n" "imagenet_patch" "${#PATCH_IMAGENET_EXPS[@]}"   "$((${#PATCH_IMAGENET_EXPS[@]} * ${#DEFAULT_SEEDS[@]}))"
    printf "  %-18s %3d 配置 → %3d 任务\n" "all"            "${#ALL_EXPERIMENTS[@]}"         "$((${#ALL_EXPERIMENTS[@]} * ${#DEFAULT_SEEDS[@]}))"
}

get_experiment_names() {
    local entries=("$@")
    local names=()
    local exp_name
    local desc
    for exp_info in "${entries[@]}"; do
        IFS=':' read -r exp_name desc <<< "$exp_info"
        names+=("$exp_name")
    done
    printf "%s\n" "${names[@]}"
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
        all)
            mapfile -t SELECTED_EXPS < <(get_experiment_names "${ALL_EXPERIMENTS[@]}")
            ;;
        patch)
            mapfile -t SELECTED_EXPS < <(get_experiment_names "${PATCH_NO_PRETRAIN_EXPS[@]}")
            ;;
        imagenet_patch)
            mapfile -t SELECTED_EXPS < <(get_experiment_names "${PATCH_IMAGENET_EXPS[@]}")
            ;;
        resnet10)
            SELECTED_EXPS=("resnet10_patch_level")
            ;;
        resnet34)
            SELECTED_EXPS=("resnet34_patch_level" "resnet34_imagenet_patch_level")
            ;;
        resnet50)
            SELECTED_EXPS=("resnet50_patch_level" "resnet50_imagenet_patch_level")
            ;;
        resnet101)
            SELECTED_EXPS=("resnet101_patch_level" "resnet101_imagenet_patch_level")
            ;;
        *)
            echo "未知类别: $CATEGORY"
            show_help
            exit 1
            ;;
    esac
fi

# 验证实验名
declare -A VALID_EXPERIMENTS
while IFS= read -r exp_name; do
    [ -n "$exp_name" ] && VALID_EXPERIMENTS["$exp_name"]=1
done < <(get_experiment_names "${ALL_EXPERIMENTS[@]}")
FILTERED_EXPS=()
for exp in "${SELECTED_EXPS[@]}"; do
    if [ -n "${VALID_EXPERIMENTS[$exp]}" ]; then
        FILTERED_EXPS+=("$exp")
    else
        echo "错误: 未知实验名: $exp"
        echo "提示: 运行 --list 查看可用实验"
        exit 1
    fi
done
SELECTED_EXPS=("${FILTERED_EXPS[@]}")

# ============================================================================
# 生成任务列表
# ============================================================================
TASK_LIST=()
for exp in "${SELECTED_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp}:${seed}")
    done
done

if [ "$RESUME" = true ]; then
    FILTERED=()
    SKIPPED=0
    for task in "${TASK_LIST[@]}"; do
        exp="${task%%:*}"
        seed="${task##*:}"

        if [ "$seed" == "42" ]; then
            run_id="${exp}"
        else
            run_id="${exp}_seed${seed}"
        fi

        result_file="${RESULT_DIR}/Baseline/${run_id}_results.pkl"
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

echo "=============================================="
echo "  缺失 Patch-Level 实验"
echo "=============================================="
echo "GPU列表:     ${GPUS[*]}"
echo "并行数:      $PARALLEL_COUNT"
echo "类别:        $CATEGORY${CUSTOM_EXPS_STR:+ (自定义: $CUSTOM_EXPS_STR)}"
echo "种子列表:    ${SEEDS[*]}"
echo "实验配置数:  ${#SELECTED_EXPS[@]}"
echo "总任务数:    $TOTAL"
echo "环境:        ${CONDA_ENV}"
echo "结果目录:    ${RESULT_DIR}/Baseline/"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "没有要运行的实验"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "将运行以下实验任务:"
    echo ""
    printf "  %-4s %-35s %-6s %s\n" "#" "实验名称" "种子" "训练脚本"
    echo "  ---- ----------------------------------- ------ -------------------------"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        exp="${task%%:*}"
        seed="${task##*:}"
        printf "  %-4d %-35s %-6s %s\n" "$((i+1))" "$exp" "$seed" "train.py"
    done
    echo ""
    echo "共 $TOTAL 个任务 (预览模式, 未实际运行)"
    exit 0
fi

# ============================================================================
# GPU 管理函数
# ============================================================================
get_free_gpu() {
    local gpu
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
    local f
    for f in "${STATUS_DIR}"/exp_*.status; do
        if [ -f "$f" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
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
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 缺失 Patch-Level 实验监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT | 环境: ${CONDA_ENV}'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
WINDOW_INDEX=0

start_experiment() {
    local exp=$1
    local gpu=$2
    local seed=$3

    local task_id="${exp}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    WINDOW_INDEX=$((WINDOW_INDEX + 1))
    local window_name="e$(printf '%04d' "$WINDOW_INDEX")"

    echo "[$(date '+%H:%M:%S')] 启动: $exp (seed=$seed) -> GPU $gpu [窗口: $window_name]"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== [$window_name] $exp | GPU: $gpu | Seed: $seed ==='" Enter

    local cmd="python train.py --exp $exp --gpu $gpu --seed $seed --normalize_on_gpu 2>&1 | tee $log_file"
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

while [ $QUEUE_INDEX -lt $TOTAL ]; do
    sleep 5

    FREE_GPU=$(get_free_gpu) || continue

    task="${TASK_LIST[$QUEUE_INDEX]}"
    exp="${task%%:*}"
    seed="${task##*:}"
    start_experiment "$exp" "$FREE_GPU" "$seed"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
done

echo "所有 $TOTAL 个任务已调度"
echo ""

# ============================================================================
# 等待完成
# ============================================================================
echo "等待实验完成... (Ctrl+C 退出脚本，实验继续在 tmux 中运行)"
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
echo "  缺失 Patch-Level 实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0
FAIL_LIST=""

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
            FAIL_LIST="${FAIL_LIST}  [失败] $exp (seed=$seed)\n"
        fi
    else
        FAIL_LIST="${FAIL_LIST}  [未知] $exp (seed=$seed)\n"
    fi
done

if [ -n "$FAIL_LIST" ]; then
    echo -e "$FAIL_LIST"
fi

echo ""
echo "成功: $SUCCESS | 失败: $FAILED | 总计: $TOTAL"
echo "结果目录: ${RESULT_DIR}/Baseline/"
echo "日志目录: $LOG_DIR"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
