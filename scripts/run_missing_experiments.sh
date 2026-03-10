#!/bin/bash
# ============================================================================
# 补充实验运行脚本 - P0/P1/P2 全部缺失实验
# run_missing_experiments.sh
#
# 目的: 运行论文完成所需的所有缺失实验
#
# P0 必做 (18 runs = 6 experiments × 3 seeds):
#   SimCLR + Concat + City   - SimCLR城市级基准(配对Patch)
#   SimCLR + FiLM + City     - SimCLR+FiLM城市级(配对Patch)
#   SimCLR + Gated + City    - SimCLR+Gated城市级(配对Patch)
#   SimCLR + FiLM + Patch    - 预测最优组合
#   SimCLR + Gated + Patch   - 融合策略对比
#   MAE + FiLM + Patch       - 预训练方法对比
#
# P1 推荐 (6 runs = 2 experiments × 3 seeds):
#   MAE + Gated + Patch      - 完善消融矩阵
#   Attention + Patch         - 补全融合对比
#
# P2 可选 (18 runs = 6 experiments × 3 seeds):
#   BERT + {Concat,FiLM,Gated} + Patch   - BERT政策Patch-Level
#   Hybrid + {Concat,FiLM,Gated} + Patch - 混合政策Patch-Level
#
# 总计: 42 runs (14 experiments × 3 seeds)
#
# 使用示例:
#   bash run_missing_experiments.sh --gpus 0,1,2,3 --resume
#   bash run_missing_experiments.sh --category p0 --dry-run
#   bash run_missing_experiments.sh --category p0,p1 --gpus 0,1,2,3
#   bash run_missing_experiments.sh --list
# ============================================================================

set -e

# 项目配置
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/missing_experiments"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_missing"
SESSION_NAME="missing_exp"

# Conda 环境
CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth12"

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
# 格式: "train_script:exp_name:result_subdir:description"
#
# train_script: train_multimodal.py / train_multimodal_bert.py
# exp_name: 实验预设名 (对应 config_multimodal.py 或 config_multimodal_bert.py)
# result_subdir: 结果文件存放子目录 (空=根目录)
# description: 简短描述
# ============================================================================

# --- P0: 必做实验 (SimCLR城市级基准 + 预测最优组合) ---
P0_EXPERIMENTS=(
    # SimCLR 城市级基准 (配对Patch实验，用于预训练×Patch交互分析)
    # "train_multimodal_bert.py:mm_simclr_cnn_concat:Multimodal:SimCLR+Concat+City_配对基准"
    # "train_multimodal_bert.py:mm_simclr_cnn_film:Multimodal:SimCLR+FiLM+City_配对基准"
    # "train_multimodal_bert.py:mm_simclr_cnn_gated:Multimodal:SimCLR+Gated+City_配对基准"
    # SimCLR/MAE Patch-level 补充
    "train_multimodal_bert.py:mm_simclr_cnn_film_patch:Multimodal:SimCLR+FiLM+Patch_预测最优"
    "train_multimodal_bert.py:mm_simclr_cnn_gated_patch:Multimodal:SimCLR+Gated+Patch_融合对比"
    "train_multimodal_bert.py:mm_mae_cnn_film_patch:Multimodal:MAE+FiLM+Patch_预训练对比"
)

# --- P1: 推荐实验 (完善消融矩阵) ---
P1_EXPERIMENTS=(
    "train_multimodal_bert.py:mm_mae_cnn_gated_patch:Multimodal:MAE+Gated+Patch_消融补全"
    "train_multimodal_bert.py:mm_cnn_attention_patch:Multimodal:Attention+Patch_融合补全"
)

# --- P2: 可选实验 (BERT/Hybrid Patch-Level) ---
P2_EXPERIMENTS=(
    "train_multimodal_bert.py:bert_cnn_concat_patch:MultimodalBert:BERT+Concat+Patch"
    "train_multimodal_bert.py:bert_cnn_film_patch:MultimodalBert:BERT+FiLM+Patch"
    "train_multimodal_bert.py:bert_cnn_gated_patch:MultimodalBert:BERT+Gated+Patch"
    "train_multimodal_bert.py:hybrid_cnn_concat_patch:MultimodalHybrid:Hybrid+Concat+Patch"
    "train_multimodal_bert.py:hybrid_cnn_film_patch:MultimodalHybrid:Hybrid+FiLM+Patch"
    "train_multimodal_bert.py:hybrid_cnn_gated_patch:MultimodalHybrid:Hybrid+Gated+Patch"
)

# 合并
ALL_EXPERIMENTS=("${P0_EXPERIMENTS[@]}" "${P1_EXPERIMENTS[@]}" "${P2_EXPERIMENTS[@]}")

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)

# ============================================================================
# 帮助
# ============================================================================
show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** 补充实验运行脚本 - 论文所需全部缺失实验 **"
    echo ""
    echo "优先级:"
    echo "  P0 必做 (${#P0_EXPERIMENTS[@]} 实验 × 3 seeds = $(( ${#P0_EXPERIMENTS[@]} * 3 )) runs)"
    echo "    SimCLR+{Concat,FiLM,Gated}+City / SimCLR+{FiLM,Gated}+Patch / MAE+FiLM+Patch"
    echo "  P1 推荐 (${#P1_EXPERIMENTS[@]} 实验 × 3 seeds = $(( ${#P1_EXPERIMENTS[@]} * 3 )) runs)"
    echo "    MAE+Gated+Patch / Attention+Patch"
    echo "  P2 可选 (${#P2_EXPERIMENTS[@]} 实验 × 3 seeds = $(( ${#P2_EXPERIMENTS[@]} * 3 )) runs)"
    echo "    BERT+{Concat,FiLM,Gated}+Patch / Hybrid+{Concat,FiLM,Gated}+Patch"
    echo ""
    echo "选项:"
    echo "  --help, -h        显示帮助"
    echo "  --list, -l        列出所有实验配置"
    echo "  --gpus GPUS       指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N      并行数量 (默认=GPU数)"
    echo "  --category CAT    实验类别，逗号分隔 (p0/p1/p2/all，默认: all)"
    echo "  --seeds S1,S2     种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --dry-run         预览模式，不实际运行"
    echo "  --resume          跳过已完成实验"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 0,1,2,3 --resume                # 所有实验，跳过已完成"
    echo "  bash $0 --category p0 --gpus 0,1,2,3            # 只运行P0"
    echo "  bash $0 --category p0,p1 --dry-run              # 预览P0+P1"
    echo "  bash $0 --category p2 --gpus 4,5,6,7 --resume   # 只运行P2，跳过已完成"
}

list_experiments() {
    echo "=============================================="
    echo "  补充实验对照表"
    echo "=============================================="
    echo ""
    echo "=== P0 必做 (预测最优组合) ==="
    for exp_info in "${P0_EXPERIMENTS[@]}"; do
        IFS=':' read -r script exp_name subdir desc <<< "$exp_info"
        printf "  %-35s %-30s %s\n" "$exp_name" "$script" "$desc"
    done
    echo ""
    echo "=== P1 推荐 (完善消融矩阵) ==="
    for exp_info in "${P1_EXPERIMENTS[@]}"; do
        IFS=':' read -r script exp_name subdir desc <<< "$exp_info"
        printf "  %-35s %-30s %s\n" "$exp_name" "$script" "$desc"
    done
    echo ""
    echo "=== P2 可选 (BERT/Hybrid Patch-Level) ==="
    for exp_info in "${P2_EXPERIMENTS[@]}"; do
        IFS=':' read -r script exp_name subdir desc <<< "$exp_info"
        printf "  %-35s %-30s %s\n" "$exp_name" "$script" "$desc"
    done
    echo ""
    echo "总计: ${#ALL_EXPERIMENTS[@]} 实验配置 × ${#DEFAULT_SEEDS[@]} seeds = $(( ${#ALL_EXPERIMENTS[@]} * ${#DEFAULT_SEEDS[@]} )) 任务"
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
CATEGORY_STR="all"
SEEDS=("${DEFAULT_SEEDS[@]}")
DRY_RUN=false
RESUME=false

while [ $# -gt 0 ]; do
    case $1 in
        --help|-h) show_help; exit 0 ;;
        --list|-l) list_experiments; exit 0 ;;
        --gpus) IFS=',' read -ra GPUS <<< "$2"; shift 2 ;;
        --parallel) PARALLEL_COUNT=$2; shift 2 ;;
        --category) CATEGORY_STR=$2; shift 2 ;;
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
# 筛选实验（支持逗号分隔多类别）
# ============================================================================
SELECTED_EXPERIMENTS=()

IFS=',' read -ra CATEGORIES <<< "$CATEGORY_STR"
for cat in "${CATEGORIES[@]}"; do
    case $cat in
        p0) SELECTED_EXPERIMENTS+=("${P0_EXPERIMENTS[@]}") ;;
        p1) SELECTED_EXPERIMENTS+=("${P1_EXPERIMENTS[@]}") ;;
        p2) SELECTED_EXPERIMENTS+=("${P2_EXPERIMENTS[@]}") ;;
        all) SELECTED_EXPERIMENTS=("${ALL_EXPERIMENTS[@]}"); break ;;
        *) echo "无效的类别: $cat (有效: p0, p1, p2, all)"; exit 1 ;;
    esac
done

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
        # 解析 script:exp_name:subdir:desc:seed
        IFS=':' read -r script exp_name subdir desc seed <<< "$task"

        # 构建 run_id
        if [ "$seed" == "42" ]; then
            run_id="${exp_name}"
        else
            run_id="${exp_name}_seed${seed}"
        fi

        # 构建结果文件路径
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
echo "  补充实验运行器 (P0/P1/P2)"
echo "=============================================="
echo "GPU列表: ${GPUS[*]}"
echo "并行数: $PARALLEL_COUNT"
echo "类别筛选: $CATEGORY_STR"
echo "种子列表: ${SEEDS[*]}"
echo "总任务数: $TOTAL"
echo "=============================================="
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "没有要运行的实验 (所有实验已完成或未选择任何类别)"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "将运行以下实验任务:"
    echo ""
    printf "  %-4s %-35s %-6s %-30s %s\n" "编号" "实验名称" "种子" "训练脚本" "描述"
    echo "  ---- ----------------------------------- ------ ------------------------------ --------------------"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        IFS=':' read -r script exp_name subdir desc seed <<< "$task"
        printf "  %-4d %-35s %-6s %-30s %s\n" "$((i+1))" "$exp_name" "$seed" "$script" "$desc"
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
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 补充实验监控 (P0/P1/P2) ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Ctrl+B w 查看窗口 | Ctrl+B d 脱离'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
start_experiment() {
    local script=$1
    local exp_name=$2
    local subdir=$3
    local desc=$4
    local gpu=$5
    local seed=$6

    local task_id="${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"
    local window_name="${exp_name}_s${seed}"

    echo "[$(date '+%H:%M:%S')] 启动: $exp_name (seed=$seed) -> GPU $gpu [$script]"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH && echo '环境已激活: ${CONDA_ENV}'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== $exp_name | GPU: $gpu | Seed: $seed | $desc ==='" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '开始时间:' \$(date)" Enter

    # 运行实验
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
    IFS=':' read -r script exp_name subdir desc seed <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$script" "$exp_name" "$subdir" "$desc" "$gpu" "$seed"
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
    IFS=':' read -r script exp_name subdir desc seed <<< "$task"
    start_experiment "$script" "$exp_name" "$subdir" "$desc" "$FREE_GPU" "$seed"
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
echo "  补充实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r script exp_name subdir desc seed <<< "$task"
    task_id="${exp_name}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  [失败] $exp_name (seed=$seed)"
        fi
    else
        echo "  [未知] $exp_name (seed=$seed)"
    fi
done

echo ""
echo "成功: $SUCCESS | 失败: $FAILED"
echo "日志目录: $LOG_DIR"
echo ""
echo "实验矩阵:"
echo "  ┌──────────┬──────────────────────────────┬──────────────────────────────────┐"
echo "  │ 优先级    │ 实验名称                      │ 描述                             │"
echo "  ├──────────┼──────────────────────────────┼──────────────────────────────────┤"
echo "  │ P0 必做  │ mm_simclr_cnn_concat         │ SimCLR City基准 (配对Patch)       │"
echo "  │          │ mm_simclr_cnn_film            │ SimCLR+FiLM City (配对Patch)      │"
echo "  │          │ mm_simclr_cnn_gated           │ SimCLR+Gated City (配对Patch)     │"
echo "  │          │ mm_simclr_cnn_film_patch      │ SimCLR+FiLM+Patch 预测最优        │"
echo "  │          │ mm_simclr_cnn_gated_patch     │ SimCLR+Gated+Patch 融合对比       │"
echo "  │          │ mm_mae_cnn_film_patch         │ MAE+FiLM+Patch 预训练对比         │"
echo "  ├──────────┼──────────────────────────────┼──────────────────────────────────┤"
echo "  │ P1 推荐  │ mm_mae_cnn_gated_patch       │ MAE+Gated+Patch 消融补全          │"
echo "  │          │ mm_cnn_attention_patch        │ Attention+Patch 融合补全          │"
echo "  ├──────────┼──────────────────────────────┼──────────────────────────────────┤"
echo "  │ P2 可选  │ bert_cnn_{concat,film,gated} │ BERT政策+Patch-Level (3种融合)    │"
echo "  │          │   _patch                      │                                  │"
echo "  │          │ hybrid_cnn_{concat,film,gated}│ 混合政策+Patch-Level (3种融合)    │"
echo "  │          │   _patch                      │                                  │"
echo "  └──────────┴──────────────────────────────┴──────────────────────────────────┘"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
