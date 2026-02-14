#!/bin/bash
# ============================================================================
# 完整政策消融实验脚本 (All Experiments)
# run_policy_ablation_all_experiments.sh
#
# 目的: 将 run_mm_simple_gpu.sh 中所有56个实验配置扩展到4种政策输入方式,
#       构建完整的政策特征消融实验矩阵
#
# 政策维度 (4种):
#   1. none       - 单模态基线 (仅卫星影像, 0维)     → train.py
#   2. structured - 结构化政策特征 (12维)             → train_multimodal_bert.py
#   3. bert       - BERT政策嵌入 (64维)               → train_multimodal_bert.py
#   4. hybrid     - 混合特征 (76维 = BERT64 + 结构化12) → train_multimodal_bert.py
#
# 结果目录:
#   none       → results/Baseline/
#   structured → results/Multimodal/
#   bert       → results/MultimodalBert/
#   hybrid     → results/MultimodalHybrid/
#
# 规模: 200配置 (32 none去重 + 56×3 多模态) × 3 seeds = 600 实验
#
# 使用示例:
#   bash run_policy_ablation_all_experiments.sh --gpus 0,1,2,3 --resume
#   bash run_policy_ablation_all_experiments.sh --policy bert,hybrid --category simclr --dry-run
#   bash run_policy_ablation_all_experiments.sh --policy none --category resnet_all --gpus 0,1
#   bash run_policy_ablation_all_experiments.sh --list
#   bash run_policy_ablation_all_experiments.sh --count
# ============================================================================

set -e

# ============================================================================
# 项目配置
# ============================================================================
PROJECT_DIR="/share_data/data101/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain"
LOG_DIR="${PROJECT_DIR}/logs/ablation_policy_all"
RESULT_DIR="${PROJECT_DIR}/results"
STATUS_DIR="${PROJECT_DIR}/.run_status_policy_ablation_all"
SESSION_NAME="policy_ablation_all"

CONDA_BASE="/share_data/data101/xiaozhenyu/anaconda3"
CONDA_ENV="alphaearth"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "${RESULT_DIR}/Baseline"
mkdir -p "${RESULT_DIR}/Multimodal"
mkdir -p "${RESULT_DIR}/MultimodalBert"
mkdir -p "${RESULT_DIR}/MultimodalHybrid"
mkdir -p "$STATUS_DIR"

# ============================================================================
# 源实验定义 (来自 run_mm_simple_gpu.sh 的56个实验)
#
# 格式: "mm_exp_name:baseline_exp_name"
#   mm_exp_name:      多模态实验名 (structured/bert/hybrid 通过前缀替换)
#   baseline_exp_name: 对应的单模态基线名 (none模式), "NONE"=无对应基线
#
# 命名规则:
#   structured → mm_exp_name (原名)
#   bert       → mm_ 替换为 bert_
#   hybrid     → mm_ 替换为 hybrid_
#   none       → baseline_exp_name
# ============================================================================

# --- LightCNN + Concat (基础组, 3个) ---
CNN_CONCAT_SRC=(
    "mm_cnn_concat:light_cnn_baseline"
    "mm_cnn_concat_median:light_cnn_median"
    "mm_cnn_concat_trimmed:light_cnn_trimmed_mean"
)

# --- LightCNN 融合变体 (6个) ---
CNN_FUSION_SRC=(
    "mm_cnn_gated:light_cnn_baseline"
    "mm_cnn_gated_trimmed:light_cnn_trimmed_mean"
    "mm_cnn_attention:light_cnn_baseline"
    "mm_cnn_attention_trimmed:light_cnn_trimmed_mean"
    "mm_cnn_film:light_cnn_baseline"
    "mm_cnn_film_trimmed:light_cnn_trimmed_mean"
)

# --- MLP (2个) ---
MLP_SRC=(
    "mm_mlp_concat:mlp_baseline"
    "mm_mlp_gated:mlp_baseline"
)

# --- ResNet18 (3个) ---
RESNET18_SRC=(
    "mm_resnet18_concat:resnet18_baseline"
    "mm_resnet18_gated:resnet18_baseline"
    "mm_resnet18_film:resnet18_baseline"
)

# --- ResNet34 (4个) ---
RESNET34_SRC=(
    "mm_resnet34_concat:resnet34_baseline"
    "mm_resnet34_gated:resnet34_baseline"
    "mm_resnet34_film:resnet34_baseline"
    "mm_resnet34_pretrained:resnet34_imagenet"
)

# --- ResNet10 (2个) ---
RESNET10_SRC=(
    "mm_resnet10_concat:resnet10_baseline"
    "mm_resnet10_gated:resnet10_baseline"
)

# --- ResNet50 (3个) ---
RESNET50_SRC=(
    "mm_resnet50_concat:resnet50_baseline"
    "mm_resnet50_gated:resnet50_baseline"
    "mm_resnet50_imagenet:resnet50_imagenet"
)

# --- ResNet101 (2个) ---
RESNET101_SRC=(
    "mm_resnet101_concat:resnet101_baseline"
    "mm_resnet101_imagenet:resnet101_imagenet"
)

# --- Patch-level (4个) ---
PATCH_SRC=(
    "mm_cnn_concat_patch:light_cnn_patch_level"
    "mm_cnn_gated_patch:light_cnn_patch_level"
    "mm_cnn_film_patch:light_cnn_patch_level"
    "mm_resnet18_concat_patch:resnet18_patch_level"
)

# --- 自定义模型 (1个, 无对应基线) ---
CUSTOM_SRC=(
    "mm_cnn_small_concat:NONE"
)

# --- LightCNN Concat 位置感知聚合 (5个) ---
CNN_AGG_SRC=(
    "mm_cnn_concat_attn_agg:light_cnn_attention"
    "mm_cnn_concat_pos_attn:light_cnn_pos_attention"
    "mm_cnn_concat_spatial_attn:light_cnn_spatial_attention"
    "mm_cnn_concat_transformer:light_cnn_transformer"
    "mm_cnn_concat_transformer_2d:light_cnn_transformer_2d"
)

# --- LightCNN Gated 位置感知聚合 (2个) ---
CNN_GATED_AGG_SRC=(
    "mm_cnn_gated_transformer:light_cnn_transformer"
    "mm_cnn_gated_transformer_2d:light_cnn_transformer_2d"
)

# --- LightCNN FiLM 位置感知聚合 (2个) ---
CNN_FILM_AGG_SRC=(
    "mm_cnn_film_transformer:light_cnn_transformer"
    "mm_cnn_film_transformer_2d:light_cnn_transformer_2d"
)

# --- ResNet18 位置感知聚合 (2个) ---
RESNET18_AGG_SRC=(
    "mm_resnet18_concat_transformer:resnet18_transformer"
    "mm_resnet18_concat_transformer_2d:resnet18_transformer_2d"
)

# --- SimCLR + LightCNN (4个) ---
SIMCLR_CNN_SRC=(
    "mm_simclr_cnn_concat:simclr_cnn"
    "mm_simclr_cnn_concat_trimmed:simclr_cnn_trimmed_mean"
    "mm_simclr_cnn_gated:simclr_cnn"
    "mm_simclr_cnn_film:simclr_cnn"
)

# --- SimCLR + LightCNN 聚合 (2个) ---
SIMCLR_CNN_AGG_SRC=(
    "mm_simclr_cnn_transformer:simclr_cnn_transformer"
    "mm_simclr_cnn_transformer_2d:simclr_cnn_transformer_2d"
)

# --- SimCLR + MLP (1个) ---
SIMCLR_MLP_SRC=(
    "mm_simclr_mlp_concat:simclr_mlp"
)

# --- SimCLR + Patch (1个) ---
SIMCLR_PATCH_SRC=(
    "mm_simclr_cnn_concat_patch:simclr_cnn_patch_level"
)

# --- MAE + LightCNN (4个) ---
MAE_CNN_SRC=(
    "mm_mae_cnn_concat:mae_cnn"
    "mm_mae_cnn_concat_trimmed:mae_cnn_trimmed_mean"
    "mm_mae_cnn_gated:mae_cnn"
    "mm_mae_cnn_film:mae_cnn"
)

# --- MAE + LightCNN 聚合 (2个) ---
MAE_CNN_AGG_SRC=(
    "mm_mae_cnn_transformer:mae_cnn_transformer"
    "mm_mae_cnn_transformer_2d:mae_cnn_transformer_2d"
)

# --- MAE + Patch (1个) ---
MAE_PATCH_SRC=(
    "mm_mae_cnn_concat_patch:mae_cnn_patch_level"
)

# ============================================================================
# 组合分类 (与 run_mm_simple_gpu.sh 保持一致)
# ============================================================================
RESNET_SRC=("${RESNET18_SRC[@]}" "${RESNET34_SRC[@]}")
ALL_RESNET_SRC=("${RESNET10_SRC[@]}" "${RESNET18_SRC[@]}" "${RESNET34_SRC[@]}" "${RESNET50_SRC[@]}" "${RESNET101_SRC[@]}")
ALL_FUSION_SRC=("${CNN_CONCAT_SRC[@]}" "${CNN_FUSION_SRC[@]}")
ALL_MODELS_SRC=("${CNN_CONCAT_SRC[@]}" "${MLP_SRC[@]}" "${RESNET_SRC[@]}")
ALL_PATCH_SRC=("${PATCH_SRC[@]}" "${SIMCLR_PATCH_SRC[@]}" "${MAE_PATCH_SRC[@]}")
ALL_AGG_SRC=("${CNN_AGG_SRC[@]}" "${CNN_GATED_AGG_SRC[@]}" "${CNN_FILM_AGG_SRC[@]}" "${RESNET18_AGG_SRC[@]}")
ALL_SIMCLR_SRC=("${SIMCLR_CNN_SRC[@]}" "${SIMCLR_CNN_AGG_SRC[@]}" "${SIMCLR_MLP_SRC[@]}" "${SIMCLR_PATCH_SRC[@]}")
ALL_MAE_SRC=("${MAE_CNN_SRC[@]}" "${MAE_CNN_AGG_SRC[@]}" "${MAE_PATCH_SRC[@]}")
ALL_SSL_SRC=("${ALL_SIMCLR_SRC[@]}" "${ALL_MAE_SRC[@]}")

# 全部源实验 (顺序与 run_mm_simple_gpu.sh ALL_EXPERIMENTS 一致)
ALL_SOURCE=(
    "${CNN_CONCAT_SRC[@]}"
    "${CNN_FUSION_SRC[@]}"
    "${MLP_SRC[@]}"
    "${RESNET_SRC[@]}"
    "${PATCH_SRC[@]}"
    "${CUSTOM_SRC[@]}"
    "${CNN_AGG_SRC[@]}"
    "${CNN_GATED_AGG_SRC[@]}"
    "${CNN_FILM_AGG_SRC[@]}"
    "${RESNET18_AGG_SRC[@]}"
    "${SIMCLR_CNN_SRC[@]}"
    "${SIMCLR_CNN_AGG_SRC[@]}"
    "${SIMCLR_MLP_SRC[@]}"
    "${SIMCLR_PATCH_SRC[@]}"
    "${MAE_CNN_SRC[@]}"
    "${MAE_CNN_AGG_SRC[@]}"
    "${MAE_PATCH_SRC[@]}"
    "${RESNET10_SRC[@]}"
    "${RESNET50_SRC[@]}"
    "${RESNET101_SRC[@]}"
)

DEFAULT_GPUS=(0 1 2 3 4 5 6 7)
DEFAULT_SEEDS=(42 123 456)
DEFAULT_POLICIES=(none structured bert hybrid)

# ============================================================================
# 辅助函数
# ============================================================================
_list_group() {
    local title=$1; shift
    local entries=("$@")
    echo "=== $title (${#entries[@]}) ==="
    for src in "${entries[@]}"; do
        IFS=':' read -r mm_name baseline <<< "$src"
        printf "  %-40s → none: %s\n" "$mm_name" "$baseline"
    done
    echo ""
}

show_help() {
    echo "用法: bash $0 [选项]"
    echo ""
    echo "** 完整政策消融实验 - 56个实验配置 × 4种政策输入方式 **"
    echo ""
    echo "选项:"
    echo "  --help, -h           显示帮助"
    echo "  --list, -l           列出所有源实验及基线映射"
    echo "  --count              统计各维度实验数量"
    echo "  --gpus GPUS          指定GPU (例如: 0,1,2,3)"
    echo "  --parallel N         并行数量 (默认=GPU数)"
    echo "  --policy P1,P2,...   政策模式 (none,structured,bert,hybrid)"
    echo "  --category CAT       实验类别 (见下方)"
    echo "  --exp EXP1,EXP2      指定源实验 (mm_前缀名)"
    echo "  --seeds S1,S2,...    种子列表 (默认: ${DEFAULT_SEEDS[*]})"
    echo "  --seed SEED          单个种子"
    echo "  --dry-run            预览模式"
    echo "  --resume             跳过已完成实验"
    echo ""
    echo "可用类别 (--category):"
    echo "  all          全部 (${#ALL_SOURCE[@]})"
    echo "  baseline     LightCNN+Concat (${#CNN_CONCAT_SRC[@]})"
    echo "  fusion       LightCNN融合变体 (${#ALL_FUSION_SRC[@]})"
    echo "  models       MLP/CNN/ResNet18/34 (${#ALL_MODELS_SRC[@]})"
    echo "  mlp          MLP (${#MLP_SRC[@]})"
    echo "  resnet       ResNet18+34 (${#RESNET_SRC[@]})"
    echo "  resnet_all   所有ResNet深度 (${#ALL_RESNET_SRC[@]})"
    echo "  patch        Patch-level (${#ALL_PATCH_SRC[@]})"
    echo "  agg          位置感知聚合 (${#ALL_AGG_SRC[@]})"
    echo "  simclr       SimCLR预训练 (${#ALL_SIMCLR_SRC[@]})"
    echo "  mae          MAE预训练 (${#ALL_MAE_SRC[@]})"
    echo "  ssl          所有自监督 (${#ALL_SSL_SRC[@]})"
    echo "  custom       自定义模型 (${#CUSTOM_SRC[@]})"
    echo ""
    echo "示例:"
    echo "  bash $0 --gpus 0,1,2,3 --resume"
    echo "  bash $0 --policy bert,hybrid --category fusion --dry-run"
    echo "  bash $0 --policy none --category resnet_all --seeds 42"
    echo "  bash $0 --exp mm_cnn_film,mm_simclr_cnn_film --policy none,structured,bert,hybrid"
}

list_experiments() {
    echo "=============================================="
    echo "  完整政策消融实验 - 源实验列表"
    echo "=============================================="
    echo ""
    echo "命名规则: structured=mm_* | bert=bert_* | hybrid=hybrid_* | none=baseline名"
    echo ""
    _list_group "LightCNN + Concat (基础)" "${CNN_CONCAT_SRC[@]}"
    _list_group "LightCNN 融合变体" "${CNN_FUSION_SRC[@]}"
    _list_group "MLP" "${MLP_SRC[@]}"
    _list_group "ResNet18" "${RESNET18_SRC[@]}"
    _list_group "ResNet34" "${RESNET34_SRC[@]}"
    _list_group "ResNet10" "${RESNET10_SRC[@]}"
    _list_group "ResNet50" "${RESNET50_SRC[@]}"
    _list_group "ResNet101" "${RESNET101_SRC[@]}"
    _list_group "Patch-level" "${PATCH_SRC[@]}"
    _list_group "自定义模型" "${CUSTOM_SRC[@]}"
    _list_group "LightCNN Concat 聚合" "${CNN_AGG_SRC[@]}"
    _list_group "LightCNN Gated 聚合" "${CNN_GATED_AGG_SRC[@]}"
    _list_group "LightCNN FiLM 聚合" "${CNN_FILM_AGG_SRC[@]}"
    _list_group "ResNet18 聚合" "${RESNET18_AGG_SRC[@]}"
    _list_group "SimCLR + LightCNN" "${SIMCLR_CNN_SRC[@]}"
    _list_group "SimCLR + LightCNN 聚合" "${SIMCLR_CNN_AGG_SRC[@]}"
    _list_group "SimCLR + MLP" "${SIMCLR_MLP_SRC[@]}"
    _list_group "SimCLR + Patch" "${SIMCLR_PATCH_SRC[@]}"
    _list_group "MAE + LightCNN" "${MAE_CNN_SRC[@]}"
    _list_group "MAE + LightCNN 聚合" "${MAE_CNN_AGG_SRC[@]}"
    _list_group "MAE + Patch" "${MAE_PATCH_SRC[@]}"

    # Count unique baselines
    declare -A _seen
    local _none_count=0
    for src in "${ALL_SOURCE[@]}"; do
        local _bl="${src##*:}"
        if [ "$_bl" != "NONE" ] && [ -z "${_seen[$_bl]}" ]; then
            _seen[$_bl]=1
            _none_count=$((_none_count + 1))
        fi
    done
    echo "=== 统计 ==="
    echo "  源实验: ${#ALL_SOURCE[@]}"
    echo "  none (去重): $_none_count"
    echo "  structured:  ${#ALL_SOURCE[@]}"
    echo "  bert:        ${#ALL_SOURCE[@]}"
    echo "  hybrid:      ${#ALL_SOURCE[@]}"
    echo "  总配置: $((_none_count + ${#ALL_SOURCE[@]} * 3))"
    echo "  × ${#DEFAULT_SEEDS[@]} seeds = $(((_none_count + ${#ALL_SOURCE[@]} * 3) * ${#DEFAULT_SEEDS[@]})) 个实验"
}

count_experiments() {
    # Count unique baselines
    declare -A _seen
    local _none_count=0
    for src in "${ALL_SOURCE[@]}"; do
        local _bl="${src##*:}"
        if [ "$_bl" != "NONE" ] && [ -z "${_seen[$_bl]}" ]; then
            _seen[$_bl]=1
            _none_count=$((_none_count + 1))
        fi
    done

    echo "=============================================="
    echo "  实验数量统计"
    echo "=============================================="
    echo ""
    echo "  按政策模式:"
    echo "    none (基线):   $_none_count 个 (去重)"
    echo "    structured:    ${#ALL_SOURCE[@]} 个"
    echo "    bert:          ${#ALL_SOURCE[@]} 个"
    echo "    hybrid:        ${#ALL_SOURCE[@]} 个"
    echo "    合计配置:      $((_none_count + ${#ALL_SOURCE[@]} * 3)) 个"
    echo "    × ${#DEFAULT_SEEDS[@]} seeds = $(((_none_count + ${#ALL_SOURCE[@]} * 3) * ${#DEFAULT_SEEDS[@]})) 个实验"
    echo ""
    echo "  按类别 (源实验数 × 4政策 × 3seeds):"
    printf "    %-15s %3d 源 → %4d 任务\n" "baseline"    "${#CNN_CONCAT_SRC[@]}" "$((${#CNN_CONCAT_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "fusion"      "${#ALL_FUSION_SRC[@]}" "$((${#ALL_FUSION_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "mlp"         "${#MLP_SRC[@]}"        "$((${#MLP_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "resnet"      "${#RESNET_SRC[@]}"     "$((${#RESNET_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "resnet_all"  "${#ALL_RESNET_SRC[@]}" "$((${#ALL_RESNET_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "patch"       "${#ALL_PATCH_SRC[@]}"  "$((${#ALL_PATCH_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "agg"         "${#ALL_AGG_SRC[@]}"    "$((${#ALL_AGG_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "simclr"      "${#ALL_SIMCLR_SRC[@]}" "$((${#ALL_SIMCLR_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "mae"         "${#ALL_MAE_SRC[@]}"    "$((${#ALL_MAE_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "ssl"         "${#ALL_SSL_SRC[@]}"    "$((${#ALL_SSL_SRC[@]} * 4 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "custom"      "${#CUSTOM_SRC[@]}"     "$((${#CUSTOM_SRC[@]} * 3 * ${#DEFAULT_SEEDS[@]}))"
    printf "    %-15s %3d 源 → %4d 任务\n" "all"         "${#ALL_SOURCE[@]}"     "$(( (_none_count + ${#ALL_SOURCE[@]} * 3) * ${#DEFAULT_SEEDS[@]}))"
    echo ""
    echo "  注: none模式多对一映射 (如 mm_cnn_concat/gated/film 共享 light_cnn_baseline)"
    echo "      custom (mm_cnn_small_concat) 无 none 基线, 仅3种多模态模式"
}

# ============================================================================
# 解析参数
# ============================================================================
GPUS=("${DEFAULT_GPUS[@]}")
PARALLEL_COUNT=0
CATEGORY="all"
CUSTOM_EXPS_STR=""
POLICY_STR=""
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
        --policy) POLICY_STR=$2; shift 2 ;;
        --seeds) IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
    esac
done

# 并行数
if [ "$PARALLEL_COUNT" -eq 0 ]; then
    PARALLEL_COUNT=${#GPUS[@]}
fi
if [ "$PARALLEL_COUNT" -gt "${#GPUS[@]}" ]; then
    PARALLEL_COUNT=${#GPUS[@]}
fi

# 政策列表
if [ -n "$POLICY_STR" ]; then
    IFS=',' read -ra POLICY_LIST <<< "$POLICY_STR"
else
    POLICY_LIST=("${DEFAULT_POLICIES[@]}")
fi

# 验证政策名
for p in "${POLICY_LIST[@]}"; do
    case $p in
        none|structured|bert|hybrid) ;;
        *) echo "无效政策模式: $p (有效: none, structured, bert, hybrid)"; exit 1 ;;
    esac
done

# ============================================================================
# 选择源实验
# ============================================================================
if [ -n "$CUSTOM_EXPS_STR" ]; then
    IFS=',' read -ra CUSTOM_EXPS <<< "$CUSTOM_EXPS_STR"
    SELECTED_SRC=()
    for src in "${ALL_SOURCE[@]}"; do
        mm_name="${src%%:*}"
        for exp in "${CUSTOM_EXPS[@]}"; do
            if [ "$mm_name" == "$exp" ]; then
                SELECTED_SRC+=("$src")
                break
            fi
        done
    done
    if [ ${#SELECTED_SRC[@]} -eq 0 ]; then
        echo "错误: 未找到匹配的源实验: $CUSTOM_EXPS_STR"
        echo "提示: 使用 mm_ 前缀名 (例如: mm_cnn_concat), 运行 --list 查看全部"
        exit 1
    fi
else
    case $CATEGORY in
        all) SELECTED_SRC=("${ALL_SOURCE[@]}") ;;
        baseline) SELECTED_SRC=("${CNN_CONCAT_SRC[@]}") ;;
        fusion) SELECTED_SRC=("${ALL_FUSION_SRC[@]}") ;;
        models) SELECTED_SRC=("${ALL_MODELS_SRC[@]}") ;;
        mlp) SELECTED_SRC=("${MLP_SRC[@]}") ;;
        resnet) SELECTED_SRC=("${RESNET_SRC[@]}") ;;
        resnet_all) SELECTED_SRC=("${ALL_RESNET_SRC[@]}") ;;
        patch) SELECTED_SRC=("${ALL_PATCH_SRC[@]}") ;;
        agg) SELECTED_SRC=("${ALL_AGG_SRC[@]}") ;;
        simclr) SELECTED_SRC=("${ALL_SIMCLR_SRC[@]}") ;;
        mae) SELECTED_SRC=("${ALL_MAE_SRC[@]}") ;;
        ssl) SELECTED_SRC=("${ALL_SSL_SRC[@]}") ;;
        custom) SELECTED_SRC=("${CUSTOM_SRC[@]}") ;;
        *) echo "未知类别: $CATEGORY"; show_help; exit 1 ;;
    esac
fi

# ============================================================================
# 展开到政策变体 (含 none 模式去重)
# ============================================================================
declare -A NONE_SEEN
EXPERIMENT_LIST=()   # 格式: "policy:script:exp_name:subdir"

# 命名例外: config_multimodal_bert.py 中部分实验的 bert_/hybrid_ 名称
# 与简单前缀替换不一致 (mm_cnn_concat_transformer → bert_cnn_transformer, 而非 bert_cnn_concat_transformer)
declare -A BERT_NAME_OVERRIDE
declare -A HYBRID_NAME_OVERRIDE
BERT_NAME_OVERRIDE["mm_cnn_concat_transformer"]="bert_cnn_transformer"
BERT_NAME_OVERRIDE["mm_cnn_concat_transformer_2d"]="bert_cnn_transformer_2d"
HYBRID_NAME_OVERRIDE["mm_cnn_concat_transformer"]="hybrid_cnn_transformer"
HYBRID_NAME_OVERRIDE["mm_cnn_concat_transformer_2d"]="hybrid_cnn_transformer_2d"

for src in "${SELECTED_SRC[@]}"; do
    IFS=':' read -r mm_name baseline_name <<< "$src"

    for policy in "${POLICY_LIST[@]}"; do
        case $policy in
            none)
                [ "$baseline_name" == "NONE" ] && continue
                [ -n "${NONE_SEEN[$baseline_name]}" ] && continue
                NONE_SEEN[$baseline_name]=1
                EXPERIMENT_LIST+=("none:train.py:${baseline_name}:Baseline")
                ;;
            structured)
                EXPERIMENT_LIST+=("structured:train_multimodal_bert.py:${mm_name}:Multimodal")
                ;;
            bert)
                bert_name="${BERT_NAME_OVERRIDE[$mm_name]:-${mm_name/mm_/bert_}}"
                EXPERIMENT_LIST+=("bert:train_multimodal_bert.py:${bert_name}:MultimodalBert")
                ;;
            hybrid)
                hybrid_name="${HYBRID_NAME_OVERRIDE[$mm_name]:-${mm_name/mm_/hybrid_}}"
                EXPERIMENT_LIST+=("hybrid:train_multimodal_bert.py:${hybrid_name}:MultimodalHybrid")
                ;;
        esac
    done
done

# ============================================================================
# 生成任务列表 (实验 × 种子)
# ============================================================================
TASK_LIST=()
for exp_entry in "${EXPERIMENT_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASK_LIST+=("${exp_entry}:${seed}")
    done
done

# Resume: 跳过已完成
# 注意: train_multimodal.py (旧脚本) 将structured结果存到 results/{run_id}_results.pkl (根目录)
#       train_multimodal_bert.py (新脚本) 将结果存到 results/{Multimodal|MultimodalBert|MultimodalHybrid}/
#       train.py 将baseline结果存到 results/Baseline/
#       因此resume需要同时检查新旧两个路径
if [ "$RESUME" = true ]; then
    FILTERED=()
    SKIPPED=0
    for task in "${TASK_LIST[@]}"; do
        IFS=':' read -r policy script exp_name subdir seed <<< "$task"

        if [ "$seed" == "42" ]; then
            run_id="${exp_name}"
        else
            run_id="${exp_name}_seed${seed}"
        fi

        # 新路径 (train_multimodal_bert.py / train.py)
        result_file="${RESULT_DIR}/${subdir}/${run_id}_results.pkl"
        # 旧路径 (train_multimodal.py 将structured结果存在根目录)
        result_file_old="${RESULT_DIR}/${run_id}_results.pkl"

        if [ -f "$result_file" ] || [ -f "$result_file_old" ]; then
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
echo "  完整政策消融实验"
echo "=============================================="
echo "GPU列表:     ${GPUS[*]}"
echo "并行数:      $PARALLEL_COUNT"
echo "政策模式:    ${POLICY_LIST[*]}"
echo "类别:        ${CATEGORY}${CUSTOM_EXPS_STR:+ (自定义: $CUSTOM_EXPS_STR)}"
echo "种子列表:    ${SEEDS[*]}"
echo "实验配置数:  ${#EXPERIMENT_LIST[@]}"
echo "总任务数:    $TOTAL"
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
    echo "将运行以下实验任务:"
    echo ""
    printf "  %-4s %-12s %-40s %-6s %-15s %s\n" "#" "政策" "实验名称" "种子" "结果目录" "训练脚本"
    echo "  ---- ------------ ---------------------------------------- ------ --------------- -------------------------"
    for i in "${!TASK_LIST[@]}"; do
        task="${TASK_LIST[$i]}"
        IFS=':' read -r policy script exp_name subdir seed <<< "$task"
        printf "  %-4d %-12s %-40s %-6s %-15s %s\n" "$((i+1))" "[$policy]" "$exp_name" "$seed" "$subdir" "$script"
    done
    echo ""
    echo "共 $TOTAL 个任务 (预览模式, 未实际运行)"
    exit 0
fi

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
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== 完整政策消融实验监控 ==='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '总任务数: $TOTAL | 并行数: $PARALLEL_COUNT | 政策: ${POLICY_LIST[*]}'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 5 'echo \"=== 运行中 ===\"; cat ${STATUS_DIR}/gpu_*.lock 2>/dev/null | head -20; echo \"\"; echo \"=== 已完成: \$(ls ${STATUS_DIR}/exp_*.status 2>/dev/null | wc -l)/$TOTAL ===\"; echo \"\"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'" Enter

# ============================================================================
# 启动实验
# ============================================================================
WINDOW_INDEX=0

start_experiment() {
    local policy=$1
    local script=$2
    local exp_name=$3
    local subdir=$4
    local gpu=$5
    local seed=$6

    local task_id="${policy}_${exp_name}_seed${seed}"
    local log_file="${LOG_DIR}/${task_id}.log"

    # 使用全局递增序号作为窗口名, 避免长名称截断后冲突
    WINDOW_INDEX=$((WINDOW_INDEX + 1))
    local window_name="e$(printf '%04d' $WINDOW_INDEX)"

    echo "[$(date '+%H:%M:%S')] 启动: [$policy] $exp_name (seed=$seed) -> GPU $gpu [窗口: $window_name]"

    lock_gpu "$gpu" "$task_id"

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $PROJECT_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export PATH=${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$gpu" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=== [$window_name] [$policy] $exp_name | GPU: $gpu | Seed: $seed ==='" Enter

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
    IFS=':' read -r policy script exp_name subdir seed <<< "$task"
    gpu="${GPUS[$i]}"
    start_experiment "$policy" "$script" "$exp_name" "$subdir" "$gpu" "$seed"
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
    IFS=':' read -r policy script exp_name subdir seed <<< "$task"
    start_experiment "$policy" "$script" "$exp_name" "$subdir" "$FREE_GPU" "$seed"
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
echo "  完整政策消融实验完成!"
echo "=============================================="

SUCCESS=0
FAILED=0
FAIL_LIST=""

for task in "${TASK_LIST[@]}"; do
    IFS=':' read -r policy script exp_name subdir seed <<< "$task"
    task_id="${policy}_${exp_name}_seed${seed}"
    status_file="${STATUS_DIR}/exp_${task_id}.status"

    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "success" ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            FAIL_LIST="${FAIL_LIST}  [失败] [$policy] $exp_name (seed=$seed)\n"
        fi
    else
        FAIL_LIST="${FAIL_LIST}  [未知] [$policy] $exp_name (seed=$seed)\n"
    fi
done

if [ -n "$FAIL_LIST" ]; then
    echo -e "$FAIL_LIST"
fi

echo ""
echo "成功: $SUCCESS | 失败: $FAILED | 总计: $TOTAL"
echo ""
echo "结果目录:"
echo "  none:       ${RESULT_DIR}/Baseline/"
echo "  structured: ${RESULT_DIR}/Multimodal/"
echo "  bert:       ${RESULT_DIR}/MultimodalBert/"
echo "  hybrid:     ${RESULT_DIR}/MultimodalHybrid/"
echo "  (旧版路径: ${RESULT_DIR}/*.pkl — resume时也会检查)"
echo ""
echo "日志目录: $LOG_DIR"
echo ""

read -p "进入 tmux 查看? [Y/n] " -n 1 -r
echo
if [ ! "$REPLY" = "n" ] && [ ! "$REPLY" = "N" ]; then
    tmux attach -t "$SESSION_NAME"
fi
