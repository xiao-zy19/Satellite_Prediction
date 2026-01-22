# Baseline Pretrain Experiments

基于 Alpha Earth 卫星嵌入数据的人口自然增长率预测实验框架。

本项目比较了不同模型架构（MLP、LightCNN、ResNet、**Multimodal**）和不同预训练策略（无预训练、SimCLR、MAE、ImageNet）对人口增长率预测性能的影响。

**v3.0 新增**: 多模态（Multimodal）模型支持，融合卫星图像特征与结构化政策特征，实现更精准的人口增长率预测。

## 项目结构

```
Baseline_Pretrain/
├── config.py                      # 基础配置（路径、模型、训练参数、实验预设）
├── config_multimodal.py           # 多模态实验配置 [NEW]
├── dataset.py                     # 基础数据集类和数据加载器
├── dataset_policy.py              # 带政策特征的数据集 [NEW]
├── train.py                       # 基础模型训练脚本
├── train_multimodal.py            # 多模态模型训练脚本 [NEW]
├── evaluate.py                    # 模型评估脚本
├── compare_results.py             # 实验结果对比分析
├── utils.py                       # 工具函数（指标计算、日志、检查点等）
├── policy_features.py             # 政策特征提取模块 [NEW]
├── verify_policy_data.py          # 政策数据验证脚本 [NEW]
├── preprocess_patches.py          # 数据预处理：TIFF → 25个patch的npy文件
├── preprocess_individual_patches.py  # 数据预处理：TIFF → 25个独立patch文件
├── requirements.txt               # 项目依赖
├── models/
│   ├── __init__.py
│   ├── mlp_model.py               # MLP 模型
│   ├── light_cnn.py               # 轻量级 CNN 模型
│   ├── resnet_baseline.py         # ResNet 基线模型
│   ├── aggregators.py             # 位置感知聚合模块
│   └── multimodal.py              # 多模态融合模型 [NEW]
├── pretrain/
│   ├── __init__.py
│   ├── simclr.py                  # SimCLR 对比学习预训练
│   └── mae.py                     # MAE 掩码自编码器预训练
├── scripts/                       # 实验运行脚本
│   ├── baseline/                  # 基础模型实验脚本
│   │   ├── run_all_experiments.sh       # 串行运行所有实验
│   │   ├── run_all_experiments_parallel.sh  # 并行运行实验
│   │   ├── run_experiments.sh           # 灵活实验脚本
│   │   └── run_simple.sh                # 简易批量运行脚本
│   ├── patch_level/               # Patch-level 实验脚本
│   │   ├── run_patch_experiments.sh     # Patch-level 批量实验
│   │   └── run_patch_level.sh           # Patch-level 单实验
│   ├── multimodal/                # 多模态实验脚本 [NEW]
│   │   ├── run_mm_simple.sh             # 多模态简易运行脚本
│   │   ├── run_mm_tmux.sh               # 多模态 tmux 管理脚本
│   │   ├── run_multimodal_experiments.sh  # 多模态批量实验
│   │   └── run_multimodal_queue.sh      # 多模态队列运行
│   └── utils/                     # 工具脚本
│       └── start_experiments.sh         # tmux 启动实验
├── checkpoints/                   # 模型检查点 (gitignore)
├── logs/                          # 训练日志 (gitignore)
├── results/                       # 实验结果 (*.pkl gitignore)
└── wandb/                         # Wandb 日志 (gitignore)
```

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain
pip install -r requirements.txt
```

### 2. 数据预处理（可选，加速训练）

```bash
# 将TIFF文件预处理为npy格式（所有25个patch合并）
python preprocess_patches.py

# 或预处理为独立patch文件（最快的patch-level训练）
python preprocess_individual_patches.py
```

### 3. 运行单个实验

```bash
# 基线模型（无预训练，mean 聚合）
python train.py --exp mlp_baseline --gpu 3
python train.py --exp light_cnn_baseline --gpu 3

# 指定随机种子以复现实验（默认 seed=42）
python train.py --exp mlp_baseline --gpu 3 --seed 42

# 使用不同 seed（结果自动保存到不同文件，不会覆盖）
python train.py --exp mlp_baseline --gpu 3 --seed 123
# 输出: results/mlp_baseline_seed123_results.pkl
# 模型: checkpoints/mlp_baseline_seed123/best_model.pth

# 基础聚合方法对比（median, trimmed_mean）
python train.py --exp mlp_median --gpu 3
python train.py --exp mlp_trimmed_mean --gpu 3
python train.py --exp light_cnn_median --gpu 3
python train.py --exp light_cnn_trimmed_mean --gpu 3

# Patch-level 训练模式（自动测试3种聚合方式）
python train.py --exp mlp_patch_level --gpu 3

# 带 SimCLR 预训练（3种基础聚合）
python train.py --exp simclr_cnn --gpu 3           # mean
python train.py --exp simclr_cnn_median --gpu 3    # median
python train.py --exp simclr_cnn_trimmed_mean --gpu 3  # trimmed_mean

# 带 MAE 预训练
python train.py --exp mae_cnn --gpu 3

# ImageNet 预训练的 ResNet（3种基础聚合）
python train.py --exp resnet18_imagenet --gpu 3
python train.py --exp resnet18_imagenet_median --gpu 3
python train.py --exp resnet18_imagenet_trimmed_mean --gpu 3

# 位置感知聚合实验
python train.py --exp light_cnn_transformer_2d --gpu 3

# ============================================
# 多模态实验（卫星图像 + 政策特征）[NEW]
# ============================================

# 多模态基线（LightCNN + Concat 融合）
python train_multimodal.py --exp mm_cnn_concat --gpu 3

# 不同融合策略
python train_multimodal.py --exp mm_cnn_gated --gpu 3      # 门控融合
python train_multimodal.py --exp mm_cnn_attention --gpu 3  # 注意力融合
python train_multimodal.py --exp mm_cnn_film --gpu 3       # FiLM 融合

# 多模态 Patch-level 模式
python train_multimodal.py --exp mm_cnn_concat_patch --gpu 3

# ResNet + 多模态
python train_multimodal.py --exp mm_resnet18_concat --gpu 3
```

### 4. 验证政策数据（可选）

```bash
# 验证政策数据完整性和正确性
python verify_policy_data.py
```

### 6. 评估模型

```bash
python evaluate.py --exp mae_cnn --gpu 0 --split test
```

### 7. 对比所有实验结果

```bash
python compare_results.py
```

---

## 数据说明

### 输入数据

| 数据类型 | 路径 | 说明 |
|---------|------|------|
| 卫星嵌入 | `data_local/city_satellite_tiles/` | Alpha Earth 64维嵌入，1000×1000像素 TIFF |
| 人口数据 | `人口数据/人口自然增长率_2018-2024_filtered-empty.xlsx` | 2018-2024年各城市人口自然增长率 |
| 政策数据 | `data_local/Fertility_Policy/fertility_policies_by_province_year.jsonl` | 2013-2024年各省份生育政策特征 [NEW] |

### 数据规格

- **空间分辨率**: 10米/像素
- **覆盖范围**: 10km × 10km 每城市
- **嵌入维度**: 64 通道
- **Patch 划分**: 5×5 = 25 个 patch，每个 2km × 2km (200×200 像素)
- **时间范围**: 2018-2024 年
- **数据划分**: 65% 训练 / 15% 验证 / 20% 测试（按城市分层）

### 预处理格式

| 格式 | 路径 | 形状 | 存储格式 | 用途 |
|-----|------|------|---------|------|
| 原始 TIFF | `city_satellite_tiles/` | (64, 1000, 1000) | int8 | 原始数据 |
| 合并 NPY | `city_patches/` | (25, 64, 200, 200) | int8 | City-level 训练 |
| 独立 NPY | `city_individual_patches/` | (64, 200, 200) × 25 | int8 | Patch-level 训练 |

> **注意**: 预处理脚本保持 int8 格式存储，相比 float32 节省 4 倍存储空间。加载时自动转换为 float32 并归一化。

### 政策特征 [NEW]

多模态模型使用 12 维结构化政策特征：

| 特征名称 | 类型 | 范围 | 说明 |
|---------|------|------|------|
| `maternity_leave_days` | 连续 | [98, 365] | 产假天数 |
| `paternity_leave_days` | 连续 | [0, 30] | 陪产假天数 |
| `parental_leave_days` | 连续 | [0, 20] | 育儿假天数/年 |
| `marriage_leave_days` | 连续 | [3, 30] | 婚假天数 |
| `birth_subsidy_amount` | 连续 | [0, 10] | 生育补贴（万元/年） |
| `housing_subsidy_amount` | 连续 | [0, 100] | 住房补贴（万元） |
| `childcare_subsidy_monthly` | 连续 | [0, 2000] | 托育补贴（元/月） |
| `tax_deduction_monthly` | 连续 | [0, 2000] | 个税抵扣（元/月） |
| `has_childcare_policy` | 二值 | {0, 1} | 是否有托育政策 |
| `has_ivf_insurance` | 二值 | {0, 1} | 辅助生殖是否纳入医保 |
| `is_minority_favorable` | 二值 | {0, 1} | 是否为少数民族优惠地区 |
| `policy_phase` | 离散 | [0, 3] | 政策阶段（0=单独二孩, 1=全面二孩, 2=三孩, 3=生育友好） |

**时间滞后机制**: 为防止数据泄露，预测第 Y 年的人口增长率时，使用第 Y-1 年的政策特征（默认 lag=1）。

---

## 训练模式

### City-level 训练（默认）

每个样本包含城市的 25 个 patch，模型内部聚合后输出单个预测值。**训练和评估使用相同的聚合方式**。

```
训练: 输入 (batch, 25, 64, 200, 200) → 模型聚合 → 输出 (batch, 1)
评估: 使用相同的模型聚合方式 → 输出指标
```

**模型聚合方式可选**:
- **基础聚合**: mean, median, trimmed_mean（无额外参数）
- **高级聚合**: attention, pos_attention, spatial_attention, transformer, transformer_2d（有可学习参数）

**特点**:
- **训练和评估聚合一致**：模型使用什么聚合训练，就用什么聚合评估
- 使用 val R² 作为 early stopping 依据
- 不同聚合方式需要分别训练和比较

**适用实验**: `mlp_baseline`, `mlp_median`, `mlp_trimmed_mean`, `light_cnn_*`, `simclr_cnn_*`, `mae_cnn_*`, `resnet*_*` 等

### Patch-level 训练

每个 patch 作为独立样本训练，**推理时自动测试 3 种聚合方式**（mean, median, trimmed_mean）。

```
训练: 每个patch独立预测 → (batch, 64, 200, 200) → (batch, 1)
推理: 25个patch预测 → 同时用3种方式聚合 → 返回3组指标 × 2数据集 = 6组结果
```

**特点**:
- 数据量扩大 25 倍
- **一次训练，6 组评估结果**（Val × 3 + Test × 3）
- 使用 trimmed_mean R² 作为 early stopping 依据

**适用实验**: `mlp_patch_level`, `light_cnn_patch_level`, `resnet*_patch_level`, `simclr_cnn_patch_level`, `mae_cnn_patch_level`

### 多模态训练 [NEW]

多模态模型融合卫星图像特征和结构化政策特征，支持 4 种融合策略：

```
训练: 输入 image(batch, 25, 64, 200, 200) + policy(batch, 12)
      → 图像编码器 → 聚合 → 融合层 → 回归头 → 输出 (batch, 1)
```

**融合策略**:
| 策略 | 说明 | 参数量增加 |
|------|------|-----------|
| `concat` | 简单拼接后映射 | ~5K |
| `gated` | 门控融合，学习混合权重 | ~10K |
| `attention` | 跨模态注意力融合 | ~15K |
| `film` | Feature-wise Linear Modulation，政策调制图像特征 | ~8K |

**训练脚本**: `train_multimodal.py`

**适用实验**: `mm_cnn_*`, `mm_mlp_*`, `mm_resnet*_*`

---

## 实验总览

当前共有 **111 个实验配置**（基础模型 80 + 多模态 31），按模型类型组织如下：

### 基础模型实验（80 个）

| 模型 | 无预训练 | SimCLR | MAE | ImageNet | 总计 |
|------|---------|--------|-----|----------|------|
| MLP | 9 | 3 | - | - | 12 |
| LightCNN | 9 | 9 | 9 | - | 27 |
| ResNet10 | 4 | - | - | - | 4 |
| ResNet18 | 9 | - | - | 4 | 13 |
| ResNet34 | 4 | - | - | 4 | 8 |
| ResNet50 | 4 | - | - | 4 | 8 |
| ResNet101 | 4 | - | - | 4 | 8 |
| **总计** | **43** | **12** | **9** | **16** | **80** |

### 多模态实验（31 个）[NEW]

| 图像编码器 | Concat | Gated | Attention | FiLM | Patch-level | 总计 |
|-----------|--------|-------|-----------|------|-------------|------|
| LightCNN | 3 | 2 | 2 | 2 | 3 | 12 |
| MLP | 2 | 1 | - | - | - | 3 |
| ResNet18 | 1 | 1 | - | 1 | 1 | 4 |
| ResNet34 (pretrained) | 1 | - | - | - | - | 1 |
| 自定义 LightCNN | 1 | - | - | - | - | 1 |
| **总计** | **8** | **4** | **2** | **3** | **4** | **31** |

> **评估说明**：
> - **City-level**：训练和评估使用相同的模型内置聚合方式（mean/median/trimmed_mean 或高级聚合）
> - **Patch-level**：评估时自动测试 3 种聚合方式（mean/median/trimmed_mean）
> - **多模态**：融合图像特征和政策特征，支持 4 种融合策略

---

## 完整实验配置表

> **聚合方式说明**：
> - **基础聚合**：mean、median、trimmed_mean（无额外参数）
> - **高级聚合**：attention、pos_attention、spatial_attention、transformer、transformer_2d（有可学习参数）
> - **City-level**：训练和评估使用相同聚合方式
> - **Patch-level**：评估时自动测试 3 种基础聚合方式

### 1. MLP 系列（12个）

#### 无预训练（9个）
| # | 实验名称 | 模型聚合 | 模式 | 参数量 |
|---|----------|----------|------|--------|
| 1 | mlp_baseline | mean | city_level | 60.8K |
| 2 | mlp_median | median | city_level | 60.8K |
| 3 | mlp_trimmed_mean | trimmed_mean | city_level | 60.8K |
| 4 | mlp_patch_level | - | patch_level | 60.8K |
| 5 | mlp_attention | attention | city_level | 69K |
| 6 | mlp_pos_attention | pos_attention | city_level | 147K |
| 7 | mlp_spatial_attention | spatial_attention | city_level | 280K |
| 8 | mlp_transformer | transformer | city_level | 461K |
| 9 | mlp_transformer_2d | transformer_2d | city_level | 475K |

#### SimCLR 预训练（3个）
| # | 实验名称 | 模型聚合 | 模式 | 参数量 |
|---|----------|----------|------|--------|
| 10 | simclr_mlp | mean | city_level | 60.8K |
| 11 | simclr_mlp_median | median | city_level | 60.8K |
| 12 | simclr_mlp_trimmed_mean | trimmed_mean | city_level | 60.8K |

### 2. LightCNN 系列（27个）

#### 无预训练（9个）
| # | 实验名称 | 模型聚合 | 模式 | 参数量 |
|---|----------|----------|------|--------|
| 13 | light_cnn_baseline | mean | city_level | 160.6K |
| 14 | light_cnn_median | median | city_level | 160.6K |
| 15 | light_cnn_trimmed_mean | trimmed_mean | city_level | 160.6K |
| 16 | light_cnn_patch_level | - | patch_level | 160.6K |
| 17 | light_cnn_attention | attention | city_level | 168.9K |
| 18 | light_cnn_pos_attention | pos_attention | city_level | 246.6K |
| 19 | light_cnn_spatial_attention | spatial_attention | city_level | 380.2K |
| 20 | light_cnn_transformer | transformer | city_level | 560.8K |
| 21 | light_cnn_transformer_2d | transformer_2d | city_level | 574.8K |

#### SimCLR 预训练（9个）
| # | 实验名称 | 模型聚合 | 模式 | 参数量 |
|---|----------|----------|------|--------|
| 22 | simclr_cnn | mean | city_level | 160.6K |
| 23 | simclr_cnn_median | median | city_level | 160.6K |
| 24 | simclr_cnn_trimmed_mean | trimmed_mean | city_level | 160.6K |
| 25 | simclr_cnn_patch_level | - | patch_level | 160.6K |
| 26 | simclr_cnn_attention | attention | city_level | 168.9K |
| 27 | simclr_cnn_pos_attention | pos_attention | city_level | 246.6K |
| 28 | simclr_cnn_spatial_attention | spatial_attention | city_level | 380.2K |
| 29 | simclr_cnn_transformer | transformer | city_level | 560.8K |
| 30 | simclr_cnn_transformer_2d | transformer_2d | city_level | 574.8K |

#### MAE 预训练（9个）
| # | 实验名称 | 模型聚合 | 模式 | 参数量 |
|---|----------|----------|------|--------|
| 31 | mae_cnn | mean | city_level | 160.6K |
| 32 | mae_cnn_median | median | city_level | 160.6K |
| 33 | mae_cnn_trimmed_mean | trimmed_mean | city_level | 160.6K |
| 34 | mae_cnn_patch_level | - | patch_level | 160.6K |
| 35 | mae_cnn_attention | attention | city_level | 168.9K |
| 36 | mae_cnn_pos_attention | pos_attention | city_level | 246.6K |
| 37 | mae_cnn_spatial_attention | spatial_attention | city_level | 380.2K |
| 38 | mae_cnn_transformer | transformer | city_level | 560.8K |
| 39 | mae_cnn_transformer_2d | transformer_2d | city_level | 574.8K |

### 3. ResNet 系列（41个）

#### ResNet10（~5.5M 参数）
| # | 实验名称 | 模型聚合 | 模式 | 预训练 |
|---|----------|----------|------|--------|
| 40 | resnet10_baseline | mean | city_level | None |
| 41 | resnet10_median | median | city_level | None |
| 42 | resnet10_trimmed_mean | trimmed_mean | city_level | None |
| 43 | resnet10_patch_level | - | patch_level | None |

#### ResNet18（~11.8M 参数）
| # | 实验名称 | 模型聚合 | 模式 | 预训练 |
|---|----------|----------|------|--------|
| 44 | resnet18_baseline | mean | city_level | None |
| 45 | resnet18_median | median | city_level | None |
| 46 | resnet18_trimmed_mean | trimmed_mean | city_level | None |
| 47 | resnet18_imagenet | mean | city_level | ImageNet |
| 48 | resnet18_imagenet_median | median | city_level | ImageNet |
| 49 | resnet18_imagenet_trimmed_mean | trimmed_mean | city_level | ImageNet |
| 50 | resnet18_patch_level | - | patch_level | None |
| 51 | resnet18_imagenet_patch_level | - | patch_level | ImageNet |
| 52 | resnet18_attention | attention | city_level | None |
| 53 | resnet18_pos_attention | pos_attention | city_level | None |
| 54 | resnet18_spatial_attention | spatial_attention | city_level | None |
| 55 | resnet18_transformer | transformer | city_level | None |
| 56 | resnet18_transformer_2d | transformer_2d | city_level | None |

#### ResNet34（~21.9M 参数）
| # | 实验名称 | 模型聚合 | 模式 | 预训练 |
|---|----------|----------|------|--------|
| 57 | resnet34_baseline | mean | city_level | None |
| 58 | resnet34_median | median | city_level | None |
| 59 | resnet34_trimmed_mean | trimmed_mean | city_level | None |
| 60 | resnet34_imagenet | mean | city_level | ImageNet |
| 61 | resnet34_imagenet_median | median | city_level | ImageNet |
| 62 | resnet34_imagenet_trimmed_mean | trimmed_mean | city_level | ImageNet |
| 63 | resnet34_patch_level | - | patch_level | None |
| 64 | resnet34_imagenet_patch_level | - | patch_level | ImageNet |

#### ResNet50（~24.9M 参数，Bottleneck）
| # | 实验名称 | 模型聚合 | 模式 | 预训练 |
|---|----------|----------|------|--------|
| 65 | resnet50_baseline | mean | city_level | None |
| 66 | resnet50_median | median | city_level | None |
| 67 | resnet50_trimmed_mean | trimmed_mean | city_level | None |
| 68 | resnet50_imagenet | mean | city_level | ImageNet |
| 69 | resnet50_imagenet_median | median | city_level | ImageNet |
| 70 | resnet50_imagenet_trimmed_mean | trimmed_mean | city_level | ImageNet |
| 71 | resnet50_patch_level | - | patch_level | None |
| 72 | resnet50_imagenet_patch_level | - | patch_level | ImageNet |

#### ResNet101（~43.9M 参数，Bottleneck）
| # | 实验名称 | 模型聚合 | 模式 | 预训练 |
|---|----------|----------|------|--------|
| 73 | resnet101_baseline | mean | city_level | None |
| 74 | resnet101_median | median | city_level | None |
| 75 | resnet101_trimmed_mean | trimmed_mean | city_level | None |
| 76 | resnet101_imagenet | mean | city_level | ImageNet |
| 77 | resnet101_imagenet_median | median | city_level | ImageNet |
| 78 | resnet101_imagenet_trimmed_mean | trimmed_mean | city_level | ImageNet |
| 79 | resnet101_patch_level | - | patch_level | None |
| 80 | resnet101_imagenet_patch_level | - | patch_level | ImageNet |

> **注意**：ResNet50/101 使用 Bottleneck blocks，特征维度为 2048（其他为 512）

### 4. 多模态系列（31个）[NEW]

#### City-level 多模态实验（27个）

**LightCNN + 不同融合策略**
| # | 实验名称 | 融合策略 | 聚合方式 | 说明 |
|---|----------|----------|----------|------|
| 81 | mm_cnn_concat | concat | mean | 基线融合 |
| 82 | mm_cnn_concat_median | concat | median | |
| 83 | mm_cnn_concat_trimmed | concat | trimmed_mean | |
| 84 | mm_cnn_gated | gated | mean | 门控融合 |
| 85 | mm_cnn_gated_trimmed | gated | trimmed_mean | |
| 86 | mm_cnn_attention | attention | mean | 注意力融合 |
| 87 | mm_cnn_attention_trimmed | attention | trimmed_mean | |
| 88 | mm_cnn_film | film | mean | FiLM 融合 |
| 89 | mm_cnn_film_trimmed | film | trimmed_mean | |

**MLP + 融合**
| # | 实验名称 | 融合策略 | 聚合方式 |
|---|----------|----------|----------|
| 90 | mm_mlp_concat | concat | mean |
| 91 | mm_mlp_gated | gated | mean |

**ResNet + 融合**
| # | 实验名称 | 融合策略 | 聚合方式 |
|---|----------|----------|----------|
| 92 | mm_resnet18_concat | concat | mean |
| 93 | mm_resnet18_gated | gated | mean |
| 94 | mm_resnet18_film | film | mean |
| 95 | mm_resnet34_pretrained | concat | mean |

**自定义 LightCNN**
| # | 实验名称 | 融合策略 | 说明 |
|---|----------|----------|------|
| 96 | mm_cnn_small_concat | concat | channels=[16,32,64] |

#### Patch-level 多模态实验（4个）
| # | 实验名称 | 图像编码器 | 融合策略 |
|---|----------|-----------|----------|
| 97 | mm_cnn_concat_patch | LightCNN | concat |
| 98 | mm_cnn_gated_patch | LightCNN | gated |
| 99 | mm_cnn_film_patch | LightCNN | film |
| 100 | mm_resnet18_concat_patch | ResNet18 | concat |

> **多模态特点**：
> - 政策特征维度：12 维（默认不编码，直接使用原始特征）
> - 图像特征维度：64 维（投影后）
> - 使用时间滞后（lag=1）防止数据泄露

---

## 模型架构

### MLP Model (`models/mlp_model.py`)

```
输入 patch (64, 200, 200)
    ↓ Global Average Pooling
特征向量 (64,)
    ↓ MLP: 64 → 256 → 128 → 64
    ↓ Aggregation [city_level only]
    ↓ Regression Head: 64 → 32 → 1
输出 (1,)
```

**参数量**: ~60K

### LightCNN Model (`models/light_cnn.py`)

```
输入 patch (64, 200, 200)
    ↓ Conv Block 1: 64 → 32, MaxPool
    ↓ Conv Block 2: 32 → 64, MaxPool
    ↓ Conv Block 3: 64 → 128, MaxPool
    ↓ Global Average Pooling
特征向量 (128,)
    ↓ FC: 128 → 256 → 64
    ↓ Aggregation [city_level only]
    ↓ Regression Head
输出 (1,)
```

**参数量**: ~160K

### ResNet Baseline (`models/resnet_baseline.py`)

支持多种 ResNet 变体：

| 模型 | 参数量 | 特征维度 | 相对大小 | 说明 |
|------|--------|---------|---------|------|
| ResNet10 | ~5.5M | 512 | 0.47x | 自定义轻量版 |
| ResNet18 | ~11.8M | 512 | 1.00x | 标准基线 |
| ResNet34 | ~21.9M | 512 | 1.86x | 更深的BasicBlock |
| ResNet50 | ~26.3M | 2048 | 2.23x | Bottleneck blocks |
| ResNet101 | ~45.3M | 2048 | 3.85x | 最深，Bottleneck |

### 聚合方式 (`models/aggregators.py`)

| 方式 | 类型 | 位置感知 | 参数量 (128d) | 说明 |
|-----|------|---------|--------------|------|
| `mean` | 基础 | - | 0 | 简单平均（默认） |
| `median` | 基础 | - | 0 | 中位数聚合 |
| `trimmed_mean` | 基础 | - | 0 | 去除极端值后平均（10%） |
| `attention` | 高级 | - | ~8K | MLP 注意力加权平均 |
| `pos_attention` | 高级 | 1D | ~86K | 可学习位置编码 + 多头注意力 |
| `spatial_attention` | 高级 | 2D | ~220K | 2D 行列位置编码 + 注意力 |
| `transformer` | 高级 | 1D | ~400K | [CLS] Token + Transformer |
| `transformer_2d` | 高级 | 2D | ~414K | 2D 位置 + [CLS] + Transformer |

### Multimodal Model (`models/multimodal.py`) [NEW]

```
输入 image patches (batch, 25, 64, 200, 200) + policy features (batch, 12)
    ↓
图像编码器 (LightCNN/MLP/ResNet)
    ↓ encoder_dim (128)
聚合层 (mean/median/trimmed_mean/attention/...)
    ↓ 图像特征 (batch, 128)
    ↓ 投影 → (batch, 64)
    ↓
融合层 (concat/gated/attention/film)
    ├── image_feat (batch, 64)
    └── policy_feat (batch, 12)
    ↓ 融合特征 (batch, 64)
    ↓
回归头: 64 → 32 → 1
    ↓
输出 (batch, 1)
```

**融合策略详解**:

| 融合类型 | 实现方式 | 数学表达 |
|---------|---------|---------|
| `concat` | 拼接后 MLP | `f = MLP([img; policy])` |
| `gated` | 门控混合 | `f = σ(gate) * img + (1-σ(gate)) * policy` |
| `attention` | 跨模态自注意力 | `f = mean(MultiHeadAttn([img, policy]))` |
| `film` | 特征调制 | `f = γ(policy) * img + β(policy)` |

**参数量**: ~180K（LightCNN + concat 融合）

---

## 自监督预训练

### SimCLR (`pretrain/simclr.py`)

**原理**: 对比学习，最大化同一样本不同增强视图的相似性。

```
样本 → 增强1, 增强2 → 编码器 → 投影头 → NT-Xent Loss
```

**配置**: Temperature=0.5, Projection dim=128, 预训练轮数=50

### MAE (`pretrain/mae.py`)

**原理**: 掩码自编码器，通过重建被遮蔽的 patch 学习表示。

```
25个patch → 随机遮蔽75% → 编码可见patch → 解码器重建 → MSE Loss
```

**配置**: Mask ratio=0.75, Decoder dim=256, Decoder depth=2

---

## 训练配置

主要参数（`config.py`）:

```python
# 训练
batch_size = 16          # city_level
batch_size = 64          # patch_level
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-4
patience = 60            # Early stopping

# 学习率调度
scheduler = "cosine_warm_restarts"
t_max = 10               # 重启周期
t_mult = 2               # 周期倍增
eta_min = 1e-7

# 预训练
pretrain_epochs = 50
pretrain_lr = 1e-3
freeze_encoder_epochs = 5

# 数据划分
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20
RANDOM_SEED = 42         # 随机种子（可通过命令行 --seed 覆盖）
```

### 实验复现

使用相同的 `--seed` 参数可以完全复现实验结果：

```bash
# 第一次运行
python train.py --exp mlp_baseline --gpu 3 --seed 42

# 复现运行（结果完全一致）
python train.py --exp mlp_baseline --gpu 3 --seed 42
```

**复现机制**：
- 数据划分：使用 seed 控制城市的 train/val/test 划分
- DataLoader shuffle：使用 seed 控制每个 epoch 的数据顺序
- 模型初始化：使用 seed 控制权重初始化
- 训练过程：固定 `torch.backends.cudnn.deterministic = True`

**多种子运行**（用于统计显著性测试）：

```bash
# 单独运行不同种子（结果文件不会覆盖）
python train.py --exp mlp_baseline --gpu 3 --seed 42   # -> results/mlp_baseline_results.pkl
python train.py --exp mlp_baseline --gpu 3 --seed 123  # -> results/mlp_baseline_seed123_results.pkl
python train.py --exp mlp_baseline --gpu 3 --seed 456  # -> results/mlp_baseline_seed456_results.pkl

# 或使用批量脚本
bash scripts/baseline/run_simple.sh --exp mlp_baseline --seeds 42,123,456
```

**输出文件命名规则**：

| Seed | run_id | 结果文件 | Checkpoint 目录 |
|------|--------|----------|-----------------|
| 42 (默认) | `mlp_baseline` | `mlp_baseline_results.pkl` | `checkpoints/mlp_baseline/` |
| 123 | `mlp_baseline_seed123` | `mlp_baseline_seed123_results.pkl` | `checkpoints/mlp_baseline_seed123/` |
| 456 | `mlp_baseline_seed456` | `mlp_baseline_seed456_results.pkl` | `checkpoints/mlp_baseline_seed456/` |

---

## 评估指标

| 指标 | 说明 |
|-----|------|
| **Pearson r** | 皮尔逊相关系数，衡量预测与真值的线性相关性 |
| **R²** | 决定系数，模型解释方差的比例 |
| **MAE** | 平均绝对误差 |
| **RMSE** | 均方根误差 |

### City-level 输出格式

City-level 实验输出单组指标（使用模型内置聚合方式）：

```
Final Results (City-Level, aggregation: attention)
============================================================

Validation Set:
  R²=0.1290 | r=0.3590 | MAE=0.5610 | RMSE=0.7800

Test Set:
  R²=0.1230 | r=0.3510 | MAE=0.5650 | RMSE=0.7850
```

> 训练和评估使用相同的聚合方式，确保一致性。

### Patch-level 输出格式

Patch-level 实验输出 **6 组指标**（3 种聚合 × 2 个数据集）：

```
Final Results (Patch-Level with 3 Aggregation Methods)
============================================================

Validation Set:
  mean        : R²=0.1234 | r=0.3512 | MAE=0.5678 | RMSE=0.7890
  median      : R²=0.1256 | r=0.3545 | MAE=0.5654 | RMSE=0.7856
  trimmed_mean: R²=0.1278 | r=0.3573 | MAE=0.5632 | RMSE=0.7823

Test Set:
  mean        : R²=0.1189 | r=0.3448 | MAE=0.5712 | RMSE=0.7934
  median      : R²=0.1201 | r=0.3465 | MAE=0.5698 | RMSE=0.7912
  trimmed_mean: R²=0.1215 | r=0.3487 | MAE=0.5678 | RMSE=0.7889
```

---

## 脚本使用

### 训练 (`train.py`)

```bash
python train.py --exp <experiment_name> --gpu <gpu_id> [--seed <random_seed>]
```

**参数**:
- `--exp`: 实验名称（必需）
- `--gpu`: GPU ID（默认: 3）
- `--seed`: 随机种子，用于复现实验（默认: 42）

**输出路径**（根据 seed 自动命名）:

| Seed | Checkpoint | Results | Logs |
|------|------------|---------|------|
| 42 (默认) | `checkpoints/{exp}/` | `results/{exp}_results.pkl` | `logs/{exp}_{timestamp}.log` |
| 其他 | `checkpoints/{exp}_seed{seed}/` | `results/{exp}_seed{seed}_results.pkl` | `logs/{exp}_seed{seed}_{timestamp}.log` |

### 批量运行脚本

脚本按类别组织在 `scripts/` 子目录中：

| 目录 | 说明 |
|------|------|
| `scripts/baseline/` | 基础模型实验脚本 |
| `scripts/patch_level/` | Patch-level 实验脚本 |
| `scripts/multimodal/` | 多模态实验脚本 |
| `scripts/utils/` | 工具脚本 |

#### 基础模型批量运行 (`scripts/baseline/run_simple.sh`)

使用 tmux 管理的并行实验运行脚本，支持多 GPU 并行和多种子运行。

```bash
cd scripts/baseline

# 查看帮助
bash run_simple.sh --help

# 列出所有实验
bash run_simple.sh --list

# 预览要运行的实验
bash run_simple.sh --category baseline --dry-run
```

**单种子运行**:
```bash
# 使用默认种子 (42) 运行所有实验
bash run_simple.sh --category all

# 使用指定种子运行
bash run_simple.sh --category baseline --seed 123

# 指定 GPU
bash run_simple.sh --category resnet --gpus 0,1,2,3

# 跳过已完成的实验
bash run_simple.sh --category all --resume
```

**多种子运行**（每个实验运行多次）:
```bash
# 使用 3 个种子运行 baseline 实验
# 6 个实验 × 3 个种子 = 18 个任务
bash run_simple.sh --category baseline --seeds 42,123,456

# 预览多种子任务
bash run_simple.sh --category baseline --seeds 42,123,456 --dry-run
# 输出:
#   1. mlp_baseline (seed=42)
#   2. mlp_baseline (seed=123)
#   3. mlp_baseline (seed=456)
#   4. mlp_median (seed=42)
#   ...
```

#### 多模态批量运行 (`scripts/multimodal/run_mm_simple.sh`)

```bash
cd scripts/multimodal

# 查看帮助
bash run_mm_simple.sh --help

# 运行所有多模态实验
bash run_mm_simple.sh --category all --gpus 0,1,2,3

# 运行特定融合策略
bash run_mm_simple.sh --category concat
bash run_mm_simple.sh --category gated
bash run_mm_simple.sh --category attention
```

**可用实验类别**:

| 类别 | 说明 | 实验数量 |
|------|------|---------|
| `all` | 所有实验 | 80 |
| `baseline` | MLP/CNN 无预训练 | 6 |
| `ssl` | SimCLR/MAE 预训练 | 9 |
| `patch` | Patch-level 训练 | 4 |
| `resnet` | 所有 ResNet 实验 | 40 |
| `resnet10/18/34/50/101` | 特定 ResNet 版本 | 4-13 |
| `agg` | Position-Aware 聚合 | 25 |

**参数说明**:

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gpus` | 指定 GPU 列表 | `--gpus 0,1,2,3` |
| `--parallel` | 并行数量 | `--parallel 4` |
| `--category` | 实验类别 | `--category baseline` |
| `--exp` | 指定实验名称 | `--exp mlp_baseline,light_cnn_baseline` |
| `--seed` | 单个随机种子 | `--seed 123` |
| `--seeds` | 多个随机种子 | `--seeds 42,123,456` |
| `--resume` | 跳过已完成实验 | `--resume` |
| `--dry-run` | 预览模式 | `--dry-run` |

### 评估 (`evaluate.py`)

```bash
python evaluate.py --exp <experiment_name> --split test
```

### 结果对比 (`compare_results.py`)

```bash
python compare_results.py
```

---

## 实验监控

### Wandb 集成

实验自动记录到 Wandb:
- 项目: `population-pretrain-comparison`
- 命名: `{exp_name}_{MMDD_HHMM}`

**City-level 模式记录**:
- `val/r2`, `val/pearson_r`, `val/mae`
- `test/r2`, `test/pearson_r`, `test/mae`

**Patch-level 模式记录**:
- `val_mean/r2`, `val_median/r2`, `val_trimmed_mean/r2`
- `test_mean/r2`, `test_median/r2`, `test_trimmed_mean/r2`

查看: https://wandb.ai/xiao-zy19/population-pretrain-comparison

---

## 依赖

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
rasterio>=1.3.0
openpyxl>=3.1.0
wandb>=0.15.0
```

---

## 更新日志

### v3.0 (2026-01-23) [NEW]

**重大更新：多模态（Multimodal）模型支持**

#### 新增模块
- **`models/multimodal.py`**: 多模态融合模型
  - 支持 4 种融合策略：concat, gated, attention, film
  - 支持 3 种图像编码器：LightCNN, MLP, ResNet
  - 支持 City-level 和 Patch-level 两种训练模式
- **`policy_features.py`**: 政策特征提取模块
  - 12 维结构化政策特征
  - 城市-省份映射（覆盖 31 省份）
  - 时间滞后机制（lag=1）防止数据泄露
  - 特征归一化支持
- **`dataset_policy.py`**: 带政策特征的数据集类
  - `CityPolicyDataset`: City-level 数据集
  - `PatchPolicyDataset`: Patch-level 数据集
  - 数据增强支持（翻转、旋转）
- **`config_multimodal.py`**: 多模态实验配置
  - 31 个预设实验配置
  - 灵活的模型配置选项
- **`train_multimodal.py`**: 多模态模型训练脚本
  - Wandb 集成
  - 早停机制
  - 多聚合策略评估
- **`verify_policy_data.py`**: 政策数据验证脚本
  - 6 步验证流程
  - 数据完整性检查

#### 政策特征说明
| 特征 | 类型 | 说明 |
|------|------|------|
| 假期类 | 4维 | 产假/陪产假/育儿假/婚假天数 |
| 补贴类 | 4维 | 生育/住房/托育/个税补贴 |
| 政策类 | 3维 | 托育政策/辅助生殖医保/少数民族优惠 |
| 阶段类 | 1维 | 政策阶段（0-3） |

#### 实验配置
- **新增实验**: 31 个多模态实验配置
- **实验总数**: 80 → 111

#### 运行脚本（重新组织目录结构）
- `scripts/multimodal/`: 多模态实验脚本
  - `run_mm_simple.sh`: 多模态实验简易运行脚本
  - `run_mm_tmux.sh`: 多模态实验 tmux 管理脚本
  - `run_multimodal_experiments.sh`: 多模态批量实验脚本
  - `run_multimodal_queue.sh`: 多模态队列运行脚本
- `scripts/baseline/`: 基础模型实验脚本
- `scripts/patch_level/`: Patch-level 实验脚本
- `scripts/utils/`: 工具脚本

---

### v2.5 (2026-01-02)

**重大更新：多种子运行支持与批量实验脚本**

#### 多种子运行支持
- **独立存储**: 不同 seed 的结果自动保存到不同文件，不会互相覆盖
  - 默认 seed (42): `{exp}_results.pkl`, `checkpoints/{exp}/`
  - 其他 seed: `{exp}_seed{seed}_results.pkl`, `checkpoints/{exp}_seed{seed}/`
- **run_id 机制**: 引入 `run_id` 标识符，统一管理 checkpoint、results、logs、wandb
- **统计显著性**: 支持多种子运行，便于计算均值和标准差

#### 批量运行脚本完善 (`scripts/baseline/run_simple.sh`)
- **实验配置更新**: 与 `config.py` 完全同步，共 80 个实验
- **多种子支持**: 新增 `--seed` 和 `--seeds` 参数
  - `--seed 123`: 使用指定种子运行
  - `--seeds 42,123,456`: 使用多个种子运行（每个实验运行多次）
- **实验类别**: 支持按类别选择实验 (`baseline`, `ssl`, `patch`, `resnet`, `agg` 等)
- **Resume 功能**: 根据 seed 正确判断是否已完成

#### 代码修改
- `train.py`:
  - 新增 `run_id` 变量，根据 seed 自动生成唯一标识
  - `Trainer` 类新增 `run_id` 参数
  - `init_wandb()` 支持 seed 和 run_id 参数
  - 结果/checkpoint 保存路径使用 run_id
- `scripts/baseline/run_simple.sh`:
  - 实验列表与 config.py 同步
  - 支持 `--seed` 和 `--seeds` 参数
  - Resume 逻辑适配新的文件命名

### v2.4 (2026-01-01)

**重大更新：实验框架完善与复现支持**

#### 新增聚合机制
- **位置感知聚合**: 新增 5 种高级聚合方式
  - `attention`: MLP 注意力加权平均
  - `pos_attention`: 可学习 1D 位置编码 + 多头自注意力
  - `spatial_attention`: 2D 行列位置编码 + 注意力聚合
  - `transformer`: ViT 风格 [CLS] Token + Transformer
  - `transformer_2d`: 2D 位置编码 + [CLS] Token + Transformer
- **基础聚合扩展**: City-level 模式支持 `median` 和 `trimmed_mean` 聚合
- **新增模块**: `models/aggregators.py` 聚合器工厂模块

#### 实验配置扩展
- **实验总数**: 37 → 80（新增 43 个实验配置）
- **新增实验类别**:
  - MLP/LightCNN/ResNet 的 median 和 trimmed_mean 变体
  - SimCLR/MAE 预训练 + 各种聚合方式组合
  - ResNet18/34/50/101 的 ImageNet 预训练 + Patch-level 实验
  - 位置感知聚合实验（19 个）

#### 训练与评估改进
- **Patch-level 多聚合评估**: 一次训练，自动测试 3 种聚合方式（mean, median, trimmed_mean）
  - 输出 6 组结果（Val × 3 + Test × 3）
  - 使用 trimmed_mean R² 作为 early stopping 依据
- **City-level 评估修正**: 训练和评估使用相同的模型内置聚合方式
- **新增函数**: `evaluate_patch_level()` 用于多聚合策略评估

#### 实验复现支持
- **随机种子控制**: `train.py` 新增 `--seed` 参数（默认: 42）
- **复现机制**:
  - 数据划分使用 seed 控制城市的 train/val/test 划分
  - DataLoader 使用 `torch.Generator` 确保 shuffle 可复现
  - 固定 `torch.backends.cudnn.deterministic = True`
- **结果保存**: `results/*.pkl` 包含使用的 seed 值

#### 代码重构
- `Trainer` 类支持 `is_patch_level` 参数
- `get_dataloaders()` 和 `get_patch_level_dataloaders()` 支持 seed 参数
- 删除冗余实验别名（resnet_baseline 等）

### v2.3 (2025-12-31)
- **ResNet Scaling 实验**: 新增 14 个 ResNet 变体实验
  - 支持 ResNet10/18/34/50/101 五种模型规模
  - ResNet10 为自定义轻量实现（~5.5M 参数）
- **Batch Size 自适应**: 大模型自动使用更小的 batch size

### v2.2.1 (2025-12-30)
- **配置优化**: `simclr_cnn_patch_level` 减小 batch_size 避免 OOM

### v2.2 (2025-12-29)
- **数据格式优化**: 预处理保持 int8 格式，存储减少 4 倍
- **数据加载优化**: 启用 `persistent_workers` 和 `prefetch_factor`
- **Wandb 改进**: 实验名称自动添加时间戳

### v2.1 (2025-12-28)
- **启用 Wandb**: 实验自动记录
- **新增 Patch-level 训练模式**
- **新增预处理脚本**: `preprocess_individual_patches.py`

### v2.0 (2025-12-26)
- **修复 MAE 权重迁移**
- **新增 evaluate.py** 和 **compare_results.py**

### v1.0 (2025-12-25)
- 初始版本
