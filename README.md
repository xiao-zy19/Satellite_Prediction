# Baseline Pretrain Experiments

基于 Alpha Earth 卫星嵌入数据的人口自然增长率预测实验框架。

本项目比较了不同模型架构（MLP、LightCNN、ResNet）和不同预训练策略（无预训练、SimCLR、MAE、ImageNet）对人口增长率预测性能的影响。

## 项目结构

```
Baseline_Pretrain/
├── config.py                      # 所有配置（路径、模型、训练参数、实验预设）
├── dataset.py                     # 数据集类和数据加载器
├── train.py                       # 统一训练脚本
├── evaluate.py                    # 模型评估脚本
├── compare_results.py             # 实验结果对比分析
├── utils.py                       # 工具函数（指标计算、日志、检查点等）
├── preprocess_patches.py          # 数据预处理：TIFF → 25个patch的npy文件
├── preprocess_individual_patches.py  # 数据预处理：TIFF → 25个独立patch文件
├── requirements.txt               # 项目依赖
├── models/
│   ├── __init__.py
│   ├── mlp_model.py               # MLP 模型
│   ├── light_cnn.py               # 轻量级 CNN 模型
│   ├── resnet_baseline.py         # ResNet 基线模型
│   └── aggregators.py             # 位置感知聚合模块
├── pretrain/
│   ├── __init__.py
│   ├── simclr.py                  # SimCLR 对比学习预训练
│   └── mae.py                     # MAE 掩码自编码器预训练
├── scripts/                       # 实验运行脚本
│   ├── run_all_experiments.sh     # 串行运行所有实验
│   ├── run_all_experiments_parallel.sh  # 并行运行实验
│   ├── run_experiments.sh         # 灵活实验脚本
│   ├── run_patch_experiments.sh   # Patch-level 实验
│   ├── run_patch_level.sh         # Patch-level 单实验
│   └── start_experiments.sh       # tmux 启动实验
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
python train.py --exp mlp_baseline --gpu 3 --seed 123  # 使用不同 seed

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
```

### 4. 评估模型

```bash
python evaluate.py --exp mae_cnn --gpu 0 --split test
```

### 5. 对比所有实验结果

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

---

## 实验总览

当前共有 **80 个实验配置**，按模型类型组织如下：

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

> **评估说明**：
> - **City-level**：训练和评估使用相同的模型内置聚合方式（mean/median/trimmed_mean 或高级聚合）
> - **Patch-level**：评估时自动测试 3 种聚合方式（mean/median/trimmed_mean）

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

**注意**：结果会保存到 `results/{exp_name}_results.pkl`，其中包含使用的 seed 值。

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

**输出**:
- `checkpoints/{exp_name}/best_model.pth`: 最佳模型
- `logs/{exp_name}_{timestamp}.log`: 训练日志
- `results/{exp_name}_results.pkl`: 完整结果

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
