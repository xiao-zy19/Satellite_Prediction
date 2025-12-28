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
├── run_all_experiments.sh         # 批量运行所有实验
├── start_experiments.sh           # 使用tmux启动实验
├── requirements.txt               # 项目依赖
├── models/
│   ├── __init__.py
│   ├── mlp_model.py               # MLP 模型
│   ├── light_cnn.py               # 轻量级 CNN 模型
│   └── resnet_baseline.py         # ResNet 基线模型
├── pretrain/
│   ├── __init__.py
│   ├── simclr.py                  # SimCLR 对比学习预训练
│   └── mae.py                     # MAE 掩码自编码器预训练
├── checkpoints/                   # 模型检查点
├── logs/                          # 训练日志
│   └── experiment_runs/           # 实验运行日志
├── results/                       # 实验结果
└── wandb/                         # Wandb 日志
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
# 基线模型（无预训练）
python train.py --exp mlp_baseline --gpu 3
python train.py --exp light_cnn_baseline --gpu 3

# Patch-level 训练模式
python train.py --exp mlp_patch_level --gpu 3

# 带 SimCLR 预训练
python train.py --exp simclr_cnn --gpu 3

# 带 MAE 预训练
python train.py --exp mae_cnn --gpu 3

# ImageNet 预训练的 ResNet
python train.py --exp resnet_imagenet --gpu 3
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

| 格式 | 路径 | 形状 | 用途 |
|-----|------|------|------|
| 原始 TIFF | `city_satellite_tiles/` | (64, 1000, 1000) | 原始数据 |
| 合并 NPY | `city_patches/` | (25, 64, 200, 200) | City-level 训练 |
| 独立 NPY | `city_individual_patches/` | (64, 200, 200) × 25 | Patch-level 训练 |

---

## 训练模式

### City-level 训练（默认）

每个样本包含城市的 25 个 patch，模型内部聚合后输出单个预测值。

```
输入: (batch, 25, 64, 200, 200) → 模型聚合 → 输出: (batch, 1)
```

**适用实验**: `mlp_baseline`, `light_cnn_baseline`, `resnet_baseline`, `simclr_cnn`, `mae_cnn` 等

### Patch-level 训练

每个 patch 作为独立样本训练，推理时聚合 25 个预测。数据量扩大 25 倍。

```
训练: 每个patch独立预测 → (batch, 64, 200, 200) → (batch, 1)
推理: 25个patch预测 → trimmed_mean聚合 → 最终预测
```

**适用实验**: `mlp_patch_level`, `light_cnn_patch_level`, `resnet_patch_level`, `simclr_cnn_patch_level`

---

## 可用实验

| 实验名称 | 模型 | 预训练 | 训练模式 | 说明 |
|---------|------|--------|---------|------|
| `mlp_baseline` | MLP | 无 | city_level | MLP 基线 |
| `light_cnn_baseline` | LightCNN | 无 | city_level | 轻量级 CNN 基线 |
| `resnet_baseline` | ResNet18 | 无 | city_level | ResNet 从头训练 |
| `resnet_imagenet` | ResNet18 | ImageNet | city_level | ImageNet 预训练权重 |
| `simclr_mlp` | MLP | SimCLR | city_level | 对比学习预训练 MLP |
| `simclr_cnn` | LightCNN | SimCLR | city_level | 对比学习预训练 CNN |
| `mae_cnn` | LightCNN | MAE | city_level | 掩码自编码器预训练 CNN |
| `mlp_patch_level` | MLP | 无 | patch_level | Patch级别训练 MLP |
| `light_cnn_patch_level` | LightCNN | 无 | patch_level | Patch级别训练 CNN |
| `resnet_patch_level` | ResNet18 | 无 | patch_level | Patch级别训练 ResNet |
| `simclr_cnn_patch_level` | LightCNN | SimCLR | patch_level | SimCLR + Patch级别 |

---

## 模型架构

### MLP Model (`models/mlp_model.py`)

```
输入 patch (64, 200, 200)
    ↓ Global Average Pooling
特征向量 (64,)
    ↓ MLP: 64 → 256 → 128 → 64
    ↓ Aggregation (mean/attention/trimmed_mean) [city_level only]
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

```
输入 patch (64, 200, 200)
    ↓ Modified ResNet18 (首层conv适配64通道)
特征向量 (512,)
    ↓ Aggregation [city_level only]
    ↓ Regression Head: 512 → 256 → 128 → 1
输出 (1,)
```

**参数量**: ~11M

### 聚合方式

| 方式 | 说明 |
|-----|------|
| `mean` | 简单平均 |
| `attention` | 注意力加权平均 |
| `trimmed_mean` | 去除极端值后平均（默认去除10%） |

---

## 自监督预训练

### SimCLR (`pretrain/simclr.py`)

**原理**: 对比学习，最大化同一样本不同增强视图的相似性。

```
样本 → 增强1, 增强2 → 编码器 → 投影头 → NT-Xent Loss
```

**增强策略**:
- 随机水平/垂直翻转
- 随机90度旋转
- 高斯噪声

**配置**:
- Temperature: 0.5
- Projection dim: 128
- 预训练轮数: 50

### MAE (`pretrain/mae.py`)

**原理**: 掩码自编码器，通过重建被遮蔽的 patch 学习表示。

```
25个patch → 随机遮蔽75% → 编码可见patch → 解码器重建 → MSE Loss
```

**特点**: 使用与下游模型相同的编码器架构（LightCNNEncoder/PatchMLP），确保权重正确迁移。

**配置**:
- Mask ratio: 0.75
- Decoder dim: 256
- Decoder depth: 2

---

## 训练流程

### 基线模型流程

```
数据加载 → 模型初始化 → 训练100轮 → Early Stopping → 测试集评估
```

### 自监督预训练流程

```
1. 预训练阶段 (50轮)
   无标签数据 → SimCLR/MAE → 学习表示

2. 权重迁移
   预训练编码器权重 → 下游模型编码器

3. 冻结编码器阶段 (5轮)
   冻结编码器，只训练回归头

4. 全量微调 (95轮)
   解冻编码器，端到端训练

5. 评估
   测试集评估，保存最佳模型
```

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
```

---

## 评估指标

| 指标 | 说明 |
|-----|------|
| **Pearson r** | 皮尔逊相关系数，衡量预测与真值的线性相关性 |
| **R²** | 决定系数，模型解释方差的比例 |
| **MAE** | 平均绝对误差 |
| **RMSE** | 均方根误差 |

---

## 脚本使用

### 训练 (`train.py`)

```bash
python train.py --exp <experiment_name> --gpu <gpu_id>
```

**参数**:
- `--exp`: 实验名称（必需）
- `--gpu`: GPU ID（默认: 3）

**输出**:
- `checkpoints/{exp_name}/best_model.pth`: 最佳模型
- `checkpoints/{exp_name}_pretrain/`: 预训练检查点（如有）
- `logs/{exp_name}_{timestamp}.log`: 训练日志
- `results/{exp_name}_results.pkl`: 完整结果

### 评估 (`evaluate.py`)

```bash
python evaluate.py --exp <experiment_name> [options]
```

**参数**:
- `--exp`: 实验名称
- `--checkpoint`: 直接指定检查点路径
- `--model`: 模型类型（使用 `--checkpoint` 时需要）
- `--split`: 数据集划分（train/val/test，默认: test）
- `--output_dir`: 输出目录

**输出**:
- `predictions_{split}.csv`: 预测结果CSV
- `predictions_{split}.png`: 预测散点图
- `residuals_{split}.png`: 残差分析图
- `eval_results_{split}.pkl`: 完整评估结果

### 结果对比 (`compare_results.py`)

```bash
python compare_results.py
```

**输出**:
- `results/comparison_table.csv`: 所有实验指标对比表
- `results/comparison_plot.png`: 指标柱状图和训练曲线
- `results/predictions_plot.png`: 各实验预测散点图

### 数据预处理

```bash
# 预处理为合并的25-patch文件
python preprocess_patches.py --input_dir <tiff_dir> --output_dir <npy_dir> --workers 8

# 预处理为独立patch文件（推荐用于patch-level训练）
python preprocess_individual_patches.py
```

---

## 批量实验

### 使用 tmux 运行

```bash
# 启动3个并行实验（推荐）
bash start_experiments.sh

# 查看实验
tmux attach -t exp_all

# 分离: Ctrl+B 然后 D
```

### 运行所有11个实验

```bash
# 分4批运行所有实验
./run_all_experiments.sh
```

---

## 实验监控

### Wandb 集成

实验自动记录到 Wandb（需要登录）:
- 项目: `population-pretrain-comparison`
- 记录: 训练/验证 loss、Pearson r、R²、MAE、学习率

查看: https://wandb.ai/xiao-zy19/population-pretrain-comparison

### 日志查看

```bash
# 查看实时日志
tail -f logs/experiment_runs/<exp_name>.log

# 查看训练日志
cat logs/<exp_name>_<timestamp>.log
```

---

## 输出文件

### 检查点 (`checkpoints/`)

```
checkpoints/
├── mlp_baseline/
│   └── best_model.pth
├── simclr_cnn/
│   └── best_model.pth
├── simclr_cnn_pretrain/
│   └── simclr_pretrain_best.pth
├── mae_cnn/
│   └── best_model.pth
└── mae_cnn_pretrain/
    └── mae_pretrain_best.pth
```

### 结果 (`results/`)

```
results/
├── mlp_baseline_results.pkl      # 单实验结果
├── light_cnn_baseline_results.pkl
├── ...
├── comparison_table.csv          # 对比表格
├── comparison_plot.png           # 对比图
└── predictions_plot.png          # 预测散点图
```

### 评估输出

```
results/eval_mae_cnn/
├── predictions_test.csv
├── predictions_test.png
├── residuals_test.png
└── eval_results_test.pkl
```

---

## 工具函数 (`utils.py`)

| 函数 | 功能 |
|-----|------|
| `set_seed(seed)` | 设置随机种子，确保可复现 |
| `get_device(device_str)` | 获取计算设备 |
| `setup_logging(exp_name)` | 配置日志记录 |
| `get_optimizer(model, lr, wd)` | 创建 AdamW 优化器 |
| `get_scheduler(optimizer, config)` | 创建学习率调度器 |
| `compute_metrics(y_true, y_pred)` | 计算评估指标 |
| `EarlyStopping` | 早停机制 |
| `AverageMeter` | 计算运行平均值 |
| `save_checkpoint(state, exp_name)` | 保存模型检查点 |
| `count_parameters(model)` | 统计模型参数量 |

---

## 数据集类 (`dataset.py`)

| 类 | 用途 |
|---|------|
| `CityDataset` | City-level 数据集，每个样本25个patch |
| `PatchLevelDataset` | Patch-level 数据集，每个patch独立 |
| `PretrainDataset` | 自监督预训练数据集（无标签） |

**数据加载函数**:

```python
# City-level
train_loader, val_loader, test_loader, info = get_dataloaders(batch_size=16)

# Patch-level
train_loader, val_loader, test_loader, info = get_patch_level_dataloaders(batch_size=64)

# 预训练
pretrain_loader = get_pretrain_dataloader(batch_size=16, contrastive=True)
```

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

### v2.1 (2024-12-28)
- **启用 Wandb**: 实验自动记录到 wandb 平台
- **新增 Patch-level 训练模式**: 支持 `mlp_patch_level`, `light_cnn_patch_level`, `resnet_patch_level`, `simclr_cnn_patch_level`
- **新增预处理脚本**: `preprocess_individual_patches.py` 用于独立patch文件
- **更新文档**: 完善 README，添加详细的使用说明

### v2.0 (2024-12-26)
- **修复 MAE 权重迁移**: MAE 使用与下游模型相同的编码器架构
- **新增检查点保存**: SimCLR 和 MAE 保存最佳预训练模型
- **新增 evaluate.py**: 独立评估脚本，支持详细指标和可视化
- **新增 compare_results.py**: 实验结果对比分析
- **改进导入处理**: 使用绝对路径导入

### v1.0 (2024-12-25)
- 初始版本
- 支持 MLP、LightCNN、ResNet 基线模型
- 支持 SimCLR、MAE 自监督预训练
