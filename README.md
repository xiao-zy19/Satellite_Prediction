# Baseline Pretrain Experiments

Population growth rate prediction using different model architectures and pretraining strategies.

## Project Structure

```
Baseline_Pretrain/
├── config.py              # All configurations
├── dataset.py             # Dataset and dataloader
├── utils.py               # Utility functions
├── train.py               # Unified training script
├── evaluate.py            # Model evaluation script
├── compare_results.py     # Compare experiment results
├── run_all_experiments.sh # Run all experiments
├── requirements.txt       # Project dependencies
├── models/
│   ├── mlp_model.py       # MLP baseline
│   ├── light_cnn.py       # Lightweight CNN
│   └── resnet_baseline.py # ResNet baseline
├── pretrain/
│   ├── simclr.py          # SimCLR self-supervised
│   └── mae.py             # Masked Autoencoder
├── checkpoints/           # Saved models
├── logs/                  # Training logs
└── results/               # Experiment results
```

## Quick Start

### 1. Installation

```bash
cd /home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain
pip install -r requirements.txt
```

### 2. Run a single experiment

```bash
# Basic baseline (no pretraining)
python train.py --exp light_cnn_baseline --gpu 3

# With SimCLR pretraining
python train.py --exp simclr_cnn --gpu 3

# With MAE pretraining
python train.py --exp mae_cnn --gpu 3
```

### 3. Run all experiments at once

```bash
# In tmux session (recommended for long runs)
tmux new-session -s pretrain_exp
./run_all_experiments.sh 3  # Use GPU 3
```

### 4. Evaluate trained models

```bash
python evaluate.py --exp mae_cnn --gpu 0
```

### 5. Compare all results

```bash
python compare_results.py
```

---

## Available Experiments

| Experiment | Model | Pretraining | Description |
|------------|-------|-------------|-------------|
| `mlp_baseline` | MLP | None | Simple MLP on pooled features |
| `light_cnn_baseline` | LightCNN | None | Lightweight CNN |
| `resnet_baseline` | ResNet18 | None | ResNet from scratch |
| `resnet_imagenet` | ResNet18 | ImageNet | ResNet with ImageNet weights |
| `simclr_mlp` | MLP | SimCLR | Contrastive pretrained MLP |
| `simclr_cnn` | LightCNN | SimCLR | Contrastive pretrained CNN |
| `mae_cnn` | LightCNN | MAE | Masked autoencoder pretrained |

---

## Detailed Usage

### Training (`train.py`)

```bash
python train.py --exp <experiment_name> --gpu <gpu_id>
```

**Arguments:**
- `--exp`: Experiment name (required). See available experiments above.
- `--gpu`: GPU ID to use (default: 0)

**Examples:**

```bash
# Train MLP baseline on GPU 0
python train.py --exp mlp_baseline --gpu 0

# Train LightCNN with SimCLR pretraining on GPU 3
python train.py --exp simclr_cnn --gpu 3

# Train LightCNN with MAE pretraining on GPU 2
python train.py --exp mae_cnn --gpu 2

# Train ResNet with ImageNet pretrained weights
python train.py --exp resnet_imagenet --gpu 1
```

**Output:**
- Training logs: `logs/{exp_name}_{timestamp}.log`
- Best model: `checkpoints/{exp_name}/best_model.pth`
- Pretrain checkpoint: `checkpoints/{exp_name}_pretrain/` (if using pretraining)
- Results: `results/{exp_name}_results.pkl`
- Wandb: https://wandb.ai/your-username/population-pretrain-comparison

---

### Evaluation (`evaluate.py`)

```bash
python evaluate.py --exp <experiment_name> [options]
```

**Arguments:**
- `--exp`: Experiment name (load from `checkpoints/{exp_name}/best_model.pth`)
- `--checkpoint`: Alternative: direct path to checkpoint file
- `--model`: Model type when using `--checkpoint` (`mlp`, `light_cnn`, `resnet`)
- `--gpu`: GPU ID (default: 0)
- `--split`: Data split to evaluate (`train`, `val`, `test`, default: `test`)
- `--output_dir`: Output directory for results

**Examples:**

```bash
# Evaluate on test set
python evaluate.py --exp mae_cnn --gpu 0

# Evaluate on validation set
python evaluate.py --exp mae_cnn --split val --gpu 0

# Evaluate custom checkpoint
python evaluate.py --checkpoint checkpoints/my_model/best.pth --model light_cnn --gpu 0

# Specify output directory
python evaluate.py --exp mae_cnn --output_dir results/eval_mae --gpu 0
```

**Output:**
- `predictions_{split}.csv`: Predictions with city, year, true/predicted values, errors
- `predictions_{split}.png`: Scatter plot of predicted vs true values
- `residuals_{split}.png`: Residual analysis plots
- `eval_results_{split}.pkl`: Full evaluation results

---

### Compare Results (`compare_results.py`)

```bash
python compare_results.py
```

**Output:**
- `results/comparison_table.csv`: Summary metrics for all experiments
- `results/comparison_plot.png`: Bar charts and training curves
- `results/predictions_plot.png`: Scatter plots for all experiments

---

### Batch Run (`run_all_experiments.sh`)

```bash
./run_all_experiments.sh [GPU_ID]
```

**Default GPU:** 3

This script runs all 7 experiments sequentially and generates comparison results at the end.

**Recommended:** Run in a tmux session for long experiments:

```bash
# Start tmux session
tmux new-session -s pretrain_exp

# Run experiments
./run_all_experiments.sh 3

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t pretrain_exp
```

---

## Model Architectures

### MLP Model
- Global average pooling on each patch (200x200 → 64-dim vector)
- MLP: 64 → 256 → 128 → 64
- Mean aggregation across 25 patches
- Regression head: 64 → 32 → 1

### LightCNN Model
- 3 conv blocks: 64 → 32 → 64 → 128 (each with MaxPool)
- Global average pooling
- FC: 128 → 256 → 64
- Regression head

### ResNet Baseline
- Modified ResNet18 for 64-channel input
- First conv: 64 channels instead of 3
- Patch-level encoding (25 patches per city)
- Mean/Attention aggregation
- Regression head

---

## Self-Supervised Pretraining

### SimCLR (Contrastive Learning)

**Principle:** Learn representations by maximizing agreement between different augmented views of the same sample.

**Process:**
1. Create two augmented views of each sample (flip, rotate, noise)
2. Encode both views
3. Project to 128-dim space
4. Minimize NT-Xent loss (push positive pairs together, negative pairs apart)

**Config:** `SimCLRConfig` in `config.py`
- Temperature: 0.5
- Projection dim: 128
- Pretrain epochs: 50

### MAE (Masked Autoencoder)

**Principle:** Learn representations by reconstructing randomly masked patches.

**Process:**
1. Randomly mask 75% of patches
2. Encode only visible patches
3. Decode to reconstruct all patches
4. MSE loss on masked patches only

**Key Feature:** Uses the **same encoder architecture** as downstream models (LightCNN/MLP), enabling proper weight transfer.

**Config:** `MAEConfig` in `config.py`
- Mask ratio: 0.75
- Decoder dim: 256
- Decoder depth: 2

---

## Training Pipeline

### For baseline experiments (no pretraining):
```
Data → Model → Train (100 epochs) → Evaluate
```

### For self-supervised experiments:
```
1. Pretraining Phase (50 epochs)
   Data → SimCLR/MAE → Learn representations (no labels)

2. Weight Transfer
   Pretrained encoder weights → Downstream model encoder

3. Frozen Encoder Phase (5 epochs)
   Train only regression head with frozen encoder

4. Full Finetuning (95 epochs)
   Train entire model end-to-end

5. Evaluate on test set
```

---

## Configuration

All configurations are in `config.py`. Key parameters:

```python
# Training
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
patience = 15  # Early stopping

# Pretraining
pretrain_epochs = 50
pretrain_lr = 1e-3
freeze_encoder_epochs = 5

# Data
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20
```

---

## Output Files

### Checkpoints (`checkpoints/`)
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

### Results (`results/`)
```
results/
├── mlp_baseline_results.pkl
├── simclr_cnn_results.pkl
├── mae_cnn_results.pkl
├── comparison_table.csv
├── comparison_plot.png
└── predictions_plot.png
```

### Evaluation Output
```
results/eval_mae_cnn/
├── predictions_test.csv
├── predictions_test.png
├── residuals_test.png
└── eval_results_test.pkl
```

---

## Changelog

### v2.0 (2024-12-26 Fixed Version)
- **Fixed MAE weight transfer**: MAE now uses the same encoder architecture (LightCNNEncoder/PatchMLP) as downstream models, enabling proper weight transfer
- **Added checkpoint saving**: Both SimCLR and MAE save best pretrained models
- **Improved import handling**: Using absolute paths for more robust imports
- **Added evaluate.py**: Standalone evaluation script with detailed metrics and visualizations
- **Added requirements.txt**: Project dependencies
- **Expanded documentation**: Detailed usage instructions in README
