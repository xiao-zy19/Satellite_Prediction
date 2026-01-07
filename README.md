# MLLM Training for 64-Channel Satellite Embeddings

This project implements a complete training pipeline for adapting Multimodal Large Language Models (MLLMs) to work with 64-channel satellite embeddings from Google Earth Engine.

## Overview

The pipeline supports training MLLMs (Qwen2-VL, Qwen2.5-VL) on high-dimensional satellite feature data for downstream tasks such as population growth prediction. It features:

- **64-Channel Input Support**: Modified vision encoder to accept 64-channel satellite embeddings instead of standard 3-channel RGB
- **MAE Pretraining**: Self-supervised learning using Masked Autoencoder for domain adaptation
- **Multi-stage Training**: Pretrain → SFT → RL/DPO pipeline
- **Data Augmentation**: Strong augmentation strategies optimized for small datasets (~660 samples → ~33,000 effective samples)

## Training Pipeline

### Stage 1: MAE Pretraining (Self-Supervised)

Uses Masked Autoencoder (MAE) to pretrain the vision encoder on unlabeled 64-channel satellite data:

- Randomly masks 75% of image patches
- Trains encoder to reconstruct masked regions
- Helps the model learn meaningful representations of satellite features
- Supports both Qwen2-VL (Conv2d) and Qwen2.5-VL (Conv3d) architectures

### Stage 2: Supervised Fine-Tuning (SFT)

Fine-tunes the pretrained model on labeled data for specific tasks:

- Population growth prediction from satellite imagery
- Staged training: freeze encoder initially, then unfreeze with lower learning rate
- Early stopping based on validation R² score

### Stage 3: Reinforcement Learning (DPO)

Optional DPO (Direct Preference Optimization) training for improved prediction accuracy:

- Uses preference pairs to further optimize the model
- Requires generating preference data from SFT model first

## Project Structure

```
MLLM_Training/
├── configs/                    # Configuration files
│   ├── pretrain_config.yaml    # MAE pretraining config
│   ├── sft_config.yaml         # SFT config
│   ├── rl_config.yaml          # DPO config
│   └── data_paths.yaml         # Data path configuration
├── data/                       # Data loading modules
│   ├── pretrain_dataset.py     # MAE pretraining dataset with augmentation
│   ├── sft_dataset.py          # SFT dataset
│   ├── dpo_dataset.py          # DPO preference dataset
│   ├── tiff_dataset.py         # TIFF/NPY data loading
│   └── pca_converter.py        # PCA for 64ch→3ch conversion (Route B)
├── models/                     # Model definitions
│   ├── qwen_vl_64ch.py         # Modified Qwen2-VL for 64ch input
│   ├── qwen25_vl_64ch.py       # Modified Qwen2.5-VL for 64ch input
│   ├── mae_decoder.py          # MAE decoder and full model
│   ├── regression_head.py      # Regression head for numerical prediction
│   └── model_utils.py          # Model utilities
├── training/                   # Training modules
│   ├── pretrain_mae.py         # MAE trainer
│   ├── sft_trainer.py          # SFT trainer
│   ├── dpo_trainer.py          # DPO trainer
│   └── training_utils.py       # Training utilities (optimizer, scheduler)
├── evaluation/                 # Evaluation modules
│   └── evaluate.py             # Model evaluation with metrics
├── scripts/                    # Utility scripts
│   ├── preprocess_tiff.py      # TIFF→NPY preprocessing
│   └── compute_pca.py          # PCA model training
├── pretrain.py                 # MAE Pretraining entry point
├── sft.py                      # SFT entry point
├── rl.py                       # RL/DPO entry point
├── utils.py                    # Utility functions
└── requirements.txt            # Dependencies
```

## Key Technical Features

### 1. Vision Encoder Modification

The core innovation is modifying the MLLM's vision encoder to accept 64-channel input:

**Qwen2-VL (Conv2d)**:
```
Original: Conv2d(3, embed_dim, kernel=[14,14])
Modified: Conv2d(64, embed_dim, kernel=[14,14])
```

**Qwen2.5-VL (Conv3d)**:
```
Original: Conv3d(3, 1280, kernel=[2,14,14])
Modified: Conv3d(64, 1280, kernel=[2,14,14])
```

Weight initialization options:
- `normal`: Random normal initialization for new channels
- `zero`: Zero initialization
- `copy`: Tile from original RGB weights

### 2. MAE Architecture

The MAE implementation follows the original paper with adaptations:

- **Encoder**: Uses MLLM's vision encoder (patch embed only, or full encoder)
- **Decoder**: Lightweight transformer decoder (4 layers, 512 dim)
- **Mask Ratio**: 75% (standard MAE setting)
- **Loss**: MSE on normalized pixel values of masked patches

### 3. Data Augmentation for Small Datasets

Optimized for ~660 city samples with extensive augmentation:

- **Spatial augmentations**: Random flip, rotation (90°/180°/270°)
- **Multi-scale cropping**: 3 scales (1.0, 0.8, 0.6)
- **Per-file sampling**: 50 augmented samples per original file
- **Online augmentation**: Different augmentation each epoch

### 4. Route B: PCA Approach

Alternative approach using PCA dimensionality reduction:

```python
# Fit PCA model
python scripts/compute_pca.py --input /path/to/data --output cache/pca_64to3.pkl

# Convert dataset
python data/pca_converter.py --action convert \
    --input-dir /path/to/64ch_data \
    --output-dir /path/to/rgb_data \
    --model-path cache/pca_64to3.pkl
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Transformers >= 4.37.0 (4.45.0+ for Qwen2.5-VL)
- CUDA-capable GPU with at least 24GB VRAM

## Quick Start

### 1. Configure Data Paths

Edit `configs/data_paths.yaml` to set your data directories:

```yaml
data:
  satellite_data_dir: "/path/to/your/satellite_tiles"
  population_labels: "/path/to/your/population_data.xlsx"
```

### 2. MAE Pretraining

```bash
python pretrain.py \
    --model-type qwen2.5-vl \
    --data-dir /path/to/satellite_tiles \
    --output-dir checkpoints/pretrain \
    --batch-size 4 \
    --epochs 10 \
    --mask-ratio 0.75 \
    --use-wandb
```

Key arguments:
- `--model-type`: `qwen2-vl` or `qwen2.5-vl`
- `--samples-per-file`: Number of augmented samples per file (default: 50)
- `--no-augment`: Disable data augmentation
- `--debug`: Use simplified model for debugging

### 3. Supervised Fine-Tuning

```bash
python sft.py \
    --data-dir /path/to/satellite_tiles \
    --labels-file /path/to/population_data.xlsx \
    --pretrain-checkpoint checkpoints/pretrain/best.pt \
    --output-dir checkpoints/sft \
    --batch-size 4 \
    --epochs 20 \
    --freeze-encoder-epochs 5
```

### 4. DPO Training (Optional)

```bash
python rl.py \
    --data-dir /path/to/satellite_tiles \
    --preference-file data/preferences.json \
    --sft-checkpoint checkpoints/sft/best.pt \
    --output-dir checkpoints/rl \
    --beta 0.1
```

### 5. Evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/sft/best.pt \
    --data-dir /path/to/satellite_tiles \
    --labels-file /path/to/population_data.xlsx \
    --output results/evaluation.json
```

## Model Configurations

### Qwen2-VL (7B)
- Vision encoder: 1024-dim
- Patch embedding: Conv2d
- Patch size: 14×14

### Qwen2.5-VL (7B)
- Vision encoder: 1280-dim
- Patch embedding: Conv3d (temporal_patch_size=2)
- Patch size: 14×14
- Window + Full attention hybrid

## Hardware Requirements

| Stage    | GPU Memory | Recommended GPUs |
|----------|------------|------------------|
| Pretrain | ~40GB      | 1× A100 80GB     |
| SFT      | ~60GB      | 2× A100 80GB     |
| DPO      | ~80GB      | 4× A100 80GB     |

For limited GPU memory:
- Use `--batch-size 1` or `2`
- Enable gradient checkpointing
- Use `--precision bf16` (default)
- Consider using LoRA (PEFT)

## Monitoring

### Weights & Biases
```bash
python pretrain.py --use-wandb --wandb-project your-project-name
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

## Citation

If you use this code, please cite:

```bibtex
@misc{mllm_satellite_training,
  title={MLLM Training for 64-Channel Satellite Embeddings},
  author={Your Name},
  year={2024},
  url={https://github.com/xiao-zy19/Satellite_Prediction}
}
```

## License

This project is for research purposes only. Please refer to the licenses of the underlying models (Qwen2-VL, Qwen2.5-VL) for usage restrictions.

## Acknowledgments

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba
- [MAE](https://arxiv.org/abs/2111.06377) by Facebook AI Research
- Google Earth Engine for satellite data
