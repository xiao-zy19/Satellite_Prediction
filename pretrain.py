#!/usr/bin/env python3
"""
MAE Pretraining entry point for 64-channel MLLM.

支持:
- Qwen2-VL (Conv2d patch embedding)
- Qwen2.5-VL (Conv3d patch embedding)

针对小数据集（~660样本）优化，通过数据增强扩展到数万样本。
"""

import argparse
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAE Pretraining for 64-channel MLLM (Qwen2-VL / Qwen2.5-VL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config
    parser.add_argument("--config", type=str, default="configs/pretrain_config.yaml",
                        help="Path to config file")

    # Model type
    parser.add_argument("--model-type", type=str, default="qwen2.5-vl",
                        choices=["qwen2-vl", "qwen2.5-vl"],
                        help="Base model type to use")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default="./data",
                        help="Directory containing TIFF/NPY data files")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/pretrain",
                        help="Output directory for checkpoints")

    # Training
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Number of warmup epochs")

    # MAE specific
    parser.add_argument("--mask-ratio", type=float, default=0.75,
                        help="Mask ratio for MAE (default 0.75 = 75%% patches masked)")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Patch size for vision encoder")
    parser.add_argument("--temporal-patch-size", type=int, default=2,
                        help="Temporal patch size (for Qwen2.5-VL)")
    parser.add_argument("--image-size", type=int, default=980,
                        help="Input image size (should be divisible by patch-size)")

    # Data augmentation
    parser.add_argument("--samples-per-file", type=int, default=50,
                        help="Number of augmented samples per original file")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--no-multi-crop", action="store_true",
                        help="Disable multi-scale random cropping")

    # Model
    parser.add_argument("--model-name", type=str, default=None,
                        help="Base model name (auto-detected if not specified)")
    parser.add_argument("--input-channels", type=int, default=64,
                        help="Number of input channels")
    parser.add_argument("--encoder-dim", type=int, default=None,
                        help="Encoder dimension (auto-detected if not specified)")

    # Hardware
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Training precision")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    # Logging
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="mllm-satellite-pretrain",
                        help="Wandb project name")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N steps")

    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode with reduced data and simplified model")

    return parser.parse_args()


def get_model_config(model_type: str) -> dict:
    """Get default configuration for each model type."""
    configs = {
        "qwen2-vl": {
            "model_name": "Qwen/Qwen2-VL-7B-Instruct",
            "encoder_dim": 1024,
            "temporal_patch_size": 1,
            "conv_type": "conv2d",
        },
        "qwen2.5-vl": {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "encoder_dim": 1280,
            "temporal_patch_size": 2,
            "conv_type": "conv3d",
        },
    }
    return configs.get(model_type, configs["qwen2.5-vl"])


def main():
    args = parse_args()

    # Load config file if exists
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = {}
        logger.warning(f"Config file {config_path} not found, using defaults")

    # Get model-specific config
    model_config = get_model_config(args.model_type)

    # Override config with command line arguments
    config.update({
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'mask_ratio': args.mask_ratio,
        'patch_size': args.patch_size,
        'temporal_patch_size': args.temporal_patch_size or model_config['temporal_patch_size'],
        'image_size': args.image_size,
        'samples_per_file': args.samples_per_file,
        'augment': not args.no_augment,
        'use_multi_crop': not args.no_multi_crop,
        'num_workers': args.num_workers,
        'precision': args.precision,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'model_name': args.model_name or model_config['model_name'],
        'encoder_dim': args.encoder_dim or model_config['encoder_dim'],
        'conv_type': model_config['conv_type'],
    })

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {output_dir / 'config.yaml'}")

    # Setup logging to file
    file_handler = logging.FileHandler(output_dir / 'pretrain.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"MAE Pretraining for {args.model_type.upper()}")
    logger.info("=" * 60)

    # Import model and training components
    import torch
    from data.pretrain_dataset import get_pretrain_dataloader, analyze_dataset
    from training.pretrain_mae import MAETrainer
    from training.training_utils import set_seed

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Analyze dataset
    logger.info(f"Data directory: {args.data_dir}")
    stats = analyze_dataset(args.data_dir)
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Original files: {stats['num_files']}")
    logger.info(f"  - Data shape: {stats['sample_shapes'][0]}")
    logger.info(f"  - Mean: {stats['mean_of_means']:.2f}")
    logger.info(f"  - Std: {stats['mean_of_stds']:.2f}")

    # Create data loader
    logger.info("Creating data loader...")
    train_loader = get_pretrain_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_ratio=args.mask_ratio,
        samples_per_file=args.samples_per_file,
        patch_size=args.patch_size,
        image_size=args.image_size,
        augment=not args.no_augment,
        use_multi_crop=not args.no_multi_crop,
    )

    total_samples = len(train_loader.dataset)
    logger.info(f"Training data:")
    logger.info(f"  - Total samples (after augmentation): {total_samples:,}")
    logger.info(f"  - Samples per file: {args.samples_per_file}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Steps per epoch: {len(train_loader):,}")

    # Initialize model
    logger.info("Initializing model...")
    logger.info(f"  - Model type: {args.model_type}")
    logger.info(f"  - Base model: {config['model_name']}")
    logger.info(f"  - Input channels: {args.input_channels}")
    logger.info(f"  - Encoder dim: {config['encoder_dim']}")

    if args.debug:
        # Debug mode: use simplified model
        logger.warning("Debug mode: Using simplified model")
        from models.mae_decoder import SimpleMAEForPretraining

        mae_model = SimpleMAEForPretraining(
            input_channels=args.input_channels,
            patch_size=args.patch_size,
            temporal_patch_size=config['temporal_patch_size'],
            image_size=args.image_size,
            encoder_dim=256,
            decoder_dim=128,
            mask_ratio=args.mask_ratio,
            conv_type=config['conv_type'],
        )
    else:
        # Full model
        if args.model_type == "qwen2.5-vl":
            from models.qwen25_vl_64ch import Qwen25VL64Ch
            from models.mae_decoder import create_mae_for_qwen25_vl

            base_model = Qwen25VL64Ch(
                model_name=config['model_name'],
                input_channels=args.input_channels,
                patch_size=args.patch_size,
                temporal_patch_size=config['temporal_patch_size'],
            )

            mae_model = create_mae_for_qwen25_vl(
                vision_encoder=base_model.get_vision_encoder(),
                encoder_dim=config['encoder_dim'],
                decoder_dim=config.get('decoder_dim', 512),
                decoder_depth=config.get('decoder_depth', 4),
                patch_size=args.patch_size,
                image_size=args.image_size,
                input_channels=args.input_channels,
                mask_ratio=args.mask_ratio,
            )
        else:
            # Qwen2-VL
            from models.qwen_vl_64ch import Qwen2VL64Ch
            from models.mae_decoder import create_mae_for_qwen2_vl

            base_model = Qwen2VL64Ch(
                model_name=config['model_name'],
                input_channels=args.input_channels,
            )

            mae_model = create_mae_for_qwen2_vl(
                vision_encoder=base_model.get_vision_encoder(),
                encoder_dim=config['encoder_dim'],
                decoder_dim=config.get('decoder_dim', 512),
                decoder_depth=config.get('decoder_depth', 4),
                patch_size=args.patch_size,
                image_size=args.image_size,
                input_channels=args.input_channels,
                mask_ratio=args.mask_ratio,
            )

    # Count parameters
    total_params = sum(p.numel() for p in mae_model.parameters())
    trainable_params = sum(p.numel() for p in mae_model.parameters() if p.requires_grad)
    logger.info(f"Model parameters:")
    logger.info(f"  - Total: {total_params:,}")
    logger.info(f"  - Trainable: {trainable_params:,}")

    # Create trainer
    # 从配置文件中读取 save_total_limit，默认保留5个step checkpoint
    save_total_limit = config.get('checkpoint', {}).get('save_total_limit', 5)
    logger.info(f"Checkpoint cleanup: keeping latest {save_total_limit} step checkpoints + all epoch checkpoints")

    trainer = MAETrainer(
        model=mae_model,
        train_loader=train_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_epochs=args.epochs,
        max_grad_norm=config.get('max_grad_norm', 1.0),
        save_dir=str(output_dir),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_total_limit=save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        precision=args.precision,
    )

    logger.info("=" * 60)
    logger.info("Starting pretraining...")
    logger.info("=" * 60)

    # Train
    history = trainer.train()

    logger.info("=" * 60)
    logger.info("Pretraining complete!")
    logger.info("=" * 60)
    logger.info(f"Final loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Best loss: {min(history['train_loss']):.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")

    # Save training history
    with open(output_dir / 'pretrain_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Print summary
    logger.info("\nTraining Summary:")
    logger.info(f"  - Model type: {args.model_type}")
    logger.info(f"  - Epochs completed: {len(history['train_loss'])}")
    logger.info(f"  - Initial loss: {history['train_loss'][0]:.4f}")
    logger.info(f"  - Final loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  - Best loss: {min(history['train_loss']):.4f}")


if __name__ == "__main__":
    main()
