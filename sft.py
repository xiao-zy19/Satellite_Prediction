#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) entry point.
"""

import argparse
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT for 64-channel MLLM")

    # Config
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--labels-file", type=str, required=True)

    # Model
    parser.add_argument("--pretrain-checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")

    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)

    # Logging
    parser.add_argument("--use-wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with args
    config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'freeze_encoder_epochs': args.freeze_encoder_epochs,
        'use_wandb': args.use_wandb,
    })

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import components
    from models.qwen_vl_64ch import Qwen2VL64Ch
    from data.sft_dataset import get_sft_dataloader
    from training.sft_trainer import SFTTrainer
    from training.training_utils import set_seed

    set_seed(config.get('seed', 42))

    logger.info("Initializing model...")

    # Create model
    model = Qwen2VL64Ch(
        model_name=args.model_name,
        input_channels=64,
    )

    # Load pretrain checkpoint if provided
    if args.pretrain_checkpoint and Path(args.pretrain_checkpoint).exists():
        import torch
        checkpoint = torch.load(args.pretrain_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Loaded pretrain checkpoint from {args.pretrain_checkpoint}")

    logger.info("Creating data loaders...")

    # Create data loaders
    train_loader = get_sft_dataloader(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        split="train",
        batch_size=args.batch_size,
    )

    val_loader = get_sft_dataloader(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        split="val",
        batch_size=args.batch_size,
    )

    test_loader = get_sft_dataloader(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        split="test",
        batch_size=args.batch_size,
    )

    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        save_dir=str(output_dir),
        use_wandb=args.use_wandb,
    )

    logger.info("Starting SFT training...")

    # Train
    history = trainer.train()

    logger.info("SFT training complete!")
    logger.info(f"Best R²: {trainer.best_r2:.4f}")

    # Save history
    import json
    with open(output_dir / 'sft_history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
