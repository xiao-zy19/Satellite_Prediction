#!/usr/bin/env python3
"""
RL (Reinforcement Learning with DPO) entry point.
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
    parser = argparse.ArgumentParser(description="DPO training for 64-channel MLLM")

    # Config
    parser.add_argument("--config", type=str, default="configs/rl_config.yaml")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--preference-file", type=str, required=True)

    # Model
    parser.add_argument("--sft-checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/rl")

    # Training
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)

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
        'beta': args.beta,
        'use_wandb': args.use_wandb,
    })

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import components
    import torch
    from models.qwen_vl_64ch import Qwen2VL64Ch
    from data.dpo_dataset import get_dpo_dataloader
    from training.dpo_trainer import DPOTrainer
    from training.training_utils import set_seed

    set_seed(config.get('seed', 42))

    logger.info("Initializing model...")

    # Create model
    model = Qwen2VL64Ch(
        model_name=args.model_name,
        input_channels=64,
    )

    # Load SFT checkpoint
    if Path(args.sft_checkpoint).exists():
        checkpoint = torch.load(args.sft_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Loaded SFT checkpoint from {args.sft_checkpoint}")
    else:
        logger.error(f"SFT checkpoint not found: {args.sft_checkpoint}")
        return

    logger.info("Creating data loader...")

    # Create data loader
    train_loader = get_dpo_dataloader(
        data_dir=args.data_dir,
        preference_file=args.preference_file,
        batch_size=args.batch_size,
    )

    logger.info(f"DPO data: {len(train_loader.dataset)} preference pairs")

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        train_loader=train_loader,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        save_dir=str(output_dir),
        use_wandb=args.use_wandb,
    )

    logger.info("Starting DPO training...")

    # Train
    history = trainer.train()

    logger.info("DPO training complete!")
    logger.info(f"Best reward margin: {trainer.best_reward:.4f}")

    # Save history
    import json
    with open(output_dir / 'dpo_history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
