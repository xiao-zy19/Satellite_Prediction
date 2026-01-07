"""
DPO Dataset for Direct Preference Optimization training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class DPODataset(Dataset):
    """
    Dataset for DPO (Direct Preference Optimization) training.

    Features:
    - Loads preference pairs (chosen vs rejected)
    - Supports pre-generated pairs or on-the-fly generation
    """

    def __init__(
        self,
        data_dir: str,
        preference_file: Optional[str] = None,
        image_size: int = 980,
        normalize: bool = True,
    ):
        """
        Initialize DPO Dataset.

        Args:
            data_dir: Directory containing image files
            preference_file: Path to JSON file with preference pairs
            image_size: Target image size
            normalize: Whether to normalize data
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize = normalize

        # Load preference pairs
        if preference_file and Path(preference_file).exists():
            with open(preference_file, 'r') as f:
                self.pairs = json.load(f)
            logger.info(f"Loaded {len(self.pairs)} preference pairs from {preference_file}")
        else:
            self.pairs = []
            logger.warning("No preference file provided. Use generate_pairs() first.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a preference pair for DPO training.

        Returns:
            Dict containing:
                - 'image': Tensor (64, H, W)
                - 'prompt': User prompt string
                - 'chosen': Chosen response string
                - 'rejected': Rejected response string
                - 'true_value': Ground truth value
        """
        pair = self.pairs[idx]

        # Load image
        image_path = Path(pair['image_path'])
        if image_path.suffix == '.npy':
            data = np.load(image_path)
        else:
            import rasterio
            with rasterio.open(image_path) as src:
                data = src.read().astype(np.float32)

        # Crop and normalize
        if data.shape[1] > self.image_size:
            data = data[:, :self.image_size, :self.image_size]

        if self.normalize:
            mean = data.mean()
            std = data.std() + 1e-6
            data = (data - mean) / std

        return {
            'image': torch.FloatTensor(data),
            'prompt': pair['prompt'],
            'chosen': pair['chosen'],
            'rejected': pair['rejected'],
            'true_value': torch.FloatTensor([pair['true_value']]),
        }

    @staticmethod
    def generate_pairs_from_model(
        model,
        dataloader,
        output_file: str,
        num_samples: int = 5,
        temperature: float = 0.7,
        device: str = "cuda",
    ) -> None:
        """
        Generate preference pairs by sampling from a trained model.

        For each sample:
        1. Generate multiple outputs
        2. Rank by distance to true value
        3. Best becomes "chosen", worst becomes "rejected"

        Args:
            model: Trained MLLM model
            dataloader: DataLoader with samples
            output_file: Path to save preference pairs
            num_samples: Number of samples to generate per image
            temperature: Sampling temperature
            device: Device to use
        """
        from .sft_dataset import parse_prediction_from_text

        model.eval()
        pairs = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(device)
                true_values = batch['labels']
                prompts = batch['conversations']

                for i in range(len(images)):
                    image = images[i:i+1]
                    true_value = true_values[i].item()
                    prompt = prompts[i][1]['content'][1]['text']  # User prompt

                    # Generate multiple outputs
                    outputs = []
                    predictions = []

                    for _ in range(num_samples):
                        # Generate response (implementation depends on model)
                        output = model.generate(
                            image,
                            prompt,
                            temperature=temperature,
                            max_new_tokens=256,
                        )
                        outputs.append(output)

                        # Parse prediction
                        pred = parse_prediction_from_text(output)
                        predictions.append(pred if pred is not None else float('inf'))

                    # Calculate errors
                    errors = [abs(p - true_value) if p != float('inf') else float('inf')
                              for p in predictions]

                    # Find best and worst
                    best_idx = np.argmin(errors)
                    worst_idx = np.argmax(errors)

                    if errors[best_idx] < errors[worst_idx]:
                        pairs.append({
                            'image_path': str(batch.get('file_paths', [''])[i]),
                            'prompt': prompt,
                            'chosen': outputs[best_idx],
                            'rejected': outputs[worst_idx],
                            'true_value': true_value,
                            'chosen_error': errors[best_idx],
                            'rejected_error': errors[worst_idx],
                        })

        # Save pairs
        with open(output_file, 'w') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Generated {len(pairs)} preference pairs, saved to {output_file}")


class DPOCollator:
    """Collator for DPO training batches."""

    def __init__(self, processor=None, tokenizer=None):
        """
        Initialize collator.

        Args:
            processor: MLLM processor
            tokenizer: Tokenizer for text
        """
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate a batch of preference pairs."""
        images = torch.stack([item['image'] for item in batch])
        true_values = torch.stack([item['true_value'] for item in batch])

        result = {
            'images': images,
            'prompts': [item['prompt'] for item in batch],
            'chosen': [item['chosen'] for item in batch],
            'rejected': [item['rejected'] for item in batch],
            'true_values': true_values,
        }

        # Tokenize if tokenizer is available
        if self.tokenizer is not None:
            result['chosen_ids'] = self.tokenizer(
                result['chosen'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            result['rejected_ids'] = self.tokenizer(
                result['rejected'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

        return result


def get_dpo_dataloader(
    data_dir: str,
    preference_file: str,
    batch_size: int = 2,
    num_workers: int = 4,
    processor=None,
    tokenizer=None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for DPO training.

    Args:
        data_dir: Directory containing image files
        preference_file: Path to preference pairs JSON
        batch_size: Batch size
        num_workers: Number of workers
        processor: MLLM processor
        tokenizer: Tokenizer
        **kwargs: Additional arguments

    Returns:
        DataLoader instance
    """
    dataset = DPODataset(
        data_dir=data_dir,
        preference_file=preference_file,
        **kwargs
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DPOCollator(processor, tokenizer),
    )
