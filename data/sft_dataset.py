"""
SFT Dataset for supervised fine-tuning on population prediction task.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning on population growth prediction.

    Features:
    - Loads 64-channel satellite embeddings
    - Pairs with population growth labels
    - Formats data for MLLM conversation format
    """

    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        image_size: int = 980,
        normalize: bool = True,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        Initialize SFT Dataset.

        Args:
            data_dir: Directory containing NPY/TIFF files
            labels_file: Path to Excel/CSV file with labels
            split: Data split ("train", "val", "test")
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            seed: Random seed for splitting
            image_size: Target image size
            normalize: Whether to normalize data
            system_prompt: System prompt for MLLM
            user_prompt: User prompt template
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize = normalize
        self.split = split

        # Default prompts
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.user_prompt = user_prompt or self._default_user_prompt()

        # Load labels
        self.samples = self._load_labels_and_match_data(labels_file)

        # Split dataset
        self.samples = self._split_dataset(
            self.samples, split, train_ratio, val_ratio, seed
        )

        logger.info(f"SFT Dataset ({split}): {len(self.samples)} samples")

    def _default_system_prompt(self) -> str:
        return (
            "你是一个遥感影像分析专家，专门分析城市卫星影像并预测人口统计指标。"
            "请根据提供的卫星影像特征，分析城市发展状况并预测人口自然增长率。"
        )

    def _default_user_prompt(self) -> str:
        return (
            "请分析这张城市卫星影像，根据以下因素预测该城市的人口自然增长率：\n"
            "- 建筑密度和城市化程度\n"
            "- 绿地覆盖率\n"
            "- 道路网络密度\n"
            "- 基础设施发展程度\n\n"
            "请以\"预测值：X.XX‰\"的格式给出你的预测结果。"
        )

    def _load_labels_and_match_data(self, labels_file: str) -> List[Dict]:
        """Load labels and match with data files."""
        labels_path = Path(labels_file)

        # Load labels from Excel or CSV
        if labels_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(labels_path)
        else:
            df = pd.read_csv(labels_path)

        logger.info(f"Loaded {len(df)} label entries from {labels_file}")

        # Find all data files
        data_files = list(self.data_dir.glob("**/*.npy"))
        if len(data_files) == 0:
            data_files = list(self.data_dir.glob("**/*.tif"))
            data_files.extend(list(self.data_dir.glob("**/*.tiff")))

        # Create filename to path mapping
        file_map = {f.stem: f for f in data_files}

        # Match labels with data files
        samples = []
        for _, row in df.iterrows():
            # Try to extract city and year from row
            city = row.get('city', row.get('城市', ''))
            year = row.get('year', row.get('年份', ''))
            label = row.get('growth_rate', row.get('人口自然增长率', row.get('label', None)))

            if label is None:
                continue

            # Try to find matching file
            # Common naming patterns: "城市_年份", "city_year", etc.
            possible_names = [
                f"{city}_{year}",
                f"{city}{year}",
                f"{year}_{city}",
            ]

            matched_file = None
            for name in possible_names:
                if name in file_map:
                    matched_file = file_map[name]
                    break

            if matched_file:
                samples.append({
                    'file_path': matched_file,
                    'city': city,
                    'year': year,
                    'label': float(label),
                })

        logger.info(f"Matched {len(samples)} samples with data files")
        return samples

    def _split_dataset(
        self,
        samples: List[Dict],
        split: str,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> List[Dict]:
        """Split dataset into train/val/test."""
        np.random.seed(seed)

        # Group by city to ensure city-level split
        cities = list(set(s['city'] for s in samples))
        np.random.shuffle(cities)

        n_train = int(len(cities) * train_ratio)
        n_val = int(len(cities) * val_ratio)

        if split == "train":
            target_cities = set(cities[:n_train])
        elif split == "val":
            target_cities = set(cities[n_train:n_train + n_val])
        else:  # test
            target_cities = set(cities[n_train + n_val:])

        return [s for s in samples if s['city'] in target_cities]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample for SFT training.

        Returns:
            Dict containing:
                - 'image': Tensor (64, H, W)
                - 'label': Float value
                - 'conversation': List of messages
                - 'city': City name
                - 'year': Year
        """
        sample = self.samples[idx]

        # Load image data
        file_path = sample['file_path']
        if file_path.suffix == '.npy':
            data = np.load(file_path)
        else:
            import rasterio
            with rasterio.open(file_path) as src:
                data = src.read().astype(np.float32)

        # Crop and normalize
        if data.shape[1] > self.image_size:
            data = data[:, :self.image_size, :self.image_size]

        if self.normalize:
            mean = data.mean()
            std = data.std() + 1e-6
            data = (data - mean) / std

        # Format conversation
        label = sample['label']
        conversation = self._format_conversation(label)

        return {
            'image': torch.FloatTensor(data),
            'label': torch.FloatTensor([label]),
            'conversation': conversation,
            'city': sample['city'],
            'year': sample['year'],
        }

    def _format_conversation(self, label: float) -> List[Dict]:
        """Format the conversation for MLLM training."""
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": f"根据对卫星影像的综合分析，该城市的人口自然增长率预测为：{label:.2f}‰"
            }
        ]


class SFTCollator:
    """Collator for SFT training batches."""

    def __init__(self, processor=None):
        """
        Initialize collator.

        Args:
            processor: MLLM processor for tokenization
        """
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate a batch of samples."""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])

        result = {
            'images': images,
            'labels': labels,
            'conversations': [item['conversation'] for item in batch],
            'cities': [item['city'] for item in batch],
            'years': [item['year'] for item in batch],
        }

        # If processor is available, tokenize conversations
        if self.processor is not None:
            # This will be implemented based on specific MLLM
            pass

        return result


def get_sft_dataloader(
    data_dir: str,
    labels_file: str,
    split: str,
    batch_size: int = 4,
    num_workers: int = 4,
    processor=None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for SFT training.

    Args:
        data_dir: Directory containing data files
        labels_file: Path to labels file
        split: Data split
        batch_size: Batch size
        num_workers: Number of workers
        processor: MLLM processor
        **kwargs: Additional arguments

    Returns:
        DataLoader instance
    """
    dataset = SFTDataset(
        data_dir=data_dir,
        labels_file=labels_file,
        split=split,
        **kwargs
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=SFTCollator(processor),
    )


def parse_prediction_from_text(text: str) -> Optional[float]:
    """
    Parse predicted value from model output text.

    Args:
        text: Model output text

    Returns:
        Extracted numerical value or None if not found
    """
    # Pattern to match numbers with optional sign and decimal
    patterns = [
        r"预测[值为]*[：:]\s*([+-]?\d+\.?\d*)[\s‰%]*",
        r"([+-]?\d+\.?\d*)[\s]*[‰%]",
        r"增长率[为是]*[：:]\s*([+-]?\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None
