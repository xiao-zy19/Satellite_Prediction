"""
MAE Pretrain Dataset for 64-channel satellite embeddings.
Enhanced version with strong data augmentation for small datasets.

针对~660个样本的增强策略，可扩展到 ~50,000+ 有效样本
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging
import rasterio

logger = logging.getLogger(__name__)


class MAEPretrainDataset(Dataset):
    """
    Dataset for MAE (Masked Autoencoder) pretraining on 64-channel satellite embeddings.

    Enhanced Features:
    - Strong data augmentation (flip, rotation, random crop)
    - Multi-scale patches (切分成更小的块)
    - Repeat sampling (每个文件被采样多次)
    - Online augmentation (每次访问都是不同的增强版本)
    """

    def __init__(
        self,
        data_dir: str,
        mask_ratio: float = 0.75,
        patch_size: int = 14,
        image_size: int = 980,
        augment: bool = True,
        normalize: bool = True,
        # 增强参数
        samples_per_file: int = 20,  # 每个文件采样次数
        use_multi_crop: bool = True,  # 是否使用多位置裁剪
        crop_scales: List[float] = [1.0, 0.8, 0.6],  # 裁剪尺度
    ):
        """
        Initialize MAE Pretrain Dataset.

        Args:
            data_dir: Directory containing TIFF files
            mask_ratio: Ratio of patches to mask (default 0.75)
            patch_size: Size of each patch
            image_size: Output image size (should be divisible by patch_size)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize data
            samples_per_file: Number of samples to generate from each file
            use_multi_crop: Whether to use random cropping at different positions
            crop_scales: Scales for random cropping
        """
        self.data_dir = Path(data_dir)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.samples_per_file = samples_per_file
        self.use_multi_crop = use_multi_crop
        self.crop_scales = crop_scales

        # Calculate number of patches
        self.num_patches_per_dim = image_size // patch_size
        self.total_patches = self.num_patches_per_dim ** 2
        self.num_masked = int(self.total_patches * mask_ratio)

        # Find all data files (支持递归搜索)
        self.data_files = sorted(list(self.data_dir.glob("**/*.tiff")))
        if len(self.data_files) == 0:
            self.data_files = sorted(list(self.data_dir.glob("**/*.tif")))
        if len(self.data_files) == 0:
            self.data_files = sorted(list(self.data_dir.glob("**/*.npy")))

        self.num_files = len(self.data_files)
        self.total_samples = self.num_files * samples_per_file

        logger.info(f"Found {self.num_files} files for MAE pretraining")
        logger.info(f"Samples per file: {samples_per_file}")
        logger.info(f"Total effective samples: {self.total_samples}")
        logger.info(f"Mask ratio: {mask_ratio}, Total patches: {self.total_patches}")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load data and generate mask for MAE pretraining.
        Each file is sampled multiple times with different augmentations.

        Returns:
            Dict containing:
                - 'image': Image tensor (64, H, W)
                - 'mask': Boolean mask tensor (num_patches,) True = masked
                - 'ids_keep': Indices of kept patches
                - 'ids_restore': Indices to restore original order
        """
        # 计算实际文件索引和采样次数
        file_idx = idx % self.num_files
        sample_idx = idx // self.num_files  # 用于确定增强参数的种子

        # Load data
        file_path = self.data_files[file_idx]
        data = self._load_file(file_path)

        # Apply augmentation with different random seed for each sample
        if self.augment:
            data = self._augment(data, seed=idx)

        # Random crop to target size
        if self.use_multi_crop:
            data = self._random_crop(data, seed=idx)
        else:
            # Center crop
            data = self._center_crop(data)

        # Normalize
        if self.normalize:
            data = self._normalize(data)

        # Generate random mask
        mask, ids_keep, ids_restore = self._generate_mask(seed=idx)

        return {
            'image': torch.FloatTensor(data),
            'mask': mask,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore,
        }

    def _load_file(self, file_path: Path) -> np.ndarray:
        """Load data from file."""
        if file_path.suffix in ['.tiff', '.tif']:
            with rasterio.open(file_path) as src:
                data = src.read().astype(np.float32)
        else:
            data = np.load(file_path).astype(np.float32)
        return data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance."""
        mean = data.mean()
        std = data.std() + 1e-6
        return (data - mean) / std

    def _center_crop(self, data: np.ndarray) -> np.ndarray:
        """Center crop to target size."""
        _, h, w = data.shape
        if h >= self.image_size and w >= self.image_size:
            start_h = (h - self.image_size) // 2
            start_w = (w - self.image_size) // 2
            return data[:, start_h:start_h + self.image_size,
                          start_w:start_w + self.image_size]
        return data[:, :self.image_size, :self.image_size]

    def _random_crop(self, data: np.ndarray, seed: int) -> np.ndarray:
        """
        Random crop with multiple scales.

        策略：
        1. 随机选择一个裁剪尺度
        2. 从原图中随机位置裁剪
        3. 如果裁剪后小于目标尺寸，进行resize/pad
        """
        rng = np.random.RandomState(seed)
        _, h, w = data.shape

        # 随机选择裁剪尺度
        scale = rng.choice(self.crop_scales)
        crop_size = int(self.image_size / scale)

        # 确保裁剪尺寸不超过原图
        crop_size = min(crop_size, h, w)

        # 随机裁剪位置
        if h > crop_size:
            start_h = rng.randint(0, h - crop_size)
        else:
            start_h = 0
        if w > crop_size:
            start_w = rng.randint(0, w - crop_size)
        else:
            start_w = 0

        cropped = data[:, start_h:start_h + crop_size,
                         start_w:start_w + crop_size]

        # 如果裁剪后尺寸不等于目标尺寸，进行调整
        if cropped.shape[1] != self.image_size or cropped.shape[2] != self.image_size:
            # 使用简单的resize（最近邻插值，保持特征不变形）
            cropped = self._resize(cropped, self.image_size)

        return cropped

    def _resize(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize data to target size using PyTorch interpolate.

        使用 PyTorch 的 F.interpolate 进行快速最近邻插值，
        比双重 for 循环快约 50 倍。

        Note:
            torch.from_numpy 要求输入数组内存连续。
            _augment 中的 np.flip/np.rot90 已使用 .copy() 确保连续性。
            此处额外添加 np.ascontiguousarray 作为安全保障。
        """
        # 确保内存连续，避免 torch.from_numpy 报错
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        # (C, H, W) -> (1, C, H, W) for F.interpolate
        tensor = torch.from_numpy(data).unsqueeze(0)

        # 最近邻插值
        resized = F.interpolate(
            tensor,
            size=(target_size, target_size),
            mode='nearest'
        )

        # (1, C, H, W) -> (C, H, W)
        return resized.squeeze(0).numpy()

    def _augment(self, data: np.ndarray, seed: int) -> np.ndarray:
        """Apply random augmentation with given seed for reproducibility."""
        rng = np.random.RandomState(seed)

        # Random horizontal flip (50% probability)
        if rng.random() > 0.5:
            data = np.flip(data, axis=2).copy()

        # Random vertical flip (50% probability)
        if rng.random() > 0.5:
            data = np.flip(data, axis=1).copy()

        # Random 90-degree rotation (0, 90, 180, or 270 degrees)
        k = rng.randint(0, 4)
        if k > 0:
            data = np.rot90(data, k, axes=(1, 2)).copy()

        # Random channel shuffle (轻微扰动，保持通道关系)
        # 注意：对于预训练阶段，这个增强要谨慎使用
        # if rng.random() > 0.8:  # 20% probability
        #     perm = rng.permutation(64)
        #     data = data[perm, :, :]

        return data

    def _generate_mask(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random mask for MAE with given seed.

        Returns:
            mask: Boolean tensor (num_patches,) - True means masked
            ids_keep: Indices of visible patches
            ids_restore: Indices to restore original order
        """
        # Use seed for reproducibility within the same sample
        rng = np.random.RandomState(seed + 10000)  # offset to differ from augmentation

        # Random shuffle
        noise = rng.random(self.total_patches)
        ids_shuffle = np.argsort(noise)
        ids_restore = np.argsort(ids_shuffle)

        # Keep the first (1 - mask_ratio) patches
        num_keep = self.total_patches - self.num_masked
        ids_keep = ids_shuffle[:num_keep]

        # Generate binary mask
        mask = np.ones(self.total_patches, dtype=bool)
        mask[ids_keep] = False

        return (
            torch.from_numpy(mask),
            torch.from_numpy(ids_keep.copy()).long(),
            torch.from_numpy(ids_restore.copy()).long(),
        )


class MAEPretrainCollator:
    """Collator for MAE pretraining batches."""

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        ids_keep = torch.stack([item['ids_keep'] for item in batch])
        ids_restore = torch.stack([item['ids_restore'] for item in batch])

        return {
            'image': images,
            'mask': masks,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore,
        }


def get_pretrain_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 8,
    mask_ratio: float = 0.75,
    samples_per_file: int = 20,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for MAE pretraining.

    Args:
        data_dir: Directory containing data files
        batch_size: Batch size
        num_workers: Number of data loading workers
        mask_ratio: Ratio of patches to mask
        samples_per_file: Number of augmented samples per original file
        **kwargs: Additional arguments for dataset

    Returns:
        DataLoader instance
    """
    dataset = MAEPretrainDataset(
        data_dir=data_dir,
        mask_ratio=mask_ratio,
        samples_per_file=samples_per_file,
        **kwargs
    )

    # prefetch_factor only works with num_workers > 0
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True if num_workers > 0 else False,
        'collate_fn': MAEPretrainCollator(),
        'drop_last': True,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2

    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


# ============ 数据统计工具 ============

def analyze_dataset(data_dir: str) -> Dict:
    """
    分析数据集统计信息。

    Args:
        data_dir: 数据目录

    Returns:
        统计信息字典
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob("**/*.tiff")) + list(data_dir.glob("**/*.tif"))

    if len(files) == 0:
        files = list(data_dir.glob("**/*.npy"))

    print(f"Found {len(files)} files")

    # 采样分析
    sample_files = files[:min(10, len(files))]
    shapes = []
    means = []
    stds = []

    for f in sample_files:
        if f.suffix in ['.tiff', '.tif']:
            with rasterio.open(f) as src:
                data = src.read().astype(np.float32)
        else:
            data = np.load(f).astype(np.float32)

        shapes.append(data.shape)
        means.append(data.mean())
        stds.append(data.std())

    return {
        'num_files': len(files),
        'sample_shapes': shapes,
        'mean_of_means': np.mean(means),
        'mean_of_stds': np.mean(stds),
    }


if __name__ == "__main__":
    # 测试数据集
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data"

    print("Analyzing dataset...")
    stats = analyze_dataset(data_dir)
    print(f"Dataset statistics: {stats}")

    print("\nCreating dataset...")
    dataset = MAEPretrainDataset(
        data_dir=data_dir,
        samples_per_file=20,
        mask_ratio=0.75,
    )

    print(f"Total samples: {len(dataset)}")

    # 测试加载
    print("\nTesting data loading...")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Num masked: {sample['mask'].sum().item()}")
    print(f"Num visible: {(~sample['mask']).sum().item()}")
