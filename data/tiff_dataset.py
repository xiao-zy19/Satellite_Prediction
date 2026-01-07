"""
TIFF Dataset loader for 64-channel satellite embeddings.
Provides efficient loading with optional NPY caching.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import logging

try:
    import rasterio
except ImportError:
    rasterio = None
    print("Warning: rasterio not installed. TIFF loading will not work.")

logger = logging.getLogger(__name__)


class TIFFDataset(Dataset):
    """
    Dataset for loading 64-channel satellite embeddings from TIFF files.

    Features:
    - Lazy loading with optional NPY caching for faster subsequent access
    - Automatic normalization
    - Support for both TIFF and pre-cached NPY files
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        normalize: bool = True,
        target_size: int = 980,
        file_pattern: str = "**/*.tif",
    ):
        """
        Initialize TIFF Dataset.

        Args:
            data_dir: Directory containing TIFF files
            cache_dir: Directory to cache NPY files (optional)
            use_cache: Whether to use NPY cache
            normalize: Whether to normalize data
            target_size: Target image size (aligned to patch_size)
            file_pattern: Glob pattern for finding TIFF files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.normalize = normalize
        self.target_size = target_size

        # Create cache directory if needed
        if self.cache_dir and self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all TIFF files
        self.tiff_files = sorted(list(self.data_dir.glob(file_pattern)))
        logger.info(f"Found {len(self.tiff_files)} TIFF files in {data_dir}")

        if len(self.tiff_files) == 0:
            # Try .tiff extension
            self.tiff_files = sorted(list(self.data_dir.glob("**/*.tiff")))
            logger.info(f"Found {len(self.tiff_files)} .tiff files")

        # Pre-compute cache paths
        self.cache_paths = []
        if self.cache_dir:
            for tiff_path in self.tiff_files:
                cache_name = tiff_path.stem + ".npy"
                self.cache_paths.append(self.cache_dir / cache_name)
        else:
            self.cache_paths = [None] * len(self.tiff_files)

    def __len__(self) -> int:
        return len(self.tiff_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return data for given index.

        Returns:
            torch.Tensor: Shape (64, target_size, target_size)
        """
        tiff_path = self.tiff_files[idx]
        cache_path = self.cache_paths[idx]

        # Try loading from cache first
        if self.use_cache and cache_path and cache_path.exists():
            data = np.load(cache_path)
        else:
            # Load from TIFF
            data = self._load_tiff(tiff_path)

            # Save to cache if enabled
            if self.use_cache and cache_path:
                np.save(cache_path, data)

        # Crop to target size
        if data.shape[1] > self.target_size:
            data = data[:, :self.target_size, :self.target_size]

        # Normalize
        if self.normalize:
            data = self._normalize(data)

        return torch.FloatTensor(data)

    def _load_tiff(self, path: Path) -> np.ndarray:
        """Load data from TIFF file."""
        if rasterio is None:
            raise ImportError("rasterio is required for TIFF loading")

        with rasterio.open(path) as src:
            data = src.read()  # (64, H, W)

        return data.astype(np.float32)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance."""
        mean = data.mean()
        std = data.std() + 1e-6
        return (data - mean) / std

    def get_file_path(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.tiff_files[idx]

    @staticmethod
    def preprocess_all_tiffs(
        input_dir: str,
        output_dir: str,
        num_workers: int = 16,
        normalize: bool = True,
    ) -> None:
        """
        Batch preprocess all TIFF files to NPY format.

        This is recommended for faster training as NPY loading is much faster.

        Args:
            input_dir: Directory containing TIFF files
            output_dir: Directory to save NPY files
            num_workers: Number of parallel workers
            normalize: Whether to normalize during preprocessing
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all TIFF files
        tiff_files = list(input_dir.glob("**/*.tif"))
        tiff_files.extend(list(input_dir.glob("**/*.tiff")))
        logger.info(f"Found {len(tiff_files)} TIFF files to preprocess")

        def convert_one(tiff_path: Path) -> Tuple[str, bool]:
            try:
                output_path = output_dir / (tiff_path.stem + ".npy")

                if output_path.exists():
                    return str(tiff_path), True

                with rasterio.open(tiff_path) as src:
                    data = src.read().astype(np.float32)

                if normalize:
                    mean = data.mean()
                    std = data.std() + 1e-6
                    data = (data - mean) / std

                np.save(output_path, data)
                return str(tiff_path), True
            except Exception as e:
                logger.error(f"Error processing {tiff_path}: {e}")
                return str(tiff_path), False

        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(convert_one, tiff_files),
                total=len(tiff_files),
                desc="Preprocessing TIFF files"
            ))

        success_count = sum(1 for _, success in results if success)
        logger.info(f"Successfully preprocessed {success_count}/{len(tiff_files)} files")


class NPYDataset(Dataset):
    """
    Dataset for loading pre-cached NPY files.
    Faster than TIFFDataset when data is already preprocessed.
    """

    def __init__(
        self,
        data_dir: str,
        normalize: bool = False,  # Usually already normalized
        target_size: int = 980,
    ):
        """
        Initialize NPY Dataset.

        Args:
            data_dir: Directory containing NPY files
            normalize: Whether to normalize data (usually False if preprocessed)
            target_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.target_size = target_size

        # Find all NPY files
        self.npy_files = sorted(list(self.data_dir.glob("**/*.npy")))
        logger.info(f"Found {len(self.npy_files)} NPY files in {data_dir}")

    def __len__(self) -> int:
        return len(self.npy_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return data for given index."""
        data = np.load(self.npy_files[idx])

        # Crop to target size
        if data.shape[1] > self.target_size:
            data = data[:, :self.target_size, :self.target_size]

        # Normalize if needed
        if self.normalize:
            mean = data.mean()
            std = data.std() + 1e-6
            data = (data - mean) / std

        return torch.FloatTensor(data)

    def get_file_path(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.npy_files[idx]


def get_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 8,
    use_npy: bool = True,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for satellite embeddings.

    Args:
        data_dir: Directory containing data files
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_npy: Whether to use NPY dataset (faster)
        cache_dir: Cache directory for TIFF dataset
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    if use_npy:
        dataset = NPYDataset(data_dir, **kwargs)
    else:
        dataset = TIFFDataset(data_dir, cache_dir=cache_dir, **kwargs)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
