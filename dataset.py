"""
Dataset classes for population growth prediction and self-supervised pretraining
"""

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

import config


class CityDataset(Dataset):
    """
    Dataset for city satellite data with population growth rate labels.

    Supports both supervised learning and self-supervised pretraining.
    """

    def __init__(
        self,
        samples: List[Dict],
        patch_size: int = config.PATCH_SIZE_PIXELS,
        num_patches_per_dim: int = config.NUM_PATCHES_PER_DIM,
        augment: bool = False,
        return_raw: bool = False,
        contrastive: bool = False
    ):
        """
        Args:
            samples: List of dicts with keys: 'city', 'year', 'tiff_path', 'growth_rate'
            patch_size: Size of each patch in pixels
            num_patches_per_dim: Number of patches per dimension
            augment: Whether to apply data augmentation
            return_raw: Return raw data without patch extraction (for MAE)
            contrastive: Return two augmented views (for SimCLR)
        """
        self.samples = samples
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        self.augment = augment
        self.return_raw = return_raw
        self.contrastive = contrastive

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load TIFF data
        with rasterio.open(sample['tiff_path']) as src:
            data = src.read().astype(np.float32)  # Shape: (64, H, W)

        # Handle nodata (-128 for Alpha Earth)
        data[data == -128] = 0
        data = data / 127.0  # Normalize to [-1, 1]

        # Pad to full size if needed
        data = self._pad_to_full_size(data)

        if self.return_raw:
            # Return full tile for MAE
            patches = torch.FloatTensor(data)
            patches = self._normalize_tensor(patches)
        else:
            # Extract patches
            patches = self._extract_patches(data)

            if self.contrastive:
                # Return two augmented views for SimCLR
                view1 = self._augment_patches(patches.copy())
                view2 = self._augment_patches(patches.copy())
                view1 = torch.FloatTensor(view1)
                view2 = torch.FloatTensor(view2)
                view1 = self._normalize_tensor(view1)
                view2 = self._normalize_tensor(view2)

                return view1, view2, {
                    'city': sample['city'],
                    'year': sample['year'],
                    'idx': idx
                }

            if self.augment:
                patches = self._augment_patches(patches)

            patches = torch.FloatTensor(patches)
            patches = self._normalize_tensor(patches)

        # Label
        label = torch.FloatTensor([sample['growth_rate']])

        info = {
            'city': sample['city'],
            'year': sample['year'],
            'idx': idx
        }

        return patches, label, info

    def _pad_to_full_size(self, data: np.ndarray, target_size: int = config.FULL_SIZE) -> np.ndarray:
        """Pad data to target size if smaller."""
        n_bands, h, w = data.shape

        if h >= target_size and w >= target_size:
            # Crop to target size
            return data[:, :target_size, :target_size]

        # Create padded array
        padded = np.zeros((n_bands, target_size, target_size), dtype=data.dtype)
        h_start = (target_size - h) // 2
        w_start = (target_size - w) // 2
        padded[:, h_start:h_start+h, w_start:w_start+w] = data

        return padded

    def _extract_patches(self, data: np.ndarray) -> np.ndarray:
        """Extract non-overlapping patches from the full tile."""
        n_bands, h, w = data.shape
        ps = self.patch_size
        n = self.num_patches_per_dim

        patches = []
        for i in range(n):
            for j in range(n):
                y_start = i * ps
                x_start = j * ps
                patch = data[:, y_start:y_start+ps, x_start:x_start+ps]
                patches.append(patch)

        return np.array(patches)  # (25, 64, 200, 200)

    def _normalize_tensor(self, patches: torch.Tensor) -> torch.Tensor:
        """Normalize patches per channel."""
        if patches.dim() == 4:
            # (num_patches, 64, H, W)
            mean = patches.mean(dim=(2, 3), keepdim=True)
            std = patches.std(dim=(2, 3), keepdim=True) + 1e-8
        else:
            # (64, H, W)
            mean = patches.mean(dim=(1, 2), keepdim=True)
            std = patches.std(dim=(1, 2), keepdim=True) + 1e-8
        return (patches - mean) / std

    def _augment_patches(self, patches: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            patches = patches[:, :, :, ::-1].copy()

        # Random vertical flip
        if random.random() > 0.5:
            patches = patches[:, :, ::-1, :].copy()

        # Random 90-degree rotation
        if random.random() > 0.5:
            k = random.randint(1, 3)
            patches = np.rot90(patches, k, axes=(2, 3)).copy()

        # Random Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.05, patches.shape).astype(np.float32)
            patches = patches + noise

        return patches


class PretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining without labels.
    Uses all available satellite data.
    """

    def __init__(
        self,
        satellite_dir: Path = config.SATELLITE_DIR,
        patch_size: int = config.PATCH_SIZE_PIXELS,
        num_patches_per_dim: int = config.NUM_PATCHES_PER_DIM,
        contrastive: bool = True
    ):
        self.satellite_dir = Path(satellite_dir)
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        self.contrastive = contrastive

        # Find all TIFF files
        self.tiff_files = []
        for province_dir in self.satellite_dir.iterdir():
            if not province_dir.is_dir():
                continue
            for city_dir in province_dir.iterdir():
                if not city_dir.is_dir():
                    continue
                for tiff_file in city_dir.glob('*.tiff'):
                    self.tiff_files.append(tiff_file)

        print(f"Found {len(self.tiff_files)} TIFF files for pretraining")

    def __len__(self) -> int:
        return len(self.tiff_files)

    def __getitem__(self, idx: int):
        tiff_path = self.tiff_files[idx]

        # Load TIFF data
        with rasterio.open(tiff_path) as src:
            data = src.read().astype(np.float32)

        # Handle nodata
        data[data == -128] = 0
        data = data / 127.0

        # Pad to full size
        data = self._pad_to_full_size(data)

        # Extract patches
        patches = self._extract_patches(data)

        if self.contrastive:
            view1 = self._augment_patches(patches.copy())
            view2 = self._augment_patches(patches.copy())
            view1 = torch.FloatTensor(view1)
            view2 = torch.FloatTensor(view2)
            view1 = self._normalize_tensor(view1)
            view2 = self._normalize_tensor(view2)
            return view1, view2
        else:
            patches = torch.FloatTensor(patches)
            patches = self._normalize_tensor(patches)
            return patches

    def _pad_to_full_size(self, data: np.ndarray, target_size: int = config.FULL_SIZE) -> np.ndarray:
        n_bands, h, w = data.shape
        if h >= target_size and w >= target_size:
            return data[:, :target_size, :target_size]
        padded = np.zeros((n_bands, target_size, target_size), dtype=data.dtype)
        h_start = (target_size - h) // 2
        w_start = (target_size - w) // 2
        padded[:, h_start:h_start+h, w_start:w_start+w] = data
        return padded

    def _extract_patches(self, data: np.ndarray) -> np.ndarray:
        n_bands, h, w = data.shape
        ps = self.patch_size
        n = self.num_patches_per_dim
        patches = []
        for i in range(n):
            for j in range(n):
                y_start = i * ps
                x_start = j * ps
                patch = data[:, y_start:y_start+ps, x_start:x_start+ps]
                patches.append(patch)
        return np.array(patches)

    def _normalize_tensor(self, patches: torch.Tensor) -> torch.Tensor:
        mean = patches.mean(dim=(2, 3), keepdim=True)
        std = patches.std(dim=(2, 3), keepdim=True) + 1e-8
        return (patches - mean) / std

    def _augment_patches(self, patches: np.ndarray) -> np.ndarray:
        # Random horizontal flip
        if random.random() > 0.5:
            patches = patches[:, :, :, ::-1].copy()
        # Random vertical flip
        if random.random() > 0.5:
            patches = patches[:, :, ::-1, :].copy()
        # Random rotation
        if random.random() > 0.5:
            k = random.randint(1, 3)
            patches = np.rot90(patches, k, axes=(2, 3)).copy()
        # Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.05, patches.shape).astype(np.float32)
            patches = patches + noise
        return patches


def load_population_data(excel_path: Path = config.POPULATION_DATA) -> pd.DataFrame:
    """Load population growth rate data from Excel file."""
    df = pd.read_excel(excel_path)
    return df


def find_satellite_data(satellite_dir: Path = config.SATELLITE_DIR) -> Dict[str, Dict[int, Path]]:
    """Find all available satellite TIFF files."""
    satellite_data = {}

    for province_dir in satellite_dir.iterdir():
        if not province_dir.is_dir() or province_dir.name.endswith('.json'):
            continue

        for city_dir in province_dir.iterdir():
            if not city_dir.is_dir():
                continue

            city_name = city_dir.name
            city_data = {}

            for tiff_file in city_dir.glob('*.tiff'):
                try:
                    year = int(tiff_file.stem.split('_')[0])
                    city_data[year] = tiff_file
                except (ValueError, IndexError):
                    continue

            if city_data:
                satellite_data[city_name] = city_data

    return satellite_data


def create_dataset_samples(
    population_df: pd.DataFrame,
    satellite_data: Dict[str, Dict[int, Path]]
) -> List[Dict]:
    """Create list of valid samples with both satellite and population data."""
    samples = []

    pop_cities = set(population_df['城市名称'].values)
    sat_cities = set(satellite_data.keys())
    common_cities = pop_cities & sat_cities

    for city in common_cities:
        city_row = population_df[population_df['城市名称'] == city].iloc[0]

        for year, tiff_path in satellite_data[city].items():
            year_col = str(year)
            if year_col in population_df.columns:
                growth_rate = city_row.get(year_col)
                if pd.notna(growth_rate):
                    samples.append({
                        'city': city,
                        'year': year,
                        'tiff_path': str(tiff_path),
                        'growth_rate': float(growth_rate)
                    })

    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.RANDOM_SEED,
    stratify_by_city: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/val/test sets."""
    random.seed(seed)
    np.random.seed(seed)

    if stratify_by_city:
        city_samples = {}
        for s in samples:
            city = s['city']
            if city not in city_samples:
                city_samples[city] = []
            city_samples[city].append(s)

        cities = list(city_samples.keys())
        random.shuffle(cities)

        n_cities = len(cities)
        n_train = int(n_cities * train_ratio)
        n_val = int(n_cities * val_ratio)

        train_cities = set(cities[:n_train])
        val_cities = set(cities[n_train:n_train + n_val])
        test_cities = set(cities[n_train + n_val:])

        train_samples = [s for s in samples if s['city'] in train_cities]
        val_samples = [s for s in samples if s['city'] in val_cities]
        test_samples = [s for s in samples if s['city'] in test_cities]
    else:
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        n = len(samples_copy)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_samples = samples_copy[:n_train]
        val_samples = samples_copy[n_train:n_train + n_val]
        test_samples = samples_copy[n_train + n_val:]

    return train_samples, val_samples, test_samples


def get_dataloaders(
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = True,
    contrastive: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create train, validation, and test dataloaders."""
    print("Loading population data...")
    pop_df = load_population_data()

    print("Finding satellite data...")
    sat_data = find_satellite_data()

    print("Creating samples...")
    samples = create_dataset_samples(pop_df, sat_data)
    print(f"Total valid samples: {len(samples)}")

    print("Splitting dataset...")
    train_samples, val_samples, test_samples = split_dataset(samples)

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")

    train_dataset = CityDataset(
        train_samples, augment=augment_train, contrastive=contrastive
    )
    val_dataset = CityDataset(val_samples, augment=False, contrastive=False)
    test_dataset = CityDataset(test_samples, augment=False, contrastive=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataset_info = {
        'num_train': len(train_samples),
        'num_val': len(val_samples),
        'num_test': len(test_samples),
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples
    }

    return train_loader, val_loader, test_loader, dataset_info


def get_pretrain_dataloader(
    batch_size: int = 16,
    num_workers: int = 4,
    contrastive: bool = True
) -> DataLoader:
    """Create dataloader for self-supervised pretraining."""
    dataset = PretrainDataset(contrastive=contrastive)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    train_loader, val_loader, test_loader, info = get_dataloaders(batch_size=2, num_workers=0)

    for patches, labels, meta in train_loader:
        print(f"Patches shape: {patches.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Cities: {meta['city']}")
        break

    print("\nTesting pretrain dataset...")
    pretrain_loader = get_pretrain_dataloader(batch_size=2, num_workers=0)

    for view1, view2 in pretrain_loader:
        print(f"View1 shape: {view1.shape}")
        print(f"View2 shape: {view2.shape}")
        break
