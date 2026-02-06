"""
Dataset classes with BERT policy features for multimodal population growth prediction.

Based on dataset_policy.py, extended to support:
1. BERT embeddings from pre-computed cache
2. Hybrid mode (BERT + structured features)
3. Original structured features (for compatibility)

Policy source options:
- "structured": Original 12-dim structured features
- "bert": BERT embeddings from cache (default 64-dim)
- "hybrid": Concatenation of BERT + structured features
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
from policy_features import get_policy_extractor, get_policy_features
from policy_bert import PolicyEmbeddingCache


class CityPolicyBertDataset(Dataset):
    """
    Dataset for city satellite data with BERT/hybrid policy features.

    Supports three policy sources:
    - "structured": 12-dim structured features (original)
    - "bert": BERT embeddings from cache
    - "hybrid": BERT + structured concatenated

    Returns:
        - patches: (25, 64, 200, 200) satellite image patches
        - policy_features: (policy_dim,) policy features
        - label: (1,) population growth rate
        - info: dict with city/year metadata
    """

    def __init__(
        self,
        samples: List[Dict],
        policy_source: str = "bert",  # "structured", "bert", "hybrid"
        bert_cache: PolicyEmbeddingCache = None,
        patch_size: int = config.PATCH_SIZE_PIXELS,
        num_patches_per_dim: int = config.NUM_PATCHES_PER_DIM,
        augment: bool = False,
        use_preprocessed: bool = True,
        normalize_policy: bool = True,
        normalize_on_gpu: bool = False
    ):
        """
        Args:
            samples: List of dicts with keys: 'city', 'year', 'tiff_path'/'npy_path', 'growth_rate'
            policy_source: Policy feature source ("structured", "bert", "hybrid")
            bert_cache: Pre-loaded BERT embedding cache (required for "bert" and "hybrid")
            patch_size: Size of each patch in pixels
            num_patches_per_dim: Number of patches per dimension
            augment: Whether to apply data augmentation
            use_preprocessed: Use preprocessed npy files if available
            normalize_policy: Whether to normalize policy features (only for structured)
            normalize_on_gpu: If True, skip CPU normalization (will be done on GPU during training)
        """
        self.samples = samples
        self.policy_source = policy_source
        self.bert_cache = bert_cache
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        self.augment = augment
        self.use_preprocessed = use_preprocessed and config.USE_PREPROCESSED_PATCHES
        self.normalize_policy = normalize_policy
        self.normalize_on_gpu = normalize_on_gpu

        # Validate configuration
        if policy_source in ["bert", "hybrid"] and bert_cache is None:
            raise ValueError(f"bert_cache is required for policy_source='{policy_source}'")

        # Initialize policy extractor for structured features
        if policy_source in ["structured", "hybrid"]:
            self.policy_extractor = get_policy_extractor()
        else:
            self.policy_extractor = None

        # Compute policy feature dimension
        if policy_source == "structured":
            self.policy_dim = 12
        elif policy_source == "bert":
            self.policy_dim = bert_cache.embedding_dim if bert_cache else 64
        elif policy_source == "hybrid":
            self.policy_dim = (bert_cache.embedding_dim if bert_cache else 64) + 12
        else:
            raise ValueError(f"Unknown policy_source: {policy_source}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load satellite image patches
        if self.use_preprocessed and 'npy_path' in sample:
            patches = np.load(sample['npy_path'])
            if patches.dtype == np.int8:
                patches = patches.astype(np.float32) / 127.0
                patches[patches < -1] = 0
            else:
                patches = patches / 127.0
                patches[patches < -1] = 0
        else:
            with rasterio.open(sample['tiff_path']) as src:
                data = src.read().astype(np.float32)
            data[data == -128] = 0
            data = data / 127.0
            data = self._pad_to_full_size(data)
            patches = self._extract_patches(data)

        if self.augment:
            patches = self._augment_patches(patches)

        patches = torch.FloatTensor(patches)
        if not self.normalize_on_gpu:
            patches = self._normalize_tensor(patches)

        # Get policy features based on source
        city = sample['city']
        year = sample['year']
        policy_feat = self._get_policy_features(city, year)
        policy_feat = torch.FloatTensor(policy_feat)

        # Label
        label = torch.FloatTensor([sample['growth_rate']])

        info = {
            'city': city,
            'year': year,
            'idx': idx
        }

        return patches, policy_feat, label, info

    def _get_policy_features(self, city: str, year: int) -> np.ndarray:
        """Get policy features based on configured source."""
        if self.policy_source == "structured":
            if self.normalize_policy:
                return self.policy_extractor.get_normalized_features(city, year)
            else:
                return self.policy_extractor.get_features(city, year)

        elif self.policy_source == "bert":
            return self.bert_cache.get_for_city(city, year)

        elif self.policy_source == "hybrid":
            bert_feat = self.bert_cache.get_for_city(city, year)
            if self.normalize_policy:
                struct_feat = self.policy_extractor.get_normalized_features(city, year)
            else:
                struct_feat = self.policy_extractor.get_features(city, year)
            return np.concatenate([bert_feat, struct_feat])

        else:
            raise ValueError(f"Unknown policy_source: {self.policy_source}")

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
        if patches.dim() == 4:
            mean = patches.mean(dim=(2, 3), keepdim=True)
            std = patches.std(dim=(2, 3), keepdim=True) + 1e-8
        else:
            mean = patches.mean(dim=(1, 2), keepdim=True)
            std = patches.std(dim=(1, 2), keepdim=True) + 1e-8
        return (patches - mean) / std

    def _augment_patches(self, patches: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            patches = patches[:, :, :, ::-1].copy()
        if random.random() > 0.5:
            patches = patches[:, :, ::-1, :].copy()
        if random.random() > 0.5:
            k = random.randint(1, 3)
            patches = np.rot90(patches, k, axes=(2, 3)).copy()
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.05, patches.shape).astype(np.float32)
            patches = patches + noise
        return patches


class PatchLevelPolicyBertDataset(Dataset):
    """
    Patch-level dataset with BERT/hybrid policy features.

    Each patch is an independent sample.

    Returns:
        - patch: (64, 200, 200) single satellite image patch
        - policy_features: (policy_dim,) policy features
        - label: (1,) population growth rate
        - info: dict with city/year/patch metadata
    """

    def __init__(
        self,
        samples: List[Dict],
        policy_source: str = "bert",
        bert_cache: PolicyEmbeddingCache = None,
        patch_size: int = config.PATCH_SIZE_PIXELS,
        num_patches_per_dim: int = config.NUM_PATCHES_PER_DIM,
        augment: bool = False,
        use_preprocessed: bool = True,
        normalize_policy: bool = True,
        normalize_on_gpu: bool = False
    ):
        self.samples = samples
        self.policy_source = policy_source
        self.bert_cache = bert_cache
        self.patch_size = patch_size
        self.num_patches_per_dim = num_patches_per_dim
        self.num_patches_total = num_patches_per_dim ** 2
        self.augment = augment
        self.use_preprocessed = use_preprocessed and config.USE_PREPROCESSED_PATCHES
        self.normalize_policy = normalize_policy
        self.normalize_on_gpu = normalize_on_gpu

        # Validate configuration
        if policy_source in ["bert", "hybrid"] and bert_cache is None:
            raise ValueError(f"bert_cache is required for policy_source='{policy_source}'")

        # Build index mapping
        self.index_mapping = []
        for sample_idx, sample in enumerate(samples):
            for patch_idx in range(self.num_patches_total):
                self.index_mapping.append((sample_idx, patch_idx))

        # Initialize policy extractor for structured features
        if policy_source in ["structured", "hybrid"]:
            self.policy_extractor = get_policy_extractor()
        else:
            self.policy_extractor = None

        # Compute policy feature dimension
        if policy_source == "structured":
            self.policy_dim = 12
        elif policy_source == "bert":
            self.policy_dim = bert_cache.embedding_dim if bert_cache else 64
        elif policy_source == "hybrid":
            self.policy_dim = (bert_cache.embedding_dim if bert_cache else 64) + 12
        else:
            raise ValueError(f"Unknown policy_source: {policy_source}")

        print(f"PatchLevelPolicyBertDataset: {len(samples)} city-years -> {len(self.index_mapping)} patch samples")
        print(f"  Policy source: {policy_source}, Policy dim: {self.policy_dim}")

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int):
        sample_idx, patch_idx = self.index_mapping[idx]
        sample = self.samples[sample_idx]

        # Load patch (same logic as PatchLevelDataset)
        if self.use_preprocessed and 'individual_patch_dir' in sample:
            patch_path = Path(sample['individual_patch_dir']) / f"{sample['base_name']}_p{patch_idx:02d}.npy"
            if patch_path.exists():
                patch = np.load(str(patch_path))
                if patch.dtype == np.int8:
                    patch = patch.astype(np.float32) / 127.0
                    patch[patch < -1] = 0
                else:
                    patch = patch / 127.0
                    patch[patch < -1] = 0
            else:
                if 'npy_path' in sample:
                    all_patches = np.load(sample['npy_path'])
                    patch = all_patches[patch_idx]
                    if patch.dtype == np.int8:
                        patch = patch.astype(np.float32) / 127.0
                        patch[patch < -1] = 0
                    else:
                        patch = patch / 127.0
                        patch[patch < -1] = 0
                else:
                    with rasterio.open(sample['tiff_path']) as src:
                        data = src.read().astype(np.float32)
                    data[data == -128] = 0
                    data = data / 127.0
                    data = self._pad_to_full_size(data)
                    patch = self._extract_single_patch(data, patch_idx)
        elif self.use_preprocessed and 'npy_path' in sample:
            all_patches = np.load(sample['npy_path'])
            patch = all_patches[patch_idx]
            if patch.dtype == np.int8:
                patch = patch.astype(np.float32) / 127.0
                patch[patch < -1] = 0
            else:
                patch = patch / 127.0
                patch[patch < -1] = 0
        else:
            with rasterio.open(sample['tiff_path']) as src:
                data = src.read().astype(np.float32)
            data[data == -128] = 0
            data = data / 127.0
            data = self._pad_to_full_size(data)
            patch = self._extract_single_patch(data, patch_idx)

        if self.augment:
            patch = self._augment_patch(patch)

        patch = torch.FloatTensor(patch)
        if not self.normalize_on_gpu:
            patch = self._normalize_tensor(patch)

        # Get policy features
        city = sample['city']
        year = sample['year']
        policy_feat = self._get_policy_features(city, year)
        policy_feat = torch.FloatTensor(policy_feat)

        # Label
        label = torch.FloatTensor([sample['growth_rate']])

        info = {
            'city': city,
            'year': year,
            'sample_idx': sample_idx,
            'patch_idx': patch_idx,
            'global_idx': idx
        }

        return patch, policy_feat, label, info

    def _get_policy_features(self, city: str, year: int) -> np.ndarray:
        """Get policy features based on configured source."""
        if self.policy_source == "structured":
            if self.normalize_policy:
                return self.policy_extractor.get_normalized_features(city, year)
            else:
                return self.policy_extractor.get_features(city, year)

        elif self.policy_source == "bert":
            return self.bert_cache.get_for_city(city, year)

        elif self.policy_source == "hybrid":
            bert_feat = self.bert_cache.get_for_city(city, year)
            if self.normalize_policy:
                struct_feat = self.policy_extractor.get_normalized_features(city, year)
            else:
                struct_feat = self.policy_extractor.get_features(city, year)
            return np.concatenate([bert_feat, struct_feat])

        else:
            raise ValueError(f"Unknown policy_source: {self.policy_source}")

    def _pad_to_full_size(self, data: np.ndarray, target_size: int = config.FULL_SIZE) -> np.ndarray:
        n_bands, h, w = data.shape
        if h >= target_size and w >= target_size:
            return data[:, :target_size, :target_size]
        padded = np.zeros((n_bands, target_size, target_size), dtype=data.dtype)
        h_start = (target_size - h) // 2
        w_start = (target_size - w) // 2
        padded[:, h_start:h_start+h, w_start:w_start+w] = data
        return padded

    def _extract_single_patch(self, data: np.ndarray, patch_idx: int) -> np.ndarray:
        ps = self.patch_size
        n = self.num_patches_per_dim
        i = patch_idx // n
        j = patch_idx % n
        y_start = i * ps
        x_start = j * ps
        return data[:, y_start:y_start+ps, x_start:x_start+ps]

    def _normalize_tensor(self, patch: torch.Tensor) -> torch.Tensor:
        mean = patch.mean(dim=(1, 2), keepdim=True)
        std = patch.std(dim=(1, 2), keepdim=True) + 1e-8
        return (patch - mean) / std

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            patch = patch[:, :, ::-1].copy()
        if random.random() > 0.5:
            patch = patch[:, ::-1, :].copy()
        if random.random() > 0.5:
            k = random.randint(1, 3)
            patch = np.rot90(patch, k, axes=(1, 2)).copy()
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.05, patch.shape).astype(np.float32)
            patch = patch + noise
        return patch

    def get_city_year_indices(self, sample_idx: int) -> List[int]:
        start_idx = sample_idx * self.num_patches_total
        return list(range(start_idx, start_idx + self.num_patches_total))


def get_bert_policy_dataloaders(
    policy_source: str = "bert",
    bert_cache: PolicyEmbeddingCache = None,
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = True,
    seed: int = config.RANDOM_SEED,
    normalize_on_gpu: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders with BERT/hybrid policy features (city-level).

    Args:
        policy_source: Policy feature source ("structured", "bert", "hybrid")
        bert_cache: Pre-loaded BERT embedding cache
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment_train: Whether to augment training data
        seed: Random seed for reproducibility
        normalize_on_gpu: If True, skip CPU normalization

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    from dataset import load_population_data, find_satellite_data, create_dataset_samples, split_dataset

    print("Loading population data...")
    pop_df = load_population_data()

    print("Finding satellite data...")
    sat_data = find_satellite_data()

    print("Creating samples...")
    samples = create_dataset_samples(pop_df, sat_data)
    print(f"Total valid samples: {len(samples)}")

    print(f"Splitting dataset (seed={seed})...")
    train_samples, val_samples, test_samples = split_dataset(samples, seed=seed)

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    print(f"  Policy source: {policy_source}")
    if normalize_on_gpu:
        print("  [GPU Normalization ENABLED] - CPU normalization will be skipped")

    train_dataset = CityPolicyBertDataset(
        train_samples, policy_source=policy_source, bert_cache=bert_cache,
        augment=augment_train, normalize_on_gpu=normalize_on_gpu
    )
    val_dataset = CityPolicyBertDataset(
        val_samples, policy_source=policy_source, bert_cache=bert_cache,
        augment=False, normalize_on_gpu=normalize_on_gpu
    )
    test_dataset = CityPolicyBertDataset(
        test_samples, policy_source=policy_source, bert_cache=bert_cache,
        augment=False, normalize_on_gpu=normalize_on_gpu
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    dataset_info = {
        'num_train': len(train_samples),
        'num_val': len(val_samples),
        'num_test': len(test_samples),
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'policy_source': policy_source,
        'policy_feature_dim': train_dataset.policy_dim,
        'seed': seed,
        'normalize_on_gpu': normalize_on_gpu
    }

    return train_loader, val_loader, test_loader, dataset_info


def get_patch_level_bert_policy_dataloaders(
    policy_source: str = "bert",
    bert_cache: PolicyEmbeddingCache = None,
    batch_size: int = 64,
    num_workers: int = 4,
    augment_train: bool = True,
    seed: int = config.RANDOM_SEED,
    normalize_on_gpu: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create patch-level dataloaders with BERT/hybrid policy features.

    Args:
        policy_source: Policy feature source ("structured", "bert", "hybrid")
        bert_cache: Pre-loaded BERT embedding cache
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment_train: Whether to augment training data
        seed: Random seed for reproducibility
        normalize_on_gpu: If True, skip CPU normalization

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    from dataset import load_population_data, find_satellite_data, create_dataset_samples, split_dataset

    print("Loading population data...")
    pop_df = load_population_data()

    print("Finding satellite data...")
    sat_data = find_satellite_data()

    print("Creating samples...")
    samples = create_dataset_samples(pop_df, sat_data)
    print(f"Total valid city-year samples: {len(samples)}")

    print(f"Splitting dataset by city (seed={seed})...")
    train_samples, val_samples, test_samples = split_dataset(samples, seed=seed)

    print(f"  Train: {len(train_samples)} city-years -> {len(train_samples) * config.NUM_PATCHES_TOTAL} patches")
    print(f"  Val: {len(val_samples)} city-years -> {len(val_samples) * config.NUM_PATCHES_TOTAL} patches")
    print(f"  Test: {len(test_samples)} city-years -> {len(test_samples) * config.NUM_PATCHES_TOTAL} patches")
    print(f"  Policy source: {policy_source}")
    if normalize_on_gpu:
        print("  [GPU Normalization ENABLED] - CPU normalization will be skipped")

    train_dataset = PatchLevelPolicyBertDataset(
        train_samples, policy_source=policy_source, bert_cache=bert_cache,
        augment=augment_train, normalize_on_gpu=normalize_on_gpu
    )
    val_dataset = PatchLevelPolicyBertDataset(
        val_samples, policy_source=policy_source, bert_cache=bert_cache,
        augment=False, normalize_on_gpu=normalize_on_gpu
    )
    test_dataset = PatchLevelPolicyBertDataset(
        test_samples, policy_source=policy_source, bert_cache=bert_cache,
        augment=False, normalize_on_gpu=normalize_on_gpu
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    dataset_info = {
        'num_train': len(train_samples),
        'num_val': len(val_samples),
        'num_test': len(test_samples),
        'num_train_patches': len(train_dataset),
        'num_val_patches': len(val_dataset),
        'num_test_patches': len(test_dataset),
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'num_patches_per_city': config.NUM_PATCHES_TOTAL,
        'training_mode': 'patch_level',
        'policy_source': policy_source,
        'policy_feature_dim': train_dataset.policy_dim,
        'seed': seed,
        'normalize_on_gpu': normalize_on_gpu
    }

    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    from policy_bert import get_policy_bert_cache

    # Test dataset loading with BERT cache
    print("=" * 60)
    print("Testing CITY-LEVEL BERT policy dataset...")
    print("=" * 60)

    # Load BERT cache
    bert_cache = get_policy_bert_cache()
    print(f"BERT cache loaded: {bert_cache.embedding_dim}-dim, {len(bert_cache.embeddings)} entries")

    # Test BERT mode
    print("\n--- BERT mode ---")
    train_loader, val_loader, test_loader, info = get_bert_policy_dataloaders(
        policy_source="bert",
        bert_cache=bert_cache,
        batch_size=2,
        num_workers=0
    )

    for patches, policy_feat, labels, meta in train_loader:
        print(f"Patches shape: {patches.shape}")        # (batch, 25, 64, 200, 200)
        print(f"Policy features shape: {policy_feat.shape}")  # (batch, 64)
        print(f"Labels shape: {labels.shape}")          # (batch, 1)
        print(f"Cities: {meta['city']}")
        print(f"Years: {meta['year']}")
        break

    # Test Hybrid mode
    print("\n--- Hybrid mode ---")
    train_loader_h, _, _, info_h = get_bert_policy_dataloaders(
        policy_source="hybrid",
        bert_cache=bert_cache,
        batch_size=2,
        num_workers=0
    )

    for patches, policy_feat, labels, meta in train_loader_h:
        print(f"Policy features shape (hybrid): {policy_feat.shape}")  # (batch, 76)
        break

    print(f"\nDataset info:")
    print(f"  Policy feature dim (BERT): {info['policy_feature_dim']}")
    print(f"  Policy feature dim (Hybrid): {info_h['policy_feature_dim']}")
