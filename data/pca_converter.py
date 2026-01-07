"""
PCA Converter for converting 64-channel embeddings to 3-channel RGB.
Used for Route B: PCA dimensionality reduction approach.
"""

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
from typing import Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
import joblib

logger = logging.getLogger(__name__)


class PCAConverter:
    """
    PCA-based converter for 64-channel to 3-channel transformation.

    This is used for Route B where we convert 64-channel satellite embeddings
    to 3-channel pseudo-RGB images for standard MLLM input.
    """

    def __init__(
        self,
        n_components: int = 3,
        model_path: Optional[str] = None,
    ):
        """
        Initialize PCA Converter.

        Args:
            n_components: Number of output components (default 3 for RGB)
            model_path: Path to load pre-fitted PCA model
        """
        self.n_components = n_components

        if model_path and Path(model_path).exists():
            self.pca = joblib.load(model_path)
            logger.info(f"Loaded PCA model from {model_path}")
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        else:
            self.pca = IncrementalPCA(n_components=n_components)
            logger.info("Created new IncrementalPCA model")

    def fit(
        self,
        data_dir: str,
        batch_size: int = 10,
        max_samples: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Fit PCA model on dataset using incremental learning.

        Args:
            data_dir: Directory containing NPY files
            batch_size: Number of files to process in each batch
            max_samples: Maximum number of samples to use (None for all)
            save_path: Path to save fitted model
        """
        data_dir = Path(data_dir)
        files = sorted(list(data_dir.glob("**/*.npy")))

        if max_samples:
            files = files[:max_samples]

        logger.info(f"Fitting PCA on {len(files)} files")

        # Process in batches
        for i in tqdm(range(0, len(files), batch_size), desc="Fitting PCA"):
            batch_files = files[i:i + batch_size]

            # Load and reshape batch
            batch_data = []
            for f in batch_files:
                data = np.load(f)  # (64, H, W)
                # Reshape to (H*W, 64)
                flat = data.reshape(64, -1).T
                # Subsample to reduce memory
                if flat.shape[0] > 10000:
                    indices = np.random.choice(flat.shape[0], 10000, replace=False)
                    flat = flat[indices]
                batch_data.append(flat)

            # Concatenate and fit
            batch_array = np.concatenate(batch_data, axis=0)
            self.pca.partial_fit(batch_array)

        logger.info(f"PCA fitting complete")
        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        logger.info(f"Total explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        # Save model
        if save_path:
            joblib.dump(self.pca, save_path)
            logger.info(f"Saved PCA model to {save_path}")

    def transform(
        self,
        data: Union[np.ndarray, torch.Tensor],
        normalize_output: bool = True,
    ) -> np.ndarray:
        """
        Transform 64-channel data to 3-channel.

        Args:
            data: Input array of shape (64, H, W)
            normalize_output: Whether to normalize to [0, 255]

        Returns:
            Transformed array of shape (3, H, W)
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # Get original shape
        c, h, w = data.shape
        assert c == 64, f"Expected 64 channels, got {c}"

        # Reshape to (H*W, 64)
        flat = data.reshape(64, -1).T

        # Transform
        transformed = self.pca.transform(flat)

        # Reshape back to (3, H, W)
        output = transformed.T.reshape(self.n_components, h, w)

        # Normalize to [0, 255] if needed
        if normalize_output:
            output = self._normalize_to_uint8(output)

        return output

    def transform_batch(
        self,
        data: Union[np.ndarray, torch.Tensor],
        normalize_output: bool = True,
    ) -> np.ndarray:
        """
        Transform a batch of 64-channel data to 3-channel.

        Args:
            data: Input array of shape (B, 64, H, W)
            normalize_output: Whether to normalize to [0, 255]

        Returns:
            Transformed array of shape (B, 3, H, W)
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        b, c, h, w = data.shape
        assert c == 64, f"Expected 64 channels, got {c}"

        outputs = []
        for i in range(b):
            output = self.transform(data[i], normalize_output)
            outputs.append(output)

        return np.stack(outputs, axis=0)

    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 255] range."""
        # Per-channel normalization
        for i in range(data.shape[0]):
            channel = data[i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                data[i] = (channel - min_val) / (max_val - min_val) * 255
            else:
                data[i] = 0

        return data.astype(np.uint8)

    def convert_dataset(
        self,
        input_dir: str,
        output_dir: str,
        num_workers: int = 8,
        normalize: bool = True,
    ) -> None:
        """
        Convert entire dataset from 64-channel to 3-channel.

        Args:
            input_dir: Directory containing 64-channel NPY files
            output_dir: Directory to save 3-channel files
            num_workers: Number of parallel workers
            normalize: Whether to normalize output
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(input_dir.glob("**/*.npy")))
        logger.info(f"Converting {len(files)} files from 64ch to 3ch")

        def convert_one(file_path: Path) -> Tuple[str, bool]:
            try:
                output_path = output_dir / file_path.name

                if output_path.exists():
                    return str(file_path), True

                data = np.load(file_path)
                converted = self.transform(data, normalize)
                np.save(output_path, converted)

                return str(file_path), True
            except Exception as e:
                logger.error(f"Error converting {file_path}: {e}")
                return str(file_path), False

        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(convert_one, files),
                total=len(files),
                desc="Converting files"
            ))

        success_count = sum(1 for _, success in results if success)
        logger.info(f"Successfully converted {success_count}/{len(files)} files")


class PseudoRGBDataset(torch.utils.data.Dataset):
    """
    Dataset that loads 3-channel pseudo-RGB images converted from 64-channel.
    Used with standard vision models expecting RGB input.
    """

    def __init__(
        self,
        data_dir: str,
        pca_model: Optional[PCAConverter] = None,
        source_64ch: bool = False,
        image_size: int = 1000,
    ):
        """
        Initialize Pseudo-RGB Dataset.

        Args:
            data_dir: Directory containing data files
            pca_model: PCA model for on-the-fly conversion
            source_64ch: If True, load 64ch and convert; if False, load pre-converted
            image_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.pca_model = pca_model
        self.source_64ch = source_64ch
        self.image_size = image_size

        self.files = sorted(list(self.data_dir.glob("**/*.npy")))
        logger.info(f"Found {len(self.files)} files for Pseudo-RGB dataset")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return RGB image."""
        data = np.load(self.files[idx])

        if self.source_64ch and self.pca_model:
            # Convert on-the-fly
            data = self.pca_model.transform(data, normalize_output=True)
        else:
            # Already 3-channel
            pass

        # Resize if needed
        if data.shape[1] != self.image_size:
            from scipy.ndimage import zoom
            scale = self.image_size / data.shape[1]
            data = zoom(data, (1, scale, scale), order=1)

        # Normalize to [0, 1]
        data = data.astype(np.float32) / 255.0

        return torch.FloatTensor(data)


def main():
    """Example usage of PCA converter."""
    import argparse

    parser = argparse.ArgumentParser(description="PCA Converter for satellite embeddings")
    parser.add_argument("--action", choices=["fit", "convert"], required=True)
    parser.add_argument("--input-dir", required=True, help="Input data directory")
    parser.add_argument("--output-dir", help="Output directory (for convert)")
    parser.add_argument("--model-path", help="Path to PCA model")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()

    if args.action == "fit":
        converter = PCAConverter(n_components=3)
        converter.fit(
            data_dir=args.input_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            save_path=args.model_path,
        )
    elif args.action == "convert":
        if not args.model_path:
            raise ValueError("--model-path required for convert action")
        if not args.output_dir:
            raise ValueError("--output-dir required for convert action")

        converter = PCAConverter(model_path=args.model_path)
        converter.convert_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
