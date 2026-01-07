#!/usr/bin/env python3
"""
Preprocess TIFF files to NPY format for faster training.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess TIFF files to NPY")
    parser.add_argument("--input", required=True, help="Input directory containing TIFF files")
    parser.add_argument("--output", required=True, help="Output directory for NPY files")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--normalize", action="store_true", help="Normalize data during preprocessing")

    args = parser.parse_args()

    # Import after argument parsing
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.tiff_dataset import TIFFDataset

    logger.info(f"Preprocessing TIFF files from {args.input} to {args.output}")

    TIFFDataset.preprocess_all_tiffs(
        input_dir=args.input,
        output_dir=args.output,
        num_workers=args.num_workers,
        normalize=args.normalize,
    )

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
