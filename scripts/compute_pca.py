#!/usr/bin/env python3
"""
Compute PCA model for 64ch to 3ch conversion (Route B).
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute PCA for channel reduction")
    parser.add_argument("--input", required=True, help="Input directory containing NPY files")
    parser.add_argument("--output", required=True, help="Output path for PCA model (.pkl)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for incremental PCA")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use")
    parser.add_argument("--n-components", type=int, default=3, help="Number of PCA components")

    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.pca_converter import PCAConverter

    logger.info(f"Computing PCA model from {args.input}")

    converter = PCAConverter(n_components=args.n_components)
    converter.fit(
        data_dir=args.input,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_path=args.output,
    )

    logger.info(f"PCA model saved to {args.output}")


if __name__ == "__main__":
    main()
