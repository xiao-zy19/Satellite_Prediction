"""
Preprocess TIFF files to extract and save patches as npy files.

This significantly speeds up training by:
1. Pre-extracting all 25 patches from each TIFF
2. Saving as compressed npy files for fast loading
3. Avoiding repeated TIFF decompression during training
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import argparse

# Configuration
PATCH_SIZE = 200  # 2km x 2km patches (200 pixels at 10m resolution)
NUM_PATCHES_PER_DIM = 5  # 5x5 = 25 patches per tile
FULL_SIZE = 1000  # Full tile size


def extract_patches(data: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Extract non-overlapping patches from a tile.
    
    Args:
        data: (C, H, W) array
        patch_size: Size of each patch
        
    Returns:
        patches: (num_patches, C, patch_size, patch_size) array
    """
    c, h, w = data.shape
    patches = []
    
    for i in range(NUM_PATCHES_PER_DIM):
        for j in range(NUM_PATCHES_PER_DIM):
            y_start = i * patch_size
            x_start = j * patch_size
            patch = data[:, y_start:y_start+patch_size, x_start:x_start+patch_size]
            patches.append(patch)
    
    return np.stack(patches, axis=0)  # (25, C, 200, 200)


def process_single_tiff(args):
    """Process a single TIFF file."""
    tiff_path, output_dir = args
    
    try:
        # Read TIFF
        with rasterio.open(tiff_path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
        
        # Validate size
        if data.shape[1] != FULL_SIZE or data.shape[2] != FULL_SIZE:
            return f"Skip {tiff_path}: unexpected size {data.shape}"
        
        # Extract patches
        patches = extract_patches(data)  # (25, 64, 200, 200)
        
        # Create output path
        # e.g., 北京/北京市/2018_北京市.tiff -> 北京/北京市/2018_北京市_patches.npy
        rel_path = tiff_path.relative_to(tiff_path.parent.parent.parent)
        output_path = output_dir / rel_path.with_suffix('.npy')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as compressed npy
        np.save(output_path, patches)
        
        return f"OK: {output_path.name}"
        
    except Exception as e:
        return f"Error {tiff_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess TIFF files to patches")
    parser.add_argument('--input_dir', type=str, 
                        default="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data_local/city_satellite_tiles",
                        help="Input directory with TIFF files")
    parser.add_argument('--output_dir', type=str,
                        default="/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data_local/city_patches",
                        help="Output directory for patches")
    parser.add_argument('--workers', type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIFF files
    tiff_files = list(input_dir.glob('**/*.tiff'))
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Filter out non-data files (visualization images, etc.)
    tiff_files = [f for f in tiff_files if f.stem.split('_')[0].isdigit()]
    print(f"Processing {len(tiff_files)} data TIFF files")
    
    # Process in parallel
    tasks = [(f, output_dir) for f in tiff_files]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_single_tiff, tasks),
            total=len(tasks),
            desc="Preprocessing"
        ))
    
    # Summary
    ok_count = sum(1 for r in results if r.startswith("OK"))
    error_count = sum(1 for r in results if r.startswith("Error"))
    skip_count = sum(1 for r in results if r.startswith("Skip"))
    
    print(f"\nDone!")
    print(f"  Success: {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")
    
    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if r.startswith("Error"):
                print(f"  {r}")
    
    # Calculate space savings
    input_size = sum(f.stat().st_size for f in tiff_files) / (1024**3)
    output_files = list(output_dir.glob('**/*.npy'))
    output_size = sum(f.stat().st_size for f in output_files) / (1024**3) if output_files else 0
    
    print(f"\nStorage:")
    print(f"  Input TIFF: {input_size:.2f} GB")
    print(f"  Output NPY: {output_size:.2f} GB")


if __name__ == "__main__":
    main()
