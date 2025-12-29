"""
Preprocess TIFF files to individual patch npy files.
Each patch saved separately for faster random access.
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

PATCH_SIZE = 200
NUM_PATCHES_PER_DIM = 5


def process_single_tiff(args):
    tiff_path, output_dir = args

    try:
        with rasterio.open(tiff_path) as src:
            # 保持int8格式，不转换为float32！
            data = src.read()  # int8

        if data.shape[1] < 1000 or data.shape[2] < 1000:
            return f"Skip {tiff_path}: size {data.shape}"

        # Extract base name
        base_name = tiff_path.stem  # e.g., "2018_北京市"
        rel_path = tiff_path.relative_to(tiff_path.parent.parent.parent)
        out_base = output_dir / rel_path.parent
        out_base.mkdir(parents=True, exist_ok=True)

        # Extract and save each patch
        for i in range(NUM_PATCHES_PER_DIM):
            for j in range(NUM_PATCHES_PER_DIM):
                patch_idx = i * NUM_PATCHES_PER_DIM + j
                y_start = i * PATCH_SIZE
                x_start = j * PATCH_SIZE
                patch = data[:, y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE]

                # Save individual patch as int8 (比float32小4倍: 2.4MB vs 9.8MB)
                patch_path = out_base / f"{base_name}_p{patch_idx:02d}.npy"
                np.save(patch_path, patch)

        return f"OK ({data.dtype})"

    except Exception as e:
        return f"Error {tiff_path}: {e}"


def main():
    input_dir = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data_local/city_satellite_tiles")
    output_dir = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data_local/city_individual_patches")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIFF files
    tiff_files = [f for f in input_dir.glob('**/*.tiff') if f.stem.split('_')[0].isdigit()]
    print(f"Processing {len(tiff_files)} TIFF files")
    
    tasks = [(f, output_dir) for f in tiff_files]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_single_tiff, tasks), total=len(tasks)))
    
    ok = sum(1 for r in results if r == "OK")
    print(f"Done! Success: {ok}/{len(tiff_files)}")


if __name__ == "__main__":
    main()
