# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pandas", "scikit-image", "scipy"]
# ///
"""
Extract per-cell morphometric features from label map.

Usage:
    uv run scripts/features.py <labels_path> <out_path> [--metadata <metadata_path>]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
from wood_stitch.features import compute_features

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels_path")
    ap.add_argument("out_path")
    ap.add_argument("--metadata", help="Path to metadata.json for pixel size conversion")
    args = ap.parse_args()

    print(f"Loading {args.labels_path} ...")
    labels = np.load(args.labels_path)

    # Read pixel size from metadata if available
    pixel_size_um = None
    meta_path = args.metadata or Path(args.labels_path).parent / "metadata.json"
    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
        pixel_size_um = meta.get("effective_pixel_size_um_per_px")
        if pixel_size_um:
            print(f"Pixel size from metadata: {pixel_size_um:.4f} µm/px")
        else:
            print("WARNING: metadata.json found but no effective_pixel_size_um_per_px")
    else:
        print("WARNING: No metadata.json found — measurements will be in pixels")

    print("Computing features ...")
    df = compute_features(labels, pixel_size_um=pixel_size_um)

    df.to_csv(args.out_path, index=False)
    print(f"Saved → {args.out_path}  ({len(df)} cells)")