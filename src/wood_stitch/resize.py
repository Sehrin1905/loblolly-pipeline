import cv2
import json
from pathlib import Path


def resize_mosaic(mosaic_path: str, factor: float) -> None:
    """
    Resize a mosaic and update its metadata.json with the new effective pixel size.
    
    Args:
        mosaic_path: path to mosaic.tif
        factor: resize factor (e.g. 0.5 for 50%, 0.25 for 25%)
    """
    mosaic_path = Path(mosaic_path)
    out_dir = mosaic_path.parent
    meta_path = out_dir / "metadata.json"

    print(f"Loading {mosaic_path} ...")
    img = cv2.imread(str(mosaic_path))
    print(f"  Original size: {img.shape[1]}×{img.shape[0]} px")

    print(f"  Resizing by factor {factor} ...")
    resized = cv2.resize(img, (0, 0), fx=factor, fy=factor)
    cv2.imwrite(str(mosaic_path), resized)
    print(f"  New size: {resized.shape[1]}×{resized.shape[0]} px")

    # Update metadata
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

        # Accumulate resize factors
        old_factor = metadata.get("resize_factor", 1.0)
        new_factor = old_factor * factor
        tile_pixel_size = metadata.get("tile_pixel_size_um_per_px")

        metadata["resize_factor"] = new_factor
        metadata["mosaic_width_px"] = resized.shape[1]
        metadata["mosaic_height_px"] = resized.shape[0]
        if tile_pixel_size:
            metadata["effective_pixel_size_um_per_px"] = tile_pixel_size / new_factor

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata updated → effective pixel size: {metadata.get('effective_pixel_size_um_per_px', 'unknown'):.4f} µm/px")
    else:
        print("  WARNING: No metadata.json found — pixel size not tracked")