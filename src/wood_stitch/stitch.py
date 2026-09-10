import cv2
import json
import re
from pathlib import Path
from PIL import Image


def read_pixel_size(tile_path: Path) -> float | None:
    """
    Extract physical pixel size (µm/px) from OME-TIFF metadata.
    Returns None if not found.
    """
    try:
        img = Image.open(str(tile_path))
        xml = img.tag_v2.get(270, "")
        match = re.search(r'PhysicalSizeX="([^"]+)"', xml)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def load_tiles(tile_dir: str) -> tuple[list[tuple[str, object]], float | None]:
    """
    Load all tile images from a directory.
    Returns (tiles, pixel_size_um_per_px).
    """
    tile_dir = Path(tile_dir)
    tiles = []
    pixel_size = None

    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
        for path in sorted(tile_dir.glob(ext)):
            img = cv2.imread(str(path))
            if img is not None:
                tiles.append((path.name, img))
                print(f"  Loaded {path.name}")
                # Read pixel size from first tile only
                if pixel_size is None:
                    pixel_size = read_pixel_size(path)

    return tiles, pixel_size


def stitch(tile_dir: str, out_path: str = "mosaic.tif") -> None:
    print("Loading tiles …")
    tiles, pixel_size = load_tiles(tile_dir)
    print(f"  {len(tiles)} tiles found")

    if pixel_size:
        print(f"  Pixel size from metadata: {pixel_size:.6f} µm/px")
    else:
        print("  WARNING: No pixel size found in tile metadata")

    images = [img for _, img in tiles]

    print("Stitching …")
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, mosaic = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(out_path, mosaic)
        print(f"  Saved → {out_path}  ({mosaic.shape[1]}×{mosaic.shape[0]} px)")

        # Save metadata
        out_dir = Path(out_path).parent
        metadata = {
            "tile_pixel_size_um_per_px": pixel_size,
            "mosaic_width_px": mosaic.shape[1],
            "mosaic_height_px": mosaic.shape[0],
            "n_tiles": len(tiles),
            "resize_factor": 1.0,
            "effective_pixel_size_um_per_px": pixel_size,
            "notes": "resize_factor updated if mosaic is resized after stitching"
        }
        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved → {meta_path}")

    else:
        print(f"  Stitching failed with status code: {status}")