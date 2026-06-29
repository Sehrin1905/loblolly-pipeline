# scripts/stitch.py
import cv2
from pathlib import Path


def load_tiles(tile_dir: str) -> list[tuple[str, object]]:
    """Load all tile images from a directory."""
    tile_dir = Path(tile_dir)
    tiles = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
        for path in sorted(tile_dir.glob(ext)):
            img = cv2.imread(str(path))
            if img is not None:
                tiles.append((path.name, img))
                print(f"  Loaded {path.name}")
    return tiles


def stitch(tile_dir: str, out_path: str = "mosaic.tif") -> None:
    print("Loading tiles …")
    tiles = load_tiles(tile_dir)
    print(f"  {len(tiles)} tiles found")

    images = [img for _, img in tiles]

    print("Stitching …")
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, mosaic = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(out_path, mosaic)
        print(f"  Saved → {out_path}  ({mosaic.shape[1]}×{mosaic.shape[0]} px)")
    else:
        print(f"  Stitching failed with status code: {status}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tile_dir")
    ap.add_argument("--out", default="mosaic.tif")
    args = ap.parse_args()
    stitch(args.tile_dir, args.out)