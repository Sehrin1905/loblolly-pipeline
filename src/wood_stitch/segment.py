# src/wood_stitch/segment.py
import numpy as np
import cv2
from cellpose import models


def segment(img: np.ndarray,
            tissue_mask: np.ndarray,
            bsize: int = 256,
            gpu: bool = False) -> np.ndarray:
    """
    Run Cellpose on the mosaic to find cell lumens.
    Returns an (H, W) int32 label map: 0 = background, 1..N = cell lumens.

    Args:
        img: RGB image (H, W, 3)
        tissue_mask: boolean mask (H, W), True = tissue
        bsize: patch size for cpsam model (must be 256)
        gpu: use GPU if available
    """
    # 1. White out non-tissue pixels so Cellpose ignores background
    img_masked = img.copy()
    img_masked[~tissue_mask] = 255

    # 2. Run Cellpose cpsam model
    print("  Loading Cellpose model ...")
    model = models.CellposeModel(gpu=gpu, model_type="cpsam")

    print("  Running segmentation ...")
    masks, flows, styles = model.eval(
        img_masked,
        diameter=None,
        bsize=bsize,
        channels=[0, 0]  # grayscale mode
    )

    print(f"  Found {masks.max()} cell lumens")
    return masks.astype(np.int32)


if __name__ == "__main__":
    import sys

    mosaic_path = sys.argv[1] if len(sys.argv) > 1 else "data/mosaic.tif"
    mask_path   = sys.argv[2] if len(sys.argv) > 2 else "data/tissue_mask.png"
    out_path    = sys.argv[3] if len(sys.argv) > 3 else "data/labels.npy"

    print(f"Loading {mosaic_path} ...")
    img = cv2.imread(mosaic_path)

    print(f"Loading {mask_path} ...")
    tissue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127

    print("Running segmentation ...")
    labels = segment(img, tissue_mask)

    np.save(out_path, labels)
    print(f"Saved → {out_path}  ({labels.max()} cells)")