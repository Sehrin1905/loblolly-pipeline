import numpy as np
import cv2
from cellpose import models


def segment(img: np.ndarray,
            tissue_mask: np.ndarray,
            bsize: int = 256,
            gpu: bool = False,
            diameter: float | None = None) -> np.ndarray:
    """
    Run Cellpose on the mosaic to find cell lumens.
    Returns an (H, W) int32 label map: 0 = background, 1..N = cell lumens.

    Args:
        img: RGB image (H, W, 3)
        tissue_mask: boolean mask (H, W), True = tissue
        bsize: patch size for cpsam model (must be 256)
        gpu: use GPU if available
        diameter: expected cell diameter in pixels (None = auto-detect)
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
        diameter=diameter,
        bsize=bsize,
        channels=[0, 0]  # grayscale mode
    )

    print(f"  Found {masks.max()} cell lumens")
    return masks.astype(np.int32)