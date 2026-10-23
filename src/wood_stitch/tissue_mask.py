# src/wood_stitch/tissue_mask.py
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import closing, opening, remove_small_objects, disk


def make_tissue_mask(
    img: np.ndarray,
    sat_thresh: int = 60,
    min_object_size: int = 5000,
    max_hole_px: int = 20000,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Returns a boolean mask (True = tissue) from a BGR uint8 mosaic.

    Tissue is stained, so it has colour saturation. Slide glass and black
    stitching seams have none. Brightness is NOT used: red-stained cell
    interiors are mid-gray, and a brightness threshold excluded them.
    sat_thresh=60 sits in the gap between glass (about 0) and tissue
    (about 160+); it is unvalidated beyond a glass-leakage check on 24 samples.

    The pixel settings (min_object_size, max_hole_px, and a disk of radius 5)
    were tuned on mosaics at half the native resolution (4082 x 7048 for
    Reference_636_Recent). `scale` is the linear size of the image relative
    to those: 1.0 for half-resolution mosaics, 2.0 for full resolution.
    """
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image; tissue detection uses colour saturation")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    mask = sat > sat_thresh

    selem = disk(max(1, round(5 * scale)))
    mask = closing(mask, selem)
    mask = opening(mask, selem)
    mask = remove_small_objects(mask, min_size=int(min_object_size * scale**2))

    # Fill only small holes fully enclosed by tissue (pale lumens). Background
    # that touches the image border is never a hole, and large enclosed gaps
    # (glass between tissue strips) stay excluded.
    holes = binary_fill_holes(mask) & ~mask
    labeled, n = label(holes)
    if n:
        sizes = np.bincount(labeled.ravel())
        small = sizes <= int(max_hole_px * scale**2)
        small[0] = False
        mask = mask | small[labeled]
    return mask


def apply_tissue_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Whites out non-tissue pixels in the image.
    """
    result = img.copy()
    result[~mask] = 255
    return result
