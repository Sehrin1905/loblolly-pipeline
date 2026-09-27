# src/wood_stitch/tissue_mask.py
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, remove_small_objects, disk


def make_tissue_mask(
    img: np.ndarray, bright_thresh: float = 0.90, dark_thresh: float = 0.05, min_object_size: int = 5000
) -> np.ndarray:
    """
    Returns a boolean mask (True = tissue) from an RGB or grayscale mosaic.
    """
    # 1. Convert to grayscale float
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.float32) / 255.0

    # 2. Hard-threshold out bright (slide glass) and dark (stitching seams)
    candidate = (gray > dark_thresh) & (gray < bright_thresh)

    # 3. Otsu on the remaining pixels to find tissue/non-tissue boundary
    candidate_pixels = gray[candidate]
    if candidate_pixels.size == 0 or np.ptp(candidate_pixels) == 0:
        return np.zeros(gray.shape, dtype=bool)
    otsu_thresh = threshold_otsu(candidate_pixels)
    mask = candidate & (gray < otsu_thresh)

    # 4. Clean up with closing → opening → remove small objects
    selem = disk(5)
    mask = closing(mask, selem)
    mask = opening(mask, selem)
    mask = remove_small_objects(mask, min_size=min_object_size)

    # Include enclosed bright lumens; black seams remain excluded. This heuristic needs image QC.
    mask = binary_fill_holes(mask) & (gray > dark_thresh)
    return mask


def apply_tissue_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Whites out non-tissue pixels in the image.
    """
    result = img.copy()
    result[~mask] = 255
    return result
