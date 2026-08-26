# src/wood_stitch/visualize.py
import numpy as np
import pandas as pd
import cv2

COLOR_MAP = {
    "tracheid": (200, 200, 200),    # light gray
    "ray": (0, 200, 0),             # green
    "resin_duct": (0, 0, 255),      # red (BGR)
}


def make_overlay(img: np.ndarray,
                  labels: np.ndarray,
                  classified_df: pd.DataFrame,
                  alpha: float = 0.5) -> np.ndarray:
    """
    Color each cell lumen by its classified type, blended over the original image.
    """
    overlay = np.zeros_like(img)

    label_to_type = dict(zip(classified_df["label"], classified_df["cell_type"]))

    for cell_type, color in COLOR_MAP.items():
        labels_of_type = [lbl for lbl, t in label_to_type.items() if t == cell_type]
        if not labels_of_type:
            continue
        mask = np.isin(labels, labels_of_type)
        overlay[mask] = color

    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended


