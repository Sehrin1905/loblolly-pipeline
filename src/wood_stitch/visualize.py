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


if __name__ == "__main__":
    import sys

    mosaic_path     = sys.argv[1] if len(sys.argv) > 1 else "data/mosaic.tif"
    labels_path     = sys.argv[2] if len(sys.argv) > 2 else "data/labels.npy"
    classified_path = sys.argv[3] if len(sys.argv) > 3 else "data/classified.csv"
    out_path        = sys.argv[4] if len(sys.argv) > 4 else "data/overlay.png"

    print(f"Loading {mosaic_path} ...")
    img = cv2.imread(mosaic_path)

    print(f"Loading {labels_path} ...")
    labels = np.load(labels_path)

    print(f"Loading {classified_path} ...")
    classified_df = pd.read_csv(classified_path)

    print("Building overlay ...")
    overlay = make_overlay(img, labels, classified_df)

    cv2.imwrite(out_path, overlay)
    print(f"Saved → {out_path}")
    print(f"  Legend: gray=tracheid, green=ray, red=resin_duct")
    