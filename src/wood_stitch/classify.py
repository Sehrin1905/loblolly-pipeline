# src/wood_stitch/classify.py
import numpy as np
import pandas as pd


def classify_cells(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based classification of cell types from morphometric features.
    Thresholds are median-relative (robust to image-level brightness/scale 
    differences) and explicitly flagged as tunable.

    Returns features_df with an added 'cell_type' column.
    """
    df = features_df.copy()
    median_area = df["area"].median()

    # Default: tracheid (everything else, see below)
    df["cell_type"] = "tracheid"

    # Resin duct: unusually large and roughly round
    is_resin_duct = (df["area"] > 10 * median_area) & (df["eccentricity"] < 0.7)
    df.loc[is_resin_duct, "cell_type"] = "resin_duct"

    # Ray: elongated and not huge, excluding resin ducts
    is_ray = (
        (df["eccentricity"] > 0.85)
        & (df["area"] < 2 * median_area)
        & (~is_resin_duct)
    )
    df.loc[is_ray, "cell_type"] = "ray"

    print(f"  Median area: {median_area:.1f} px")
    print(f"  Resin ducts: {is_resin_duct.sum()}")
    print(f"  Rays: {is_ray.sum()}")
    print(f"  Tracheids: {(df['cell_type'] == 'tracheid').sum()}")

    return df


if __name__ == "__main__":
    import sys

    features_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.csv"
    out_path      = sys.argv[2] if len(sys.argv) > 2 else "data/classified.csv"

    print(f"Loading {features_path} ...")
    df = pd.read_csv(features_path)

    print("Classifying cells ...")
    df = classify_cells(df)

    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")