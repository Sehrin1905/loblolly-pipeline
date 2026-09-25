# src/wood_stitch/classify.py
import pandas as pd


def classify_cells(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based classification of cell types from morphometric features.
    Thresholds are median-relative (robust to image-level brightness/scale
    differences) and explicitly flagged as tunable.

    Returns features_df with an added 'cell_type' column.
    """
    df = features_df.copy()
    if "area_um2" not in df:
        raise ValueError("Classification requires calibrated area_um2; regenerate legacy feature tables")
    median_area = df["area_um2"].median()

    # Default: tracheid (everything else, see below)
    df["cell_type"] = "tracheid"
    df["classification_method"] = "unvalidated_area_eccentricity_rules_v1"

    # Resin duct: unusually large and roughly round
    is_resin_duct = (df["area_um2"] > 10 * median_area) & (df["eccentricity"] < 0.7)
    df.loc[is_resin_duct, "cell_type"] = "resin_duct"

    # Ray: elongated and not huge, excluding resin ducts
    is_ray = (df["eccentricity"] > 0.90) & (df["area_um2"] < 2 * median_area) & (~is_resin_duct)
    df.loc[is_ray, "cell_type"] = "ray"

    print(f"  Median area: {median_area:.1f} µm²")
    print(f"  Resin ducts: {is_resin_duct.sum()}")
    print(f"  Rays: {is_ray.sum()}")
    print(f"  Tracheids: {(df['cell_type'] == 'tracheid').sum()}")

    return df
