# src/wood_stitch/classify.py
import pandas as pd


def classify_cells(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based classification of segmented cell lumens.

    Rays are NOT classified. In these transverse sections rays are dark
    astra blue bands that Cellpose mostly does not segment, and shape rules
    (eccentricity, orientation) mislabel compressed tracheids as rays.
    Every segmented cell is therefore a tracheid, except very large round
    resin ducts. Ray counts are unmeasured, not zero.

    Returns features_df with an added 'cell_type' column.
    """
    df = features_df.copy()
    if "area_um2" not in df:
        raise ValueError("Classification requires calibrated area_um2; regenerate legacy feature tables")
    median_area = df["area_um2"].median()

    # Default: tracheid
    df["cell_type"] = "tracheid"
    df["classification_method"] = "unvalidated_tracheid_only_rules_v3"

    # Resin duct: unusually large and roughly round (tunable, unvalidated)
    is_resin_duct = (df["area_um2"] > 10 * median_area) & (df["eccentricity"] < 0.7)
    df.loc[is_resin_duct, "cell_type"] = "resin_duct"

    print(f"  Median area: {median_area:.1f} µm²")
    print(f"  Resin ducts: {is_resin_duct.sum()}")
    print(f"  Tracheids: {(df['cell_type'] == 'tracheid').sum()}")
    print("  Rays: not classified (unmeasured)")

    return df
