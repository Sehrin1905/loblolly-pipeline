"""Descriptive sample summaries with one explicit cell-count denominator."""

import pandas as pd
import numpy as np

CLASSES = ["ray", "tracheid", "resin_duct"]


def summarize(cells, samples):
    if "area_um2" not in cells or not {"sample_id", "cell_type"} <= set(cells):
        raise ValueError("Expected calibrated cells with explicit sample IDs")
    if not set(cells["cell_type"].dropna()) <= set(CLASSES):
        raise ValueError("Unknown cell class")
    if (~np.isfinite(cells["area_um2"])).any() or (cells["area_um2"] <= 0).any():
        raise ValueError("Cell areas must be finite positive µm² values")
    rows = []
    for sample in samples:
        name = sample["sample_id"]
        subset = cells[cells["sample_id"] == name]
        row = dict(sample)
        row["total_cells"] = len(subset)
        row["ray_fraction_denominator"] = "all_classified_lumens"
        for kind in CLASSES:
            count = int((subset["cell_type"] == kind).sum())
            row[kind + "_count"] = count
            row[kind + "_percent"] = 100 * count / len(subset) if len(subset) else float("nan")
        row["mean_lumen_area_um2"] = subset["area_um2"].mean()
        row["classification_validation"] = "unvalidated rules; descriptive only"
        rows.append(row)
    if set(cells["sample_id"]) - {s["sample_id"] for s in samples}:
        raise ValueError("Cells include an unregistered sample")
    return pd.DataFrame(rows)
