# tests/test_zones.py
import numpy as np
import pandas as pd

from wood_stitch.classify import classify_cells


def _grid(n=20, spacing=10.0, growth=0.15):
    """n x n grid of cells whose area grows smoothly along x."""
    rows = []
    for iy in range(n):
        for ix in range(n):
            rows.append(
                {
                    "label": len(rows) + 1,
                    "centroid_x_um": ix * spacing,
                    "centroid_y_um": iy * spacing,
                    "area_um2": 100.0 * np.exp(growth * ix),
                }
            )
    return pd.DataFrame(rows)


def test_zones_follow_area_gradient():
    out = classify_cells(_grid())
    assert set(out["zone"].dropna()) == {"latewood-like", "transition", "earlywood-like"}
    left = out[out["centroid_x_um"] == 0]
    right = out[out["centroid_x_um"] == 190]
    assert (left["zone"] == "latewood-like").all()
    assert (right["zone"] == "earlywood-like").all()
    assert out["zone_score"].corr(out["centroid_x_um"]) > 0.9


def test_large_object_gets_no_zone_and_does_not_distort():
    df = _grid()
    base = classify_cells(df)
    big = pd.DataFrame([{"label": 9999, "centroid_x_um": 5.0, "centroid_y_um": 5.0, "area_um2": 1e7}])
    out = classify_cells(pd.concat([df, big], ignore_index=True))
    row = out[out["label"] == 9999].iloc[0]
    assert row["cell_type"] == "large_object"
    assert pd.isna(row["zone"]) and pd.isna(row["zone_score"])
    same = out[out["label"] != 9999].reset_index(drop=True)
    # The tear sits next to the left edge but must not change anyone's score.
    # (Median area shifts slightly, but no tracheid becomes a large object.)
    assert np.allclose(same["zone_score"], base["zone_score"])


def test_small_sample_leaves_zones_empty():
    out = classify_cells(_grid(n=8))  # 64 cells
    assert out["zone"].isna().all()
    assert out["zone_score"].isna().all()


def test_constant_areas_leave_zones_empty():
    df = _grid()
    df["area_um2"] = 250.0
    out = classify_cells(df)
    assert out["zone"].isna().all()
    assert out["zone_score"].isna().all()
