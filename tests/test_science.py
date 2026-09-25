import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import tifffile
from wood_stitch.image_io import read_image, write_image, tiff_info
from wood_stitch.resize import resize_mosaic
from wood_stitch.features import compute_features, compute_wall_thickness
from wood_stitch.classify import classify_cells
from wood_stitch.adjacency import compute_adjacency
from wood_stitch.tissue_mask import make_tissue_mask
from wood_stitch.segment import segment
from wood_stitch.stitch import stitch
from wood_stitch.visualize import make_overlay
from wood_stitch.analysis import summarize


def test_calibrated_area_anisotropic_centroids_and_empty():
    labels = np.zeros((30, 30), np.uint32)
    labels[5:15, 8:18] = 1
    f = compute_features(labels, pixel_size_um=(0.5, 0.25)).iloc[0]
    assert f.area_um2 == 12.5
    assert f.area_px2 == 100
    assert f.centroid_x_um == f.centroid_x_px * 0.5
    assert f.centroid_y_um == f.centroid_y_px * 0.25
    assert "wall_thickness" not in f.index
    empty = compute_features(np.zeros((30, 30), np.uint32), pixel_size_um=(0.5, 0.5))
    assert empty.empty and "area_um2" in empty
    assert classify_cells(empty).empty


@pytest.mark.parametrize("scale", [None, 0, -1, float("nan"), float("inf")])
def test_bad_calibration_rejected(scale):
    with pytest.raises(ValueError):
        compute_features(np.ones((10, 10), np.uint32), pixel_size_um=scale)


def test_wall_metric_disabled():
    with pytest.raises(NotImplementedError, match="not wall thickness"):
        compute_wall_thickness(np.ones((5, 5), np.uint32))


def test_tiff_copy_resize_round_trip(source, tmp_path):
    renamed = tmp_path / "renamed.tiff"
    renamed.write_bytes(source.read_bytes())
    original = source.read_bytes()
    out = resize_mosaic(renamed, 0.5)
    image, scale = read_image(out)
    assert image.shape == (32, 32, 3) and scale == (1.0, 1.0)
    assert source.read_bytes() == renamed.read_bytes() == original
    assert 64 * 64 * 0.5 * 0.5 == 32 * 32 * scale[0] * scale[1]
    with pytest.raises(ValueError):
        resize_mosaic(source, 0.5, source)


def test_rounding_updates_actual_pixel_scale(tmp_path):
    p = tmp_path / "odd.tif"
    write_image(p, np.zeros((31, 33), np.uint8), (0.5, 0.5))
    image, (sx, sy) = read_image(resize_mosaic(p, 0.5))
    assert np.isclose(image.shape[1] * sx, 33 * 0.5)
    assert np.isclose(image.shape[0] * sy, 31 * 0.5)


def test_non_um_units_and_embedded_identity(tmp_path):
    p = tmp_path / "mm.tif"
    tifffile.imwrite(
        p,
        np.ones((10, 10), np.uint8),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.0005,
            "PhysicalSizeY": 0.001,
            "PhysicalSizeXUnit": "mm",
            "PhysicalSizeYUnit": "mm",
        },
    )
    assert read_image(p)[1] == (0.5, 1.0)
    write_image(p, np.ones((10, 10), np.uint8), (0.5, 1), {"sample_id": "sample-1"})
    with tifffile.TiffFile(p) as t:
        assert "sample-1" in t.ome_metadata
    resized = resize_mosaic(p, 0.5)
    with tifffile.TiffFile(resized) as t:
        assert "sample-1" in t.ome_metadata


def test_missing_calibration_rejected(tmp_path):
    p = tmp_path / "uncalibrated.tif"
    tifffile.imwrite(p, np.ones((10, 10), np.uint8))
    with pytest.raises(ValueError, match="OME"):
        tiff_info(p)


def test_proximity_renumbering_and_local_work():
    import wood_stitch.adjacency as module

    labels = np.zeros((300, 300), np.uint32)
    labels[5:20, 5:10] = 1
    labels[10:13, 11:21] = 2
    swapped = np.where(labels == 1, 2, np.where(labels == 2, 1, 0)).astype(np.uint32)
    sizes = []
    real_dilate = module.binary_dilation

    def spy(image, **kw):
        sizes.append(image.shape)
        return real_dilate(image, **kw)

    with patch.object(module, "binary_dilation", spy):
        a = compute_adjacency(labels, radius_um=1.5, pixel_size_um=(0.5, 0.5))
        b = compute_adjacency(swapped, radius_um=1.5, pixel_size_um=(0.5, 0.5))
    assert a.to_dict("records") == b.to_dict("records") == [{"label_a": 1, "label_b": 2}]
    assert max(h * w for h, w in sizes) < 1000
    assert list(a.columns) == ["label_a", "label_b"]
    assert compute_adjacency(labels, radius_um=0.5, pixel_size_um=(0.5, 0.5)).empty


@pytest.mark.parametrize("value", [0, 128, 255])
def test_uniform_mask_is_empty(value):
    assert not make_tissue_mask(np.full((30, 30, 3), value, np.uint8)).any()


def test_segment_color_device_and_boundary_policy():
    class Model:
        def eval(self, image, **kwargs):
            assert image[10, 10].tolist() == [30, 20, 10]
            assert kwargs["channel_axis"] == -1 and "channels" not in kwargs
            a = np.zeros(image.shape[:2], np.uint32)
            a[5:15, 5:15] = 1
            a[0:3, 20:25] = 2
            return a, None, None

    image = np.full((32, 32, 3), [10, 20, 30], np.uint8)
    labels = segment(image, np.ones((32, 32), bool), model=Model())
    assert set(np.unique(labels)) == {0, 1}


def test_overlay_preserves_unlabeled_background():
    image = np.full((20, 20, 3), 200, np.uint8)
    result = make_overlay(image, np.zeros((20, 20), np.uint32), pd.DataFrame({"label": [], "cell_type": []}))
    np.testing.assert_array_equal(image, result)


def test_disconnected_tile_rejected_without_mosaic(tmp_path):
    rng = np.random.default_rng(4)
    base = rng.integers(0, 256, (600, 1000, 3), np.uint8)
    tiles = [base[:, :650], base[:, 350:], rng.integers(0, 256, (600, 650, 3), np.uint8)]
    paths = []
    for i, image in enumerate(tiles):
        p = tmp_path / f"{i}.tif"
        write_image(p, image, (0.5, 0.5))
        paths.append(p)
    output = tmp_path / "mosaic.ome.tif"
    with pytest.raises(ValueError, match="Incomplete mosaic"):
        stitch(None, output, paths=paths)
    assert not output.exists()
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert not coverage["complete"] and coverage["excluded"] == ["2.tif"]


def test_mixed_calibration_and_memory_guard(source, tmp_path):
    other = tmp_path / "different.tif"
    write_image(other, np.ones((64, 64, 3), np.uint8), (1, 1))
    with pytest.raises(ValueError, match="inconsistent"):
        stitch(None, tmp_path / "out.tif", paths=[source, other])
    with pytest.raises(MemoryError):
        stitch(None, tmp_path / "out.tif", paths=[source], max_input_gb=1e-9)


def test_summary_includes_ducts_and_empty_specimens():
    cells = pd.DataFrame(
        {"sample_id": ["x", "x", "x"], "cell_type": ["ray", "tracheid", "resin_duct"], "area_um2": [1, 2, 3]}
    )
    result = summarize(cells, [{"sample_id": "x"}, {"sample_id": "empty"}]).set_index("sample_id")
    assert result.loc["x", "ray_percent"] == pytest.approx(100 / 3)
    assert result.loc["empty", "total_cells"] == 0
    assert pd.isna(result.loc["empty", "ray_percent"])
