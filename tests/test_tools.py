import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from botocore.exceptions import ClientError
import numpy as np
import pandas as pd
import pytest

from wood_stitch import pipeline
from wood_stitch.deconvolve import build_unmixing_matrix

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uploader_conflicts_errors_and_real_byte_verification(storage, source, tmp_path, monkeypatch):
    uploader = script("upload_to_r2")
    key = "raw/REF_636_RECENT/a.tif"
    assert uploader.upload_file(storage, "b", source, key)["status"] == "skipped"
    before = storage.objects[key]
    modified = tmp_path / "changed.tif"
    modified.write_bytes(b"different")
    with pytest.raises(ValueError, match="conflict"):
        uploader.upload_file(storage, "b", modified, key)
    assert storage.objects[key] == before

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")

    monkeypatch.setattr(storage, "head_object", denied)
    with pytest.raises(ClientError):
        uploader.upload_file(storage, "b", source, key)


def test_uploader_rejects_corrupted_remote_bytes_even_with_matching_metadata(storage, source):
    uploader = script("upload_to_r2")
    key = "raw/REF_636_RECENT/a.tif"
    storage.objects[key] = b"x" * len(storage.objects[key])
    with pytest.raises(OSError, match="byte verification"):
        uploader.upload_file(storage, "b", source, key)


def test_uploader_nested_case_insensitive_and_sidecars(storage, source, tmp_path):
    uploader = script("upload_to_r2")
    directory = tmp_path / "dataset"
    directory.mkdir()
    nested = directory / "nested"
    nested.mkdir()
    (nested / "tile.TIFF").write_bytes(source.read_bytes())
    (directory / "._tile.tif").write_bytes(b"AppleDouble")
    assert len(uploader.source_files(directory)) == 1
    manifest = uploader.upload_sample(storage, "b", "new sample", directory)
    assert manifest["files"][0]["key"] == "raw/new_sample/nested/tile.TIFF"
    assert manifest["files"][0]["status"] == "uploaded"


def test_no_manifest_after_upload_failure(storage, source, tmp_path):
    uploader = script("upload_to_r2")
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "a.tif").write_bytes(source.read_bytes())
    storage.fail_upload_suffix = "/a.tif"
    with pytest.raises(OSError):
        uploader.upload_sample(storage, "b", "new", directory)
    assert "manifests/new.json" not in storage.objects


def test_cli_help_uses_project_dependencies():
    for path in (ROOT / "scripts").glob("*.py"):
        result = subprocess.run(
            [sys.executable, str(path), "--help"], cwd=ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0, (path.name, result.stderr)


def test_degenerate_stains_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        build_unmixing_matrix(np.array([[1, 0, 0], [1, 0, 0]], float))


def test_analysis_budget_before_resize(storage, cfg, fake_model, monkeypatch):
    cfg["pipeline"]["max_analysis_pixels"] = 10

    def forbidden(*a, **k):
        raise AssertionError("Oversized analysis must fail before resize allocation")

    monkeypatch.setattr(pipeline, "resize_mosaic", forbidden)
    assert pipeline.execute(cfg, client=storage) == 1
    states = [json.loads(v) for k, v in storage.objects.items() if k.endswith("/status.json")]
    assert "max_analysis_pixels" in states[0]["samples"]["REF_636_RECENT"]["error"]


def test_model_resolution_fail_closed_and_float32(monkeypatch, tmp_path):
    # No Cellpose weights needed: assert the adapter's device/model contract at its boundary.
    import types
    from wood_stitch.segment import create_model

    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    seen = {}

    def constructor(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(pretrained_model=str(weights), device=kwargs["device"])

    models = types.SimpleNamespace(MODEL_NAMES=["cpsam_v2"], CellposeModel=constructor)
    torch = types.SimpleNamespace(
        __version__="test",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        device=lambda x: x,
    )
    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(models=models))
    monkeypatch.setitem(sys.modules, "torch", torch)
    _, info = create_model({"device": "auto", "model": "cpsam_v2"})
    assert seen["use_bfloat16"] is False and seen["device"] == "cpu"
    assert info["weights_sha256"]
    with pytest.raises(ValueError, match="Unknown"):
        create_model({"device": "cpu", "model": "typo"})
    with pytest.raises(RuntimeError, match="CUDA"):
        create_model({"device": "cuda"})


def test_figure_script_completed_reviewed_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"state": "succeeded"}))
    pd.DataFrame(
        [
            {
                "sample_id": "x",
                "tree_id": "tree-1",
                "site": "Reference",
                "period": "Recent",
                "total_cells": 3,
                "ray_percent": 100 / 3,
                "tracheid_percent": 100 / 3,
                "resin_duct_percent": 100 / 3,
                "mean_lumen_area_um2": 25,
            }
        ]
    ).to_csv(run / "sample_summary.csv", index=False)
    figures = script("make_figures")
    assert figures.main([str(run)]) == 0
    assert len(list((run / "figures").glob("*.png"))) == 2
    (run / "status.json").write_text(json.dumps({"state": "failed"}))
    with pytest.raises(ValueError, match="completed"):
        figures.main([str(run)])
