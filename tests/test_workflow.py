import json
from pathlib import Path
import subprocess
import pandas as pd
import pytest
from wood_stitch import pipeline
from wood_stitch.artifacts import sha256_file
from wood_stitch.storage import contained_path, sample_id, load_manifest, validate_manifest, sync_inputs


@pytest.mark.parametrize(
    "relative", ["../tiles-other/file", "..", "./a", "a/../b", "/tmp/file", "a//b", "a\\b", "a:bad", "x\x00y"]
)
def test_unsafe_paths_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        contained_path(tmp_path / "tiles", relative)


def test_symlink_escape(tmp_path):
    (tmp_path / "tiles").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "tiles" / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError):
        contained_path(tmp_path / "tiles", "link/file.tif")


@pytest.mark.parametrize("name", ["..", ".", "/absolute", "a/b", "a\\b", "a\n"])
def test_invalid_sample_ids(name):
    with pytest.raises(ValueError):
        sample_id(name)


def test_hash_cache_and_exact_prefix(storage, tmp_path):
    storage.add("raw/REF_636_RECENT_extra/b.tif", b"unrelated")
    manifest = load_manifest(storage, "b", "REF_636_RECENT")
    first = sync_inputs(storage, "b", manifest, tmp_path / "tiles")
    assert len(storage.downloads) == 1
    sync_inputs(storage, "b", manifest, tmp_path / "tiles")
    assert len(storage.downloads) == 1  # multipart ETag is irrelevant
    first[0].write_bytes(b"corrupt")
    sync_inputs(storage, "b", manifest, tmp_path / "tiles")
    assert len(storage.downloads) == 2
    assert sha256_file(first[0]) == manifest["files"][0]["sha256"]


def test_corrupt_download_does_not_replace_previous_file(storage, tmp_path, monkeypatch):
    manifest = load_manifest(storage, "b", "REF_636_RECENT")
    dest = tmp_path / "a.tif"
    dest.write_bytes(b"previous")
    monkeypatch.setattr(storage, "download_file", lambda b, k, p: Path(p).write_bytes(b"wrong"))
    with pytest.raises(ValueError, match="SHA-256"):
        sync_inputs(storage, "b", manifest, tmp_path)
    assert dest.read_bytes() == b"previous"


def test_manifest_sidecars_excluded_and_failed_upload_rejected(storage):
    raw = json.loads(storage.objects["manifests/REF_636_RECENT.json"])
    raw["files"].append({"key": "raw/REF_636_RECENT/._a.tif", "size": 4096})
    good = validate_manifest(raw, "REF_636_RECENT")
    assert len(good["files"]) == 1 and len(good["excluded"]) == 1
    raw["files"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="Incomplete"):
        validate_manifest(raw, "REF_636_RECENT")


def test_unmanifested_image_fails(storage, tmp_path):
    manifest = load_manifest(storage, "b", "REF_636_RECENT")
    storage.add("raw/REF_636_RECENT/new.tif", b"new")
    with pytest.raises(ValueError, match="unmanifested"):
        sync_inputs(storage, "b", manifest, tmp_path)


def run_statuses(storage):
    return [
        json.loads(v)
        for k, v in storage.objects.items()
        if k.startswith("runs/") and k.endswith("/status.json")
    ]


def test_real_runner_upload_units_and_cache(storage, cfg, fake_model):
    assert pipeline.execute(cfg, client=storage) == 0
    state = run_statuses(storage)[0]
    assert state["state"] == "succeeded"
    cells_key = next(k for k in storage.objects if k.startswith("runs/") and k.endswith("/cells.csv"))
    import io

    cells = pd.read_csv(io.BytesIO(storage.objects[cells_key]))
    assert cells.iloc[0].area_um2 == 100  # 400 pixels at 0.5 µm/px
    assert cells.iloc[0].run_id == state["run_id"]
    assert not any(k.startswith("outputs/") or k.startswith("samples/") for k in storage.uploads)
    original_output = next((Path(cfg["cache"]["local_cache_dir"]) / "artifacts").rglob("cells.csv"))
    count = len(storage.downloads)
    assert pipeline.execute(cfg, client=storage) == 0
    assert len(fake_model) == 1 and len(storage.downloads) == count
    # Parameter changes create a different cache identity and trigger real processing.
    cfg["cellpose"]["diameter"] = 20
    assert pipeline.execute(cfg, client=storage) == 0
    assert len(fake_model) == 2
    # Corrupting an artifact invalidates its completion marker.
    original_output.write_text("corrupt")
    cfg["cellpose"].pop("diameter")
    assert pipeline.execute(cfg, client=storage) == 0
    assert len(fake_model) == 3


def test_failed_upload_never_marks_success(storage, cfg, fake_model):
    storage.fail_upload_suffix = "/labels.ome.tif"
    assert pipeline.execute(cfg, client=storage) == 1
    state = run_statuses(storage)[0]
    assert state["state"] == "failed"
    assert state["samples"]["REF_636_RECENT"]["state"] == "failed"
    assert not any(k.endswith("/all_samples.csv") for k in storage.uploads)


def test_processing_failure_returns_failure(storage, cfg, monkeypatch):
    def fail(settings):
        raise RuntimeError("test device unavailable")

    monkeypatch.setattr(pipeline, "create_model", fail)
    assert pipeline.execute(cfg, client=storage) == 1
    assert run_statuses(storage)[0]["state"] == "failed"


def test_unknown_requested_sample_and_empty_bucket_fail(storage, cfg):
    cfg["pipeline"]["samples"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown"):
        pipeline.execute(cfg, client=storage)
    cfg["pipeline"]["samples"] = []
    storage.objects.clear()
    with pytest.raises(ValueError, match="No samples"):
        pipeline.execute(cfg, client=storage)


def test_preflight_no_downloads_or_uploads(storage, cfg):
    assert pipeline.execute(cfg, client=storage, preflight=True) == 0
    assert not storage.downloads and not storage.uploads


def test_changed_remote_tile_invalidates_manifest(storage, cfg):
    storage.add("raw/REF_636_RECENT/a.tif", b"changed")
    with pytest.raises(ValueError, match="differs from manifest"):
        pipeline.execute(cfg, client=storage, preflight=True)


def test_unknown_registry_label_rejected(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"samples": [{"sample_id": "x", "site": "unknown", "period": "Old", "tree_id": "1"}]})
    )
    with pytest.raises(ValueError):
        pipeline.read_registry(path)


@pytest.mark.parametrize("failure", [False, True])
def test_slurm_exit_status_and_paths_with_spaces(tmp_path, failure):
    repo = Path(__file__).resolve().parents[1]
    directory = tmp_path / "path with spaces"
    directory.mkdir()
    code = 17 if failure else 0
    shell = f'uv() {{ return {code}; }}; export SLURM_SUBMIT_DIR="$PWD"; source "$1"'
    result = subprocess.run(
        ["bash", "-c", shell, "test", str(repo / "scripts/submit_pegasus.sh")],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code
    assert "EOF" not in result.stderr
