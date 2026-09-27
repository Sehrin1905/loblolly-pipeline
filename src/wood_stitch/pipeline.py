"""Manifest-verified runs, calibrated artifacts, and explicit completion records."""

import argparse
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from uuid import uuid4

import cv2
from filelock import FileLock
import numpy as np
import pandas as pd

from .analysis import summarize
from .artifacts import fingerprint, sha256_file, write_json, write_csv, write_preview
from .image_io import read_image, write_image, tiff_info
from .storage import (
    sample_id,
    contained_path,
    list_objects,
    load_manifest,
    check_remote_inventory,
    sync_inputs,
    upload_artifact,
)
from .stitch import stitch
from .resize import resize_mosaic
from .features import compute_features
from .adjacency import compute_adjacency
from .classify import classify_cells
from .tissue_mask import make_tissue_mask
from .visualize import make_overlay
from .segment import create_model, segment

log = logging.getLogger(__name__)


def get_r2_client():
    import boto3
    from dotenv import load_dotenv

    load_dotenv()
    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def validate_config(cfg):
    allowed = {
        "cache": {"local_cache_dir"},
        "r2": {"bucket_name"},
        "pipeline": {
            "samples",
            "mosaic_resize_factor",
            "max_input_gb",
            "max_analysis_pixels",
            "adjacency_radius_um",
            "sample_registry",
        },
        "cellpose": {"device", "model", "diameter"},
    }
    for section, values in cfg.items():
        if section not in allowed or not isinstance(values, dict) or set(values) - allowed[section]:
            raise ValueError(f"Unknown configuration in {section}; migrate using config.toml")
    if not cfg.get("r2", {}).get("bucket_name") or not cfg.get("cache", {}).get("local_cache_dir"):
        raise ValueError("Bucket and local cache directory are required")
    pipeline = cfg.setdefault("pipeline", {})
    pipeline.setdefault("samples", [])
    pipeline.setdefault("mosaic_resize_factor", 0.5)
    pipeline.setdefault("max_input_gb", 2.0)
    pipeline.setdefault("max_analysis_pixels", 25_000_000)
    if not isinstance(pipeline["samples"], list) or len(set(pipeline["samples"])) != len(pipeline["samples"]):
        raise ValueError("Samples must be a list of unique IDs")
    for name in pipeline["samples"]:
        sample_id(name)
    for key in ("mosaic_resize_factor", "max_input_gb", "max_analysis_pixels", "adjacency_radius_um"):
        if key in pipeline and (
            isinstance(pipeline[key], bool) or not math.isfinite(pipeline[key]) or pipeline[key] <= 0
        ):
            raise ValueError(f"{key} must be positive and finite")
    if pipeline["mosaic_resize_factor"] > 1:
        raise ValueError("Analysis resize factor cannot exceed 1")
    settings = cfg.setdefault("cellpose", {})
    settings.setdefault("device", "auto")
    settings.setdefault("model", "cpsam_v2")
    if settings["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("Device must be auto, cpu, cuda, or mps")
    if "diameter" in settings and (not math.isfinite(settings["diameter"]) or settings["diameter"] <= 0):
        raise ValueError("Diameter must be positive and finite")
    return cfg


def code_identity():
    package = Path(__file__).parent
    files = {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    versions = {}
    for name in (
        "loblolly-pipeline",
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "tifffile",
        "cellpose",
        "torch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    repo = package.parent.parent
    revision = None
    if (repo / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return {"source_sha256": fingerprint(files), "git_revision": revision, "versions": versions}


def read_registry(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    entries = data.get("samples", [])
    result = {}
    for row in entries:
        name = sample_id(row["sample_id"])
        if (
            name in result
            or row.get("site") not in ("Marsh", "Reference")
            or row.get("period") not in ("Old", "Recent")
        ):
            raise ValueError(f"Invalid/duplicate sample metadata: {name}")
        if not isinstance(row.get("tree_id"), str) or not row["tree_id"].strip():
            raise ValueError(f"Missing tree ID for {name}")
        result[name] = {k: row[k] for k in ("sample_id", "site", "period", "tree_id")}
    if not result:
        raise ValueError("Sample registry is empty")
    return result


def cached_result(directory, identity):
    marker = directory / "complete.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text())
        if value["identity"] != identity:
            return None
        for entry in value["artifacts"]:
            path = contained_path(directory, entry["name"])
            if (
                not path.is_file()
                or path.stat().st_size != entry["size"]
                or sha256_file(path) != entry["sha256"]
            ):
                return None
        return value
    except (ValueError, KeyError, OSError, TypeError):
        return None


def run_sample(name, directory, cfg, *, paths, model, provenance):
    """Produce a complete artifact set. A marker is committed only after all outputs validate."""
    directory = Path(directory)
    identity = fingerprint(provenance)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".lock")):
        cached = cached_result(directory, identity)
        if cached:
            log.info("%s: reusing verified artifacts %s", name, identity[:12])
            return cached
        (directory / "complete.json").unlink(missing_ok=True)
        produced = []
        timings = {}

        def timed(stage, fn):
            start = time.perf_counter()
            value = fn()
            timings[stage] = time.perf_counter() - start
            return value

        annotations = {
            "sample_id": name,
            "artifact_id": identity,
            "sample": provenance["sample"],
            "code": provenance["code"],
            "input_inventory_sha256": fingerprint(provenance["inputs"]),
            "processing": provenance["settings"],
            "model": provenance["model"],
        }
        mosaic = directory / "mosaic.ome.tif"
        timed(
            "stitch",
            lambda: stitch(
                None,
                mosaic,
                paths=paths,
                max_input_gb=cfg["pipeline"]["max_input_gb"],
                provenance=annotations,
            ),
        )
        produced += ["mosaic.ome.tif", "coverage.json"]
        header = tiff_info(mosaic)
        factor = cfg["pipeline"]["mosaic_resize_factor"]
        height, width = header["shape"][:2]
        if (
            max(1, round(height * factor)) * max(1, round(width * factor))
            > cfg["pipeline"]["max_analysis_pixels"]
        ):
            raise MemoryError(
                "Requested analysis dimensions exceed max_analysis_pixels; profile before raising it"
            )
        analysis = timed(
            "resize",
            lambda: resize_mosaic(
                mosaic,
                cfg["pipeline"]["mosaic_resize_factor"],
                directory / "analysis.ome.tif",
                provenance=annotations,
            ),
        )
        produced.append("analysis.ome.tif")
        image, scale = read_image(analysis)
        if image.shape[0] * image.shape[1] > cfg["pipeline"]["max_analysis_pixels"]:
            raise MemoryError(
                "Analysis image exceeds max_analysis_pixels; use a smaller pilot or profile before raising it"
            )
        mask = timed("tissue_mask", lambda: make_tissue_mask(image))
        if not mask.any():
            raise ValueError("No tissue detected; inspect the input and mask settings")
        write_image(directory / "tissue-mask.ome.tif", mask.astype(np.uint8), scale, annotations)
        labels = timed(
            "segment", lambda: segment(image, mask, diameter=cfg["cellpose"].get("diameter"), model=model)
        )
        write_image(directory / "labels.ome.tif", labels, scale, annotations)
        produced += ["tissue-mask.ome.tif", "labels.ome.tif"]
        cells = timed("features", lambda: compute_features(labels, pixel_size_um=scale))
        cells = classify_cells(cells)
        cells["sample_id"] = name
        cells["artifact_id"] = identity
        for key, value in provenance["sample"].items():
            cells[key] = value
        write_csv(directory / "cells.csv", cells)
        produced.append("cells.csv")
        if "adjacency_radius_um" in cfg["pipeline"]:
            edges = timed(
                "proximity",
                lambda: compute_adjacency(
                    labels, radius_um=cfg["pipeline"]["adjacency_radius_um"], pixel_size_um=scale
                ),
            )
            write_csv(directory / "proximity.csv", edges)
            produced.append("proximity.csv")
        overlay = timed("overlay", lambda: make_overlay(image, labels, cells))
        preview_scale = min(1, 2048 / max(overlay.shape[:2]))
        if preview_scale < 1:
            overlay = cv2.resize(
                overlay, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA
            )
        write_preview(directory / "overlay.jpg", overlay)
        produced.append("overlay.jpg")
        write_json(
            directory / "qc.json",
            {
                "sample_id": name,
                "cells": len(cells),
                "tissue_area_um2": float(mask.sum() * scale[0] * scale[1]),
                "timings_seconds": timings,
                "segmentation_validation": "not biologically validated",
                "classification_validation": "unvalidated area/eccentricity rules",
                "wall_thickness": "not measured",
                "boundary_policy": "exclude labels crossing tissue/image edges",
            },
        )
        produced.append("qc.json")
        result = {
            "identity": identity,
            "provenance": provenance,
            "artifacts": [
                {
                    "name": name,
                    "size": (directory / name).stat().st_size,
                    "sha256": sha256_file(directory / name),
                }
                for name in produced
            ],
        }
        write_json(directory / "complete.json", result)
        return result


def upload_sample_outputs(client, bucket, name, directory, run_id, result, run_dir):
    records = []
    for entry in result["artifacts"]:
        path = contained_path(directory, entry["name"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Artifact changed before upload: {entry['name']}")
        if entry["name"] == "cells.csv":
            cells = pd.read_csv(path)
            cells["run_id"] = run_id
            path = contained_path(run_dir, f"samples/{name}/cells.csv")
            write_csv(path, cells)
        key = f"runs/{run_id}/samples/{name}/{entry['name']}"
        records.append(upload_artifact(client, bucket, path, key))
    return records


def execute(cfg, *, client=None, preflight=False):
    cfg = validate_config(cfg)
    client = client if client is not None else get_r2_client()
    bucket = cfg["r2"]["bucket_name"]
    registry = read_registry(cfg["pipeline"].get("sample_registry"))
    remote = list_objects(client, bucket, "manifests/")
    available = {sample_id(k[len("manifests/") : -5]) for k in remote if k.endswith(".json")}
    samples = cfg["pipeline"]["samples"] or sorted(available)
    if not samples or set(samples) - available:
        raise ValueError(f"No samples or unknown requested samples: {sorted(set(samples) - available)}")
    if registry and set(samples) - registry.keys():
        raise ValueError(f"Samples missing registry metadata: {sorted(set(samples) - registry.keys())}")
    manifests = {name: load_manifest(client, bucket, name) for name in samples}
    for manifest in manifests.values():
        check_remote_inventory(client, bucket, manifest)
    if preflight:
        log.info(
            "Preflight passed: %d samples, %d images, %d excluded metadata objects",
            len(samples),
            sum(len(m["files"]) for m in manifests.values()),
            sum(len(m["excluded"]) for m in manifests.values()),
        )
        return 0
    cache = Path(cfg["cache"]["local_cache_dir"]).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    code = code_identity()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    run_dir = contained_path(cache / "runs", run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {
        "run_id": run_id,
        "state": "running",
        "samples": {name: {"state": "pending"} for name in samples},
    }
    record = {
        "run_id": run_id,
        "code": code,
        "settings": cfg,
        "input_manifests": manifests,
        "sample_registry": registry,
    }

    def publish_json(filename, value):
        path = run_dir / filename
        write_json(path, value)
        upload_artifact(client, bucket, path, f"runs/{run_id}/{filename}")

    publish_json("run.json", record)
    publish_json("status.json", status)
    model = model_info = None
    for name in samples:
        try:
            status["samples"][name] = {"state": "running"}
            publish_json("status.json", status)
            if model is None:
                model, model_info = create_model(cfg["cellpose"])
                record["resolved_model"] = model_info
                publish_json("run.json", record)
            manifest = manifests[name]
            inputs = contained_path(cache / "inputs", f"{name}/{fingerprint(manifest)}/tiles")
            paths = sync_inputs(client, bucket, manifest, inputs)
            provenance = {
                "schema": 2,
                "inputs": manifest,
                "code": code,
                "model": model_info,
                "sample": registry.get(name, {"sample_id": name}),
                "settings": {
                    "pipeline": {
                        k: v for k, v in cfg["pipeline"].items() if k not in ("samples", "sample_registry")
                    },
                    "cellpose": cfg["cellpose"],
                },
            }
            identity = fingerprint(provenance)
            output = contained_path(cache / "artifacts", f"{name}/{identity}")
            result = run_sample(name, output, cfg, paths=paths, model=model, provenance=provenance)
            artifacts = upload_sample_outputs(client, bucket, name, output, run_id, result, run_dir)
            status["samples"][name] = {"state": "succeeded", "artifact_id": identity, "artifacts": artifacts}
        except Exception as exc:
            log.exception("%s failed", name)
            status["samples"][name] = {"state": "failed", "error": str(exc)}
        publish_json("status.json", status)
    failed = any(s["state"] != "succeeded" for s in status["samples"].values())
    status["state"] = "failed" if failed else "succeeded"
    if not failed:
        # Publish run completion only after aggregation/upload succeeds too.
        try:
            frames = [pd.read_csv(contained_path(run_dir, f"samples/{name}/cells.csv")) for name in samples]
            cells = pd.concat(frames, ignore_index=True)
            write_csv(run_dir / "all_samples.csv", cells)
            metadata = [registry.get(name, {"sample_id": name}) for name in samples]
            write_csv(run_dir / "sample_summary.csv", summarize(cells, metadata))
            upload_artifact(
                client, bucket, run_dir / "sample_summary.csv", f"runs/{run_id}/sample_summary.csv"
            )
            upload_artifact(client, bucket, run_dir / "all_samples.csv", f"runs/{run_id}/all_samples.csv")
        except Exception as exc:
            status["state"] = "failed"
            status["aggregation_error"] = str(exc)
            failed = True
            log.exception("Run aggregation failed")
    publish_json("status.json", status)
    log.info("Run %s: %s. Local record: %s", run_id, status["state"], run_dir)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate remote manifests/objects without downloading images or running a model",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        with open(args.config, "rb") as stream:
            cfg = tomllib.load(stream)
        registry = cfg.get("pipeline", {}).get("sample_registry")
        if registry:
            cfg["pipeline"]["sample_registry"] = str(
                (Path(args.config).resolve().parent / registry).resolve()
            )
        return execute(cfg, preflight=args.preflight)
    except Exception:
        log.exception("Pipeline failed")
        return 1
