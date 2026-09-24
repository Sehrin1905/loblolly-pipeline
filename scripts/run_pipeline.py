# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "boto3",
#     "python-dotenv",
#     "opencv-python",
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "scipy",
#     "cellpose",
#     "Pillow",
# ]
# ///
"""
Full end-to-end pipeline: sync from R2, stitch, segment, classify, upload results.

Usage:
    python3 scripts/run_pipeline.py --config config.toml
"""
import sys
import os
import argparse
import logging
import tomllib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── R2 sync ──────────────────────────────────────────────────────────────────

def get_r2_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def list_r2_objects(client, bucket: str, prefix: str = "") -> dict[str, str]:
    """Returns {key: etag} for all objects under prefix."""
    paginator = client.get_paginator("list_objects_v2")
    objects = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = obj["ETag"].strip('"')
    return objects


def sync_r2_to_local(client, bucket: str, prefix: str, local_dir: Path) -> None:
    """
    Sync R2 objects under prefix to local_dir.
    Only downloads files that are missing or have changed (by ETag).
    """
    log.info(f"Syncing R2 s3://{bucket}/{prefix} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    remote = list_r2_objects(client, bucket, prefix)
    if not remote:
        log.warning(f"  No objects found under {prefix}")
        return

    for key, etag in remote.items():
        rel_path = key[len(prefix):].lstrip("/")
        # Security: prevent path traversal attacks
        local_path = (local_dir / rel_path).resolve()
        if not str(local_path).startswith(str(local_dir.resolve())):
            log.warning(f"  Skipping unsafe path: {rel_path}")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file needs updating
        needs_download = True
        if local_path.exists():
            # Compare ETag (MD5 for single-part uploads)
            import hashlib
            with open(local_path, "rb") as f:
                local_md5 = hashlib.md5(f.read()).hexdigest()
            if local_md5 == etag:
                needs_download = False

        if needs_download:
            log.info(f"  Downloading {key} ...")
            client.download_file(bucket, key, str(local_path))
        else:
            log.info(f"  Up to date: {rel_path}")


def upload_to_r2(client, bucket: str, local_path: Path, r2_key: str) -> None:
    """Upload a single file to R2."""
    log.info(f"  Uploading {local_path.name} → s3://{bucket}/{r2_key}")
    client.upload_file(str(local_path), bucket, r2_key)


def upload_sample_outputs(client, bucket: str, sample_name: str, sample_dir: Path, run_id: str) -> None:
    """Upload all pipeline outputs for a sample to R2."""
    output_files = [
        "mosaic.tif",
        "tissue_mask.png",
        "safranin.png",
        "astra_blue.png",
        "residual.png",
        "labels.npy",
        "features.csv",
        "adjacency.csv",
        "classified.csv",
        "overlay.png",
        "metadata.json",
    ]
    for filename in output_files:
        local_path = sample_dir / filename
        if local_path.exists():
            r2_key = f"outputs/{run_id}/{sample_name}/{filename}"
            upload_to_r2(client, bucket, local_path, r2_key)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_sample(sample_name: str, sample_dir: Path, cfg: dict) -> None:
    import cv2
    import numpy as np
    import pandas as pd
    from wood_stitch.stitch import stitch
    from wood_stitch.resize import resize_mosaic
    from wood_stitch.tissue_mask import make_tissue_mask
    from wood_stitch.deconvolve import deconvolve
    from wood_stitch.segment import segment
    from wood_stitch.features import compute_features
    from wood_stitch.adjacency import compute_adjacency
    from wood_stitch.classify import classify_cells
    from wood_stitch.visualize import make_overlay

    tile_dir = sample_dir / "tiles"
    mosaic_path = sample_dir / "mosaic.tif"

    log.info(f"\n{'='*60}")
    log.info(f"Processing {sample_name}")
    log.info(f"{'='*60}")

    # 1. Stitch
    if not mosaic_path.exists():
        log.info("Step 1: Stitching ...")
        stitch(str(tile_dir), str(mosaic_path))
    else:
        log.info("Step 1: Mosaic already exists, skipping stitch")

    # 2. Resize
    resize_factor = cfg["pipeline"].get("mosaic_resize_factor", 0.5)
    import json
    meta_path = sample_dir / "metadata.json"
    already_resized = False
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        already_resized = meta.get("resize_factor") is not None and meta.get("resize_factor") != 1.0

    if not already_resized and resize_factor != 1.0:
        log.info(f"Step 2: Resizing mosaic by {resize_factor} ...")
        resize_mosaic(str(mosaic_path), resize_factor)
    else:
        log.info("Step 2: Mosaic already resized, skipping")

    # Load mosaic
    img = cv2.imread(str(mosaic_path))

    # 3. Tissue mask
    tissue_mask_path = sample_dir / "tissue_mask.png"
    if not tissue_mask_path.exists():
        log.info("Step 3: Computing tissue mask ...")
        mask = make_tissue_mask(img)
        cv2.imwrite(str(tissue_mask_path), (mask * 255).astype(np.uint8))
    else:
        log.info("Step 3: Tissue mask already exists, skipping")
        mask = cv2.imread(str(tissue_mask_path), cv2.IMREAD_GRAYSCALE) > 127

    # 4. Deconvolution
    safranin_path = sample_dir / "safranin.png"
    if not safranin_path.exists():
        log.info("Step 4: Running color deconvolution ...")
        results = deconvolve(img)
        for key in ("safranin", "astra_blue", "residual"):
            channel = results[key]
            normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(str(sample_dir / f"{key}.png"), normalized.astype(np.uint8))
    else:
        log.info("Step 4: Deconvolution already done, skipping")

    # 5. Segmentation
    labels_path = sample_dir / "labels.npy"
    if not labels_path.exists():
        log.info("Step 5: Running segmentation ...")
        use_gpu = cfg["cellpose"].get("use_gpu", True)
        diameter = cfg["cellpose"].get("diameter", None)
        labels = segment(img, mask, gpu=use_gpu, diameter=diameter)
        np.save(str(labels_path), labels)
        log.info(f"  Found {labels.max()} cells")
    else:
        log.info("Step 5: Labels already exist, skipping")
        labels = np.load(str(labels_path))

    # 6. Features
    features_path = sample_dir / "features.csv"
    if not features_path.exists():
        log.info("Step 6: Computing features ...")
        df = compute_features(labels)
        df.to_csv(features_path, index=False)
    else:
        log.info("Step 6: Features already exist, skipping")
        df = pd.read_csv(features_path)

    # 7. Adjacency
    adjacency_path = sample_dir / "adjacency.csv"
    if not adjacency_path.exists():
        log.info("Step 7: Computing adjacency ...")
        edge_df = compute_adjacency(labels)
        edge_df.to_csv(adjacency_path, index=False)
    else:
        log.info("Step 7: Adjacency already exists, skipping")

    # 8. Classify
    classified_path = sample_dir / "classified.csv"
    log.info("Step 8: Classifying cells ...")
    df = classify_cells(df)
    df.to_csv(classified_path, index=False)
    ray_pct = (df["cell_type"] == "ray").mean() * 100
    log.info(f"  Rays: {(df['cell_type']=='ray').sum()} ({ray_pct:.1f}%)")
    log.info(f"  Tracheids: {(df['cell_type']=='tracheid').sum()}")

    # 9. Visualize
    overlay_path = sample_dir / "overlay.png"
    log.info("Step 9: Generating overlay ...")
    overlay = make_overlay(img, labels, df)
    cv2.imwrite(str(overlay_path), overlay)

    log.info(f"✅ {sample_name} complete!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.toml", help="Path to config TOML file")
    args = ap.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    cache_dir = Path(cfg["cache"]["local_cache_dir"])
    bucket = cfg["r2"]["bucket_name"]
    samples_to_run = cfg["pipeline"].get("samples", [])

    # Generate unique run ID
    from datetime import datetime, timezone
    import subprocess
    git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{git_rev}"
    log.info(f"Run ID: {run_id}")

    log.info("Loblolly Pipeline starting ...")
    log.info(f"Cache dir: {cache_dir}")
    log.info(f"R2 bucket: {bucket}")

    # Connect to R2
    client = get_r2_client()

    # List available samples from R2
    log.info("Listing samples in R2 ...")
    remote_objects = list_r2_objects(client, bucket, "raw/")
    available_samples = set()
    for key in remote_objects:
        parts = key.split("/")
        if len(parts) >= 2:
            available_samples.add(parts[1])

    if samples_to_run:
        samples = [s for s in samples_to_run if s in available_samples]
    else:
        samples = sorted(available_samples)

    log.info(f"Samples to process: {samples}")

    # Process each sample
    for sample_name in samples:
        sample_dir = cache_dir / sample_name

        # Sync tiles from R2
        log.info(f"\nSyncing {sample_name} from R2 ...")
        sync_r2_to_local(client, bucket, f"raw/{sample_name}", sample_dir / "tiles")

        # Run pipeline
        try:
            run_sample(sample_name, sample_dir, cfg)
        except Exception as e:
            log.error(f"❌ {sample_name} failed: {e}", exc_info=True)
            continue

        # Upload outputs to R2
        log.info(f"Uploading {sample_name} outputs to R2 ...")
        upload_sample_outputs(client, bucket, sample_name, sample_dir)

    log.info("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()