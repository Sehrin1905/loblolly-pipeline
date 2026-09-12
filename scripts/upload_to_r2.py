# /// script
# requires-python = ">=3.13"
# dependencies = ["boto3", "python-dotenv"]
# ///
"""
Upload raw tile data to Cloudflare R2.

Usage:
    python3 scripts/upload_to_r2.py <dataset_dir> [--sample SAMPLE_NAME]

Example:
    python3 scripts/upload_to_r2.py /Volumes/BRO
    python3 scripts/upload_to_r2.py /Volumes/BRO --sample "Marsh 1 Recent"
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# Folders to exclude from upload
EXCLUDE = {
    "Etc",
    "Image Registry",
    "REF 801 OLD - DON'T USE ",
    "System Volume Information",
}


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_remote_sha256(client, bucket: str, key: str) -> str | None:
    """Get SHA-256 from object metadata if it exists."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        return resp.get("Metadata", {}).get("sha256")
    except client.exceptions.ClientError:
        return None


def upload_file(client, bucket: str, local_path: Path, r2_key: str, 
                max_retries: int = 3) -> dict:
    """
    Upload a single file to R2 with SHA-256 verification and retries.
    Returns result dict with status, key, size, sha256.
    """
    import time

    local_sha256 = sha256_file(local_path)
    size = local_path.stat().st_size

    # Check if already uploaded with same hash
    remote_sha256 = get_remote_sha256(client, bucket, r2_key)
    if remote_sha256 == local_sha256:
        return {"status": "skipped", "key": r2_key, "size": size, "sha256": local_sha256}

    # Upload with retries
    for attempt in range(max_retries):
        try:
            client.upload_file(
                str(local_path),
                bucket,
                r2_key,
                ExtraArgs={"Metadata": {"sha256": local_sha256}},
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"  Retry {attempt + 1}/{max_retries} for {local_path.name} (waiting {wait}s)...")
                time.sleep(wait)
            else:
                return {"status": "failed", "key": r2_key, "size": size, 
                        "sha256": local_sha256, "error": str(e)}

    # Verify upload
    remote_sha256 = get_remote_sha256(client, bucket, r2_key)
    if remote_sha256 != local_sha256:
        return {"status": "failed", "key": r2_key, "size": size, "sha256": local_sha256,
                "error": "SHA-256 mismatch after upload"}

    return {"status": "uploaded", "key": r2_key, "size": size, "sha256": local_sha256}


def build_manifest(sample_name: str, files: list[dict]) -> dict:
    return {
        "sample": sample_name,
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "files": files,
    }


def upload_sample(client, bucket: str, sample_name: str,
                  sample_dir: Path, max_workers: int = 4) -> dict:
    """Upload all tiles for one sample to R2."""

    # Collect all tif files
    tif_files = sorted(sample_dir.glob("*.tif"))
    if not tif_files:
        print(f"  WARNING: No .tif files found in {sample_dir}")
        return {}

    # Sanitize sample name for R2 key (replace spaces with underscores)
    safe_name = sample_name.replace(" ", "_")

    print(f"\n{'='*60}")
    print(f"Sample: {sample_name}")
    print(f"Files: {len(tif_files)}")
    total_size = sum(f.stat().st_size for f in tif_files)
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"R2 prefix: raw/{safe_name}/")
    print(f"{'='*60}")

    # Test with first file
    print("\nTesting with first file...")
    test_file = tif_files[0]
    test_key = f"raw/{safe_name}/{test_file.name}"
    test_result = upload_file(client, bucket, test_file, test_key)
    print(f"  {test_result['status'].upper()}: {test_file.name}")

    # Verify test download
    if test_result["status"] in ("uploaded", "skipped"):
        print("  Verifying download...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        client.download_file(bucket, test_key, tmp_path)
        downloaded_sha256 = sha256_file(Path(tmp_path))
        Path(tmp_path).unlink()
        if downloaded_sha256 == test_result["sha256"]:
            print("  ✅ Download verification passed!")
        else:
            print("  ❌ Download verification FAILED - aborting upload")
            return {}

    # Upload remaining files with concurrency
    print(f"\nUploading {len(tif_files)} files...")
    results = [test_result]
    remaining = tif_files[1:]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                upload_file, client, bucket, f,
                f"raw/{safe_name}/{f.name}"
            ): f for f in remaining
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = result["status"].upper()
            print(f"  [{i}/{len(remaining)}] {status}: {Path(result['key']).name}")

    # Summary
    uploaded = sum(1 for r in results if r["status"] == "uploaded")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_bytes = sum(r["size"] for r in results if r["status"] in ("uploaded", "skipped"))

    print(f"\nSummary for {sample_name}:")
    print(f"  Uploaded: {uploaded}")
    print(f"  Skipped (already up to date): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total bytes: {total_bytes / 1e9:.2f} GB")

    # Save manifest
    manifest = build_manifest(safe_name, results)
    manifest_key = f"manifests/{safe_name}.json"
    manifest_json = json.dumps(manifest, indent=2)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(manifest_json)
        tmp_path = tmp.name
    client.upload_file(tmp_path, bucket, manifest_key)
    Path(tmp_path).unlink()
    print(f"  Manifest saved → R2:{manifest_key}")

    if failed > 0:
        failed_files = [r["key"] for r in results if r["status"] == "failed"]
        print(f"\n  ❌ Failed files:")
        for f in failed_files:
            print(f"    {f}")
        print(f"  To retry: python3 scripts/upload_to_r2.py <dataset_dir> --sample '{sample_name}'")

    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", help="Root directory containing sample folders (e.g. /Volumes/BRO)")
    ap.add_argument("--sample", help="Upload only this sample (folder name on flash drive)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent upload threads")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} does not exist")
        sys.exit(1)

    client = get_r2_client()
    bucket = os.environ["R2_BUCKET"]

    print(f"Cloudflare R2 Upload")
    print(f"Bucket: {bucket}")
    print(f"Dataset: {dataset_dir}")

    if args.sample:
        # Upload single sample
        sample_dir = dataset_dir / args.sample
        if not sample_dir.exists():
            print(f"ERROR: {sample_dir} does not exist")
            sys.exit(1)
        upload_sample(client, bucket, args.sample, sample_dir, args.workers)
    else:
        # Upload all samples excluding unwanted folders
        sample_dirs = [d for d in sorted(dataset_dir.iterdir())
                      if d.is_dir()
                      and not d.name.startswith(".")
                      and d.name not in EXCLUDE]

        print(f"\nFound {len(sample_dirs)} sample folders:")
        total_files = 0
        total_size = 0
        for d in sample_dirs:
            tifs = list(d.glob("*.tif"))
            size = sum(f.stat().st_size for f in tifs)
            total_files += len(tifs)
            total_size += size
            print(f"  {d.name}: {len(tifs)} tiles ({size/1e9:.2f} GB)")

        print(f"\nTotal: {total_files} files, {total_size/1e9:.2f} GB")
        print("\nProceed? (y/n): ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

        for d in sample_dirs:
            upload_sample(client, bucket, d.name, d, args.workers)

    print("\n✅ Upload complete!")


if __name__ == "__main__":
    main()