"""Upload calibrated source TIFFs without replacing conflicting raw objects."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path

from botocore.exceptions import ClientError
from wood_stitch.artifacts import sha256_file
from wood_stitch.image_io import tiff_info
from wood_stitch.pipeline import get_r2_client
from wood_stitch.storage import sample_id, is_sidecar, contained_path

EXCLUDE = {"Etc", "Image Registry", "REF 801 OLD - DON'T USE ", "System Volume Information"}


def source_files(directory):
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".tif", ".tiff")
        and not is_sidecar(p.relative_to(directory).as_posix())
    )


def remote_head(client, bucket, key):
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def upload_file(client, bucket, path, key):
    digest = sha256_file(path)
    size = path.stat().st_size
    head = remote_head(client, bucket, key)
    if head is not None:
        if head["ContentLength"] != size or head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError(f"Raw object conflict; use a new source version instead of overwriting: {key}")
        state = "skipped"
    else:
        client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        state = "uploaded"
    # Verify object bytes, not just the SHA field that this client set.
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    checksum, count = hashlib.sha256(), 0
    try:
        for block in iter(lambda: body.read(1024 * 1024), b""):
            checksum.update(block)
            count += len(block)
    finally:
        body.close()
    if count != size or checksum.hexdigest() != digest:
        raise OSError(f"Uploaded object byte verification failed: {key}")
    return {"key": key, "size": size, "sha256": digest, "status": state}


def upload_sample(client, bucket, name, directory, max_workers=4):
    name = sample_id(name.replace(" ", "_"))
    files = source_files(directory)
    if not files:
        raise ValueError(f"No TIFFs found: {directory}")
    for path in files:
        contained_path(directory, path.relative_to(directory).as_posix())
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Input symlink escapes the source directory: {path}")
        tiff_info(path)  # validate header and embedded X/Y calibration before any uploads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(
            executor.map(
                lambda p: upload_file(client, bucket, p, f"raw/{name}/{p.relative_to(directory).as_posix()}"),
                files,
            )
        )
    manifest = {
        "sample": name,
        "file_count": len(records),
        "total_bytes": sum(r["size"] for r in records),
        "files": records,
    }
    # Commit the manifest only when every selected object was verified. Failed attempts can be retried.
    payload = json.dumps(manifest, indent=2).encode()
    client.upload_fileobj(io.BytesIO(payload), bucket, f"manifests/{name}.json")
    print(f"{name}: verified {len(records)} TIFFs and published manifest")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--sample")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1 or not args.dataset_dir.is_dir():
        parser.error("Existing dataset directory and positive worker count required")
    if args.sample:
        directory = args.dataset_dir / sample_id(args.sample)
        if not directory.is_dir():
            parser.error(f"Sample directory not found: {directory}")
        dirs = [directory]
    else:
        dirs = sorted(
            p
            for p in args.dataset_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in EXCLUDE
        )
    ids = [sample_id(p.name.replace(" ", "_")) for p in dirs]
    if not dirs or len(ids) != len(set(ids)):
        parser.error("Empty dataset or colliding normalized sample IDs")
    client = get_r2_client()
    bucket = os.environ["R2_BUCKET"]
    for directory in dirs:
        upload_sample(client, bucket, directory.name, directory, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
