"""Manifest-based R2 transfers. ETags are never treated as content hashes."""

import json
import re
from pathlib import Path, PurePosixPath

from .artifacts import atomic_path, sha256_file


def sample_id(value):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or value != value.strip()
        or any(c in value for c in "/\\:")
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"Unsafe sample ID: {value!r}")
    return value


def contained_path(root, relative):
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or ":" in relative
        or any(x in ("", ".", "..") for x in relative.split("/"))
        or any(ord(c) < 32 or ord(c) == 127 for c in relative)
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    root = Path(root).resolve()
    dest = (root / relative).resolve()
    if dest == root or not dest.is_relative_to(root):
        raise ValueError(f"Path escapes cache: {relative!r}")
    return dest


def list_objects(client, bucket, prefix):
    return {
        obj["Key"]: obj
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    }


def is_sidecar(key):
    return any(part.startswith("._") or part == "__MACOSX" for part in PurePosixPath(key).parts)


def load_manifest(client, bucket, name):
    sample_id(name)
    response = client.get_object(Bucket=bucket, Key=f"manifests/{name}.json")
    body = response["Body"]
    try:
        raw = body.read(16 * 1024 * 1024 + 1)
    finally:
        body.close()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Manifest is too large")
    manifest = json.loads(raw)
    return validate_manifest(manifest, name)


def validate_manifest(manifest, name):
    sample_id(name)
    prefix = f"raw/{name}/"
    if manifest.get("sample") != name or not isinstance(manifest.get("files"), list):
        raise ValueError(f"Invalid manifest for {name}")
    files, excluded, seen = [], [], set()
    for entry in manifest["files"]:
        key = entry.get("key", "")
        if not key.startswith(prefix) or key in seen:
            raise ValueError(f"Unexpected or duplicate manifest key: {key!r}")
        relative = key[len(prefix) :]
        contained_path(Path.cwd(), relative)  # lexical validation, including traversal
        seen.add(key)
        if is_sidecar(relative):
            excluded.append({"key": key, "reason": "AppleDouble/macOS metadata, not an image"})
            continue
        if Path(relative).suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Manifest contains a non-TIFF input: {key}")
        if entry.get("status", "uploaded") not in ("uploaded", "skipped"):
            raise ValueError(f"Incomplete upload in manifest: {key}")
        digest = entry.get("sha256", "")
        size = entry.get("size")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or type(size) is not int or size <= 0:
            raise ValueError(f"Invalid SHA-256/size in manifest: {key}")
        files.append({"key": key, "size": size, "sha256": digest})
    if not files:
        raise ValueError(f"No image inputs in manifest for {name}")
    return {"sample": name, "files": sorted(files, key=lambda x: x["key"]), "excluded": excluded}


def check_remote_inventory(client, bucket, manifest):
    prefix = f"raw/{manifest['sample']}/"
    remote = list_objects(client, bucket, prefix)
    expected = {e["key"] for e in manifest["files"]}
    actual = {k for k in remote if not k.endswith("/") and not is_sidecar(k)}
    if actual != expected:
        raise ValueError(
            f"Raw inventory differs from manifest; missing={sorted(expected - actual)}, "
            f"unmanifested={sorted(actual - expected)}"
        )
    for entry in manifest["files"]:
        head = client.head_object(Bucket=bucket, Key=entry["key"])
        if (
            head["ContentLength"] != entry["size"]
            or head.get("Metadata", {}).get("sha256") != entry["sha256"]
        ):
            raise ValueError(f"Remote object differs from manifest: {entry['key']}")


def sync_inputs(client, bucket, manifest, local_dir):
    """Verify bytes against the manifest; atomically replace only a corrupt/missing cache file."""
    prefix = f"raw/{manifest['sample']}/"
    destinations = [
        (entry, contained_path(local_dir, entry["key"][len(prefix) :])) for entry in manifest["files"]
    ]
    check_remote_inventory(client, bucket, manifest)
    for entry, path in destinations:
        if path.is_file() and path.stat().st_size == entry["size"] and sha256_file(path) == entry["sha256"]:
            continue
        with atomic_path(path) as temp:
            client.download_file(bucket, entry["key"], str(temp))
            if temp.stat().st_size != entry["size"] or sha256_file(temp) != entry["sha256"]:
                raise ValueError(f"Downloaded bytes failed SHA-256 verification: {entry['key']}")
    return [path for _, path in destinations]


def upload_artifact(client, bucket, path, key):
    digest = sha256_file(path)
    client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
    head = client.head_object(Bucket=bucket, Key=key)
    if head["ContentLength"] != Path(path).stat().st_size or head.get("Metadata", {}).get("sha256") != digest:
        raise OSError(f"Uploaded object size/metadata verification failed: {key}")
    return {"key": key, "sha256": digest, "size": Path(path).stat().st_size}
