import hashlib
import io
import json
from pathlib import Path
from botocore.exceptions import ClientError
import numpy as np
import pytest
from wood_stitch import pipeline
from wood_stitch.image_io import write_image


class FakeR2:
    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.downloads = []
        self.uploads = []
        self.fail_upload_suffix = None

    def add(self, key, data):
        self.objects[key] = data
        self.metadata[key] = {"sha256": hashlib.sha256(data).hexdigest()}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        return [
            {
                "Contents": [
                    {"Key": k, "Size": len(v), "ETag": '"multipart-5"'}
                    for k, v in self.objects.items()
                    if k.startswith(Prefix)
                ]
            }
        ]

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key]), "Metadata": self.metadata[Key]}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, filename):
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        if self.fail_upload_suffix and key.endswith(self.fail_upload_suffix):
            raise OSError("Injected upload failure")
        self.uploads.append(key)
        self.objects[key] = Path(filename).read_bytes()
        self.metadata[key] = (ExtraArgs or {}).get("Metadata", {})

    def upload_fileobj(self, stream, bucket, key):
        self.add(key, stream.read())


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.ome.tif"
    write_image(path, np.full((64, 64, 3), 150, np.uint8), (0.5, 0.5))
    return path


@pytest.fixture
def storage(source):
    client = FakeR2()
    key = "raw/REF_636_RECENT/a.tif"
    data = source.read_bytes()
    client.add(key, data)
    manifest = {
        "sample": "REF_636_RECENT",
        "files": [
            {"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "status": "uploaded"}
        ],
    }
    client.add("manifests/REF_636_RECENT.json", json.dumps(manifest).encode())
    return client


@pytest.fixture
def cfg(tmp_path):
    return {
        "cache": {"local_cache_dir": str(tmp_path / "cache")},
        "r2": {"bucket_name": "test"},
        "pipeline": {"samples": ["REF_636_RECENT"], "mosaic_resize_factor": 1.0},
        "cellpose": {"device": "cpu", "model": "cpsam_v2"},
    }


@pytest.fixture
def fake_model(monkeypatch):
    calls = []

    class Model:
        def eval(self, image, **kwargs):
            calls.append(image.shape)
            labels = np.zeros(image.shape[:2], np.uint32)
            labels[8:28, 8:28] = 1
            return labels, None, None

    model = Model()
    monkeypatch.setattr(
        pipeline, "create_model", lambda settings: (model, {"device": "fake", "weights_sha256": "test-model"})
    )
    monkeypatch.setattr(pipeline, "make_tissue_mask", lambda image: np.ones(image.shape[:2], bool))
    return calls
