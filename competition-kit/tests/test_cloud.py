import base64
import hashlib
import sys
import types
from pathlib import Path

import pytest
from ensemble_benchmark.cloud import download_data


class Blob:
    def __init__(self, name, data=b"{}", generation=1):
        self.name = name
        self.data = data
        self.size = len(data)
        self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.crc32c = None
        self.generation = generation
        self.calls = []
        self.interrupt = False

    def download_to_filename(self, filename, **kwargs):
        self.calls.append(kwargs)
        Path(filename).write_bytes(self.data)
        if self.interrupt:
            raise InterruptedError("connection lost")


@pytest.fixture
def remote(monkeypatch):
    blobs = [
        Blob("prepared/train/manifest.json"),
        Blob("prepared/train/audio.wav", b"audio"),
        Blob("prepared/validation/manifest.json"),
        Blob("prepared/test/private.wav", b"never download"),
    ]
    requests = []

    class Client:
        def __init__(self, project=None):
            self.project = project

        def list_blobs(self, bucket, *, prefix):
            requests.append((bucket, prefix))
            return [blob for blob in blobs if blob.name.startswith(prefix)]

    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = Client
    cloud.storage = storage
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)
    return blobs, requests


def test_download_only_requested_splits_and_reuse_verified_files(remote, tmp_path):
    blobs, requests = remote
    assert download_data("gs://dataset/prepared", tmp_path) == tmp_path
    download_data("gs://dataset/prepared", tmp_path)
    assert requests == [
        ("dataset", f"prepared/{split}/") for _ in range(2) for split in ("train", "validation")
    ]
    assert [len(blob.calls) for blob in blobs] == [1, 1, 1, 0]
    assert blobs[1].calls == [{"if_generation_match": 1, "checksum": "auto"}]
    assert not (tmp_path / "test").exists()


def test_repair_same_size_corruption_and_remote_changed_version(remote, tmp_path):
    blobs, _ = remote
    download_data("gs://dataset/prepared", tmp_path)
    (tmp_path / "train/audio.wav").write_bytes(b"wrong")
    download_data("gs://dataset/prepared", tmp_path)
    assert (tmp_path / "train/audio.wav").read_bytes() == b"audio"
    blobs[1] = Blob("prepared/train/audio.wav", b"newer", generation=2)
    download_data("gs://dataset/prepared", tmp_path)
    assert (tmp_path / "train/audio.wav").read_bytes() == b"newer"
    assert blobs[1].calls[0]["if_generation_match"] == 2


def test_interrupted_download_cleans_temp_and_can_retry(remote, tmp_path):
    blobs, _ = remote
    blobs[1].interrupt = True
    with pytest.raises(InterruptedError):
        download_data("gs://dataset/prepared", tmp_path)
    assert not (tmp_path / "train/audio.wav").exists()
    assert not list(tmp_path.rglob("*.part"))
    blobs[1].interrupt = False
    download_data("gs://dataset/prepared", tmp_path)
    assert len(blobs[0].calls) == 1
    assert (tmp_path / "train/audio.wav").read_bytes() == b"audio"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a\\escape"])
def test_unsafe_object_paths_rejected_before_download(remote, tmp_path, name):
    blobs, _ = remote
    blobs.insert(1, Blob(f"prepared/train/{name}"))
    with pytest.raises(ValueError, match="Unsafe"):
        download_data("gs://dataset/prepared", tmp_path)
    assert all(not blob.calls for blob in blobs)


def test_symlink_escape_rejected(remote, tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "train").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        download_data("gs://dataset/prepared", destination)


def test_missing_manifest_fails_before_any_download(remote, tmp_path):
    blobs, _ = remote
    blobs[:] = [blob for blob in blobs if "validation" not in blob.name]
    with pytest.raises(ValueError, match="validation/manifest.json"):
        download_data("gs://dataset/prepared", tmp_path)
    assert all(not blob.calls for blob in blobs)


def test_missing_file_is_redownloaded(remote, tmp_path):
    blobs, _ = remote
    download_data("gs://dataset/prepared", tmp_path)
    (tmp_path / "train/audio.wav").unlink()
    download_data("gs://dataset/prepared", tmp_path)
    assert len(blobs[1].calls) == 2
    assert len(blobs[0].calls) == 1


def test_failed_replacement_preserves_existing_file(remote, tmp_path):
    blobs, _ = remote
    download_data("gs://dataset/prepared", tmp_path)
    blobs[1] = Blob("prepared/train/audio.wav", b"changed", generation=2)
    blobs[1].interrupt = True
    with pytest.raises(InterruptedError):
        download_data("gs://dataset/prepared", tmp_path)
    assert (tmp_path / "train/audio.wav").read_bytes() == b"audio"
    assert not list(tmp_path.rglob("*.part"))


def test_size_alone_never_reuses_file(remote, tmp_path):
    blobs, _ = remote
    blobs[1].md5_hash = None
    download_data("gs://dataset/prepared", tmp_path)
    download_data("gs://dataset/prepared", tmp_path)
    assert len(blobs[1].calls) == 2


def test_crc32c_reuses_composite_object(remote, tmp_path, monkeypatch):
    blobs, _ = remote
    checksum_module = types.ModuleType("google_crc32c")
    # A deterministic fake exercises the SDK checksum boundary without network access.
    checksum_module.Checksum = hashlib.md5
    monkeypatch.setitem(sys.modules, "google_crc32c", checksum_module)
    blobs[1].crc32c = blobs[1].md5_hash
    blobs[1].md5_hash = None
    download_data("gs://dataset/prepared", tmp_path)
    download_data("gs://dataset/prepared", tmp_path)
    assert len(blobs[1].calls) == 1
