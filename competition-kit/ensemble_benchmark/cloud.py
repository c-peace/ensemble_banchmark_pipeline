"""Download an unpacked GCS dataset; DVC cache layouts are not supported."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _matches(path: Path, blob: Any) -> bool:
    if not path.is_file() or path.stat().st_size != blob.size:
        return False
    if blob.md5_hash:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").digest()
        return base64.b64encode(digest).decode() == blob.md5_hash
    if blob.crc32c:
        import google_crc32c

        checksum = google_crc32c.Checksum()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode() == blob.crc32c
    return False  # Size alone cannot establish that a local file is reusable.


def _target(root: Path, name: str) -> Path:
    parts = name.split("/")
    if "\\" in name or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe GCS object path: {name!r}")
    target = root.joinpath(*parts)
    for path in (root, *target.relative_to(root).parents):
        # Relative parents need to be interpreted under the selected dataset root.
        candidate = path if path.is_absolute() else root / path
        if candidate.is_symlink():
            raise ValueError(f"Dataset paths must not be symlinks: {candidate}")
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe dataset destination: {target}")
    return target


def download_data(uri: str, destination: str | Path, *, project: str | None = None) -> Path:
    """Download train/ and validation/ from an unpacked ``gs://bucket/prefix``.

    Both splits must contain a nonempty manifest.json. Files are replaced atomically
    and reused only when their checksum matches GCS. Interrupted calls can be rerun.
    Extra local files are retained; the downloaded manifests define dataset membership.
    The caller supplies Google credentials (for example Colab's authenticate_user()).
    """
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Expected gs://bucket/path to an unpacked prepared dataset")
    prefix = parsed.path.strip("/")
    if "\\" in prefix or any(part in (".", "..") for part in prefix.split("/")):
        raise ValueError("Invalid GCS dataset prefix")
    prefix = f"{prefix}/" if prefix else ""
    from google.cloud import storage

    client = storage.Client(project=project)
    root = Path(destination).expanduser().absolute()
    objects: list[tuple[Any, Path]] = []
    for split in ("train", "validation"):
        split_prefix = f"{prefix}{split}/"
        blobs = list(client.list_blobs(parsed.netloc, prefix=split_prefix))
        if not any(b.name == f"{split_prefix}manifest.json" and b.size for b in blobs):
            raise ValueError(
                f"Missing nonempty {split}/manifest.json at {uri}; DVC caches unsupported"
            )
        for blob in blobs:
            if not blob.name.startswith(split_prefix):
                raise ValueError(f"Unexpected object outside requested split: {blob.name}")
            if blob.name.endswith("/") and blob.size == 0:
                continue
            target = _target(root, blob.name[len(prefix) :])
            objects.append((blob, target))
    # Validate both remote splits before modifying any local dataset files.
    for blob, target in objects:
        _target(root, target.relative_to(root).as_posix())
        if _matches(target, blob):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        os.close(fd)
        try:
            blob.download_to_filename(
                temporary, if_generation_match=blob.generation, checksum="auto"
            )
            if (blob.md5_hash or blob.crc32c) and not _matches(Path(temporary), blob):
                raise ValueError(f"Checksum mismatch downloading {blob.name}")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return root
