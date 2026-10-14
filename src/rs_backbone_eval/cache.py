"""Content-addressed cache keys for worker-side derived data.

Normalization statistics and frozen features are expensive to recompute, but a
cache keyed on a fixed name silently goes stale when the manifest, the
checkpoint or the preprocessing changes. ``cache_key`` folds the manifest
content hash and the caller's extra parts into one short, stable key.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(manifest_path: Path, *parts: object) -> str:
    """Stable 16-hex key from the manifest content plus the given parts."""
    digest = hashlib.sha256()
    digest.update(file_sha256(Path(manifest_path)).encode("ascii"))
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
    return digest.hexdigest()[:16]
