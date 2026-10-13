"""Shared probe helpers for the AnySat worker.

The probe validates the environment contract (manifest, checkpoint, torch,
upstream source pin) and performs one real forward pass on a T1 chip using
AnySat's native Sentinel-2 entry: 10 bands selected from the 13-band L1C
chips, raw reflectance, normalized with train-split statistics computed the
same way AnySat's own dataset code does (mean over chips of per-chip mean).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# AnySat's official S2 band selection: 0-based indices into the 13-band
# Sen1Floods11 L1C chip (B1..B12): B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12.
S2_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
S2_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
UPSTREAM_REPO = "https://github.com/gastruc/AnySat"
NORM_FILENAME = "NORM_s2_patch.json"


def add_upstream_to_path() -> Path:
    """Expose the pinned AnySat checkout (outside the repo) for import."""
    source = os.environ.get("ANYSAT_SRC")
    if not source:
        raise RuntimeError("ANYSAT_SRC is not set; clone gastruc/AnySat outside the repo")
    root = Path(source).resolve()
    if not (root / "hubconf.py").is_file():
        raise RuntimeError(f"ANYSAT_SRC does not look like the AnySat checkout: {root}")
    import sys

    for candidate in (root, root / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


def upstream_commit(source_root: Path) -> str:
    head = source_root / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        ref_path = source_root / ".git" / ref.removeprefix("ref: ").strip()
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
    return ref


def read_chip(path: Path, *, band_indices: list[int] | None = None):
    """Read one Sen1Floods11 S2Hand chip as a float array (bands first)."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as source:
        data = source.read()
    array = np.asarray(data, dtype=np.float32)
    if band_indices is not None:
        array = array[band_indices]
    return array


def training_normalization(manifest_path: Path, data_root: Path, cache_dir: Path, limit: int):
    """Per-band mean/std over training chips, mirroring AnySat's own method."""
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / NORM_FILENAME
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return (
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
            0,
            str(cache_file),
        )

    means, stds = [], []
    used = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if used >= limit:
                break
            row = json.loads(line)
            if row["split"] != "train":
                continue
            chip = read_chip(data_root / row["image_paths"][0], band_indices=S2_BAND_INDICES)
            means.append(chip.mean(axis=(1, 2)))
            stds.append(chip.std(axis=(1, 2)))
            used += 1
    if not means:
        raise RuntimeError("no training chips found for normalization statistics")
    mean = np.stack(means).mean(axis=0).astype(float)
    std = np.stack(stds).mean(axis=0).astype(float)
    cache_file.write_text(
        json.dumps({"mean": list(mean), "std": list(std), "chips": used}, indent=2),
        encoding="utf-8",
    )
    return mean.astype(np.float32), std.astype(np.float32), used, str(cache_file)
