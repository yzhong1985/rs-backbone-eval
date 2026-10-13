"""Shared helpers for the Clay v1.5 worker.

Clay has no usable PyPI package layout (the sdist imports a missing
``src.backbone``), so the official repository is cloned outside the repo at a
pinned commit and exposed via ``CLAY_SRC``.

Sentinel-2 input follows Clay's own ``configs/metadata.yaml``: ten bands in
its documented order (blue..swir22, which are B2..B12 of the S2 L1C chips),
raw reflectance normalized with the sensor mean/std, plus time/latlon/gsd/
wavelength metadata. Segmentation features come from ``SegmentEncoder``,
Clay's own finetuning recipe that loads the ``model.encoder.*`` weights from
the checkpoint and returns a patch-token grid.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SENSOR = "sentinel-2-l2a"
# 0-based indices into the 13-band Sen1Floods11 L1C chip, matching Clay's
# sentinel-2-l2a band_order (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12).
S2_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
UPSTREAM_REPO = "https://github.com/Clay-foundation/model"


def add_upstream_to_path() -> Path:
    source = os.environ.get("CLAY_SRC")
    if not source:
        raise RuntimeError("CLAY_SRC is not set; clone Clay-foundation/model outside the repo")
    root = Path(source).resolve()
    if not (root / "claymodel" / "module.py").is_file():
        raise RuntimeError(f"CLAY_SRC does not look like the Clay checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
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


def load_sensor_metadata(source_root: Path) -> dict:
    import yaml

    metadata_path = source_root / "configs" / "metadata.yaml"
    if not metadata_path.is_file():
        raise RuntimeError(f"Clay sensor metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if SENSOR not in metadata:
        raise RuntimeError(f"{SENSOR} missing from Clay metadata")
    return metadata[SENSOR]


def read_chip(path: Path):
    import numpy as np
    import rasterio

    with rasterio.open(path) as source:
        data = source.read()
    return np.asarray(data, dtype=np.float32)[S2_BAND_INDICES]
