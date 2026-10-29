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
S2_BAND_ORDER = [
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
]
# High-resolution aerial RGB (iSAID/T9 and FAIR1M/T10) enters through Clay's
# `naip` entry point: 1.0 m, wavelengths 0.48/0.56/0.65 um. The manifests carry
# three bands in R, G, B order, so the indices are simply 0..2 and the NIR entry
# of Clay's naip order is dropped (a missing band is not a zero band).
RGB_SENSOR = "naip"
RGB_BAND_ORDER = ["red", "green", "blue"]
RGB_BAND_INDICES = [0, 1, 2]
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


def prepare_encoder(encoder, freeze_backbone: bool) -> int:
    """Set encoder ``requires_grad`` for the protocol and return trainable params.

    Clay's upstream ``SegmentEncoder.load_from_ckpt()`` freezes every parameter
    it loads, so the fine-tune track must explicitly unfreeze them; otherwise
    B1 silently runs as a frozen probe. The returned count is asserted to be
    non-zero on the fine-tune track.
    """
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze_backbone
    if freeze_backbone:
        encoder.eval()
    trainable = sum(
        parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
    )
    if not freeze_backbone and trainable == 0:
        raise RuntimeError("fine-tune track needs a trainable Clay backbone but none were unfrozen")
    return trainable


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


def load_sensor_metadata(source_root: Path, sensor: str = SENSOR) -> dict:
    import yaml

    metadata_path = source_root / "configs" / "metadata.yaml"
    if not metadata_path.is_file():
        raise RuntimeError(f"Clay sensor metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if sensor not in metadata:
        raise RuntimeError(f"{sensor} missing from Clay metadata")
    return metadata[sensor]


def probe_bands(task_spec) -> tuple[str, list[str], list[int]]:
    """``(clay sensor, band order, band indices)`` for the task's real bands.

    The probe used to hard-code Sentinel-2, so a T9 run (three RGB bands) died
    with `index 3 is out of bounds for axis 0 with size 3` before it ever reached
    the encoder — the Clay iSAID smoke could never have run (review 2026-10-29).
    """
    bands = [band.upper() for band in task_spec.bands]
    if bands == ["RED", "GREEN", "BLUE"]:
        return RGB_SENSOR, list(RGB_BAND_ORDER), list(RGB_BAND_INDICES)
    return SENSOR, list(S2_BAND_ORDER), list(S2_BAND_INDICES)


def pad_chip(chip, pad_to: int):
    """Bottom-right zero pad to ``pad_to x pad_to`` (Clay needs a square grid)."""
    import numpy as np

    height, width = chip.shape[-2:]
    if height > pad_to or width > pad_to:
        raise RuntimeError(f"chip {width}x{height} exceeds the {pad_to}px tile")
    if (height, width) == (pad_to, pad_to):
        return chip
    padded = np.zeros((chip.shape[0], pad_to, pad_to), dtype=chip.dtype)
    padded[:, :height, :width] = chip
    return padded


def read_chip(
    path: Path,
    band_indices: list[int] | None = None,
    window: tuple[int, int, int, int] | None = None,
    pad_to: int | None = None,
):
    """Read a chip (optionally one tile window) and pad it to a square.

    Clay's positional encoding derives a square grid from ``sqrt(num_patches)``,
    so a non-square input raises `Sizes of tensors must match ... Expected size
    18769 but got size 18906` inside ``add_encodings``. The training path always
    pads tiles to ``crop x crop``; the probe read the raw chip and therefore
    crashed on any non-square source scene (review 2026-10-29).
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    indices = S2_BAND_INDICES if band_indices is None else band_indices
    with rasterio.open(path) as source:
        if window is None:
            data = source.read()
        else:
            x, y, width, height = (int(value) for value in window)
            data = source.read(window=Window(x, y, width, height), boundless=True, fill_value=0)
    if max(indices) >= data.shape[0]:
        raise RuntimeError(
            f"{path}: chip has {data.shape[0]} bands, probe asked for {indices}; "
            "the manifest bands and the pixel file disagree"
        )
    chip = np.asarray(data, dtype=np.float32)[indices]
    return chip if pad_to is None else pad_chip(chip, pad_to)
