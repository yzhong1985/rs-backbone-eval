"""iSAID manifest: whole-image tiling over RGB aerial imagery (T9).

iSAID ships annotations and *semantic masks* only; the images are DOTA-v1.0,
so the builder expects the DOTA archives extracted next to the annotations:

    raw/isaid/train/iSAID_train.json          annotations
    raw/isaid/val/iSAID_val.json
    derived/isaid/masks/{train,val}/images/   semantic masks (PNG, RGB palette)
    raw/isaid/dota/images/{train,val}/        DOTA PNGs (extracted)

Each manifest row is one **tile**, not one image: iSAID images are 400-13000 px
wide, and the protocol tiles at 1024/stride 512 with padded edges (see
``data.tiling``). Rows therefore carry:

* ``group_id`` = the source image stem (``P0000``), so a split never puts two
  tiles of the same image on both sides — a prerequisite for ``holdout-manifest``
  to carve a validation split;
* ``window`` = ``[x, y, w, h]`` in source pixels;
* ``valid_fraction`` = share of the tile that is real data (a 387x278 image
  padded to 1024 is ~10% real).

Padding is filled with the ignore index by the worker, never labelled
(review 2026-10-29).

The official split is ``train``/``val``; ``val`` becomes the test split and a
validation subset is carved from ``train`` by ``holdout-manifest``.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample
from ..tiling import crop_from_tile, tiles_covering, valid_fraction
from .sen1floods11 import ManifestBuildError, file_sha256

DEFAULT_CROP = 1024
DEFAULT_STRIDE = 512
# iSAID/DOTA carry no per-image GSD in the download we use (the `gsd` line lives
# in DOTA's labelTxt, which is not part of iSAID and was not fetched). The plan
# declares a fixed 1.0 m and records `gsd_source=declared` on every run.
DEFAULT_GSD_M = 1.0
SPLIT_MAP = {"train": "train", "val": "test"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def png_size(path: Path) -> tuple[int, int]:
    """Width/height from a PNG header.

    The control plane must not depend on Pillow just to learn an image size;
    PNG puts the two integers at a fixed offset in the IHDR chunk.
    """
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ManifestBuildError(f"{path.name}: not a PNG with an IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ManifestBuildError(f"{path.name}: invalid PNG size {width}x{height}")
    return int(width), int(height)


def build_isaid_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    raw = data_root / "raw" / "isaid"
    masks_root = data_root / "derived" / "isaid" / "masks"
    images_root = raw / "dota" / "images"
    crop = _env_int("RS_EVAL_ISAID_CROP", DEFAULT_CROP)
    stride = _env_int("RS_EVAL_ISAID_STRIDE", DEFAULT_STRIDE)
    gsd = _env_float("RS_EVAL_ISAID_GSD", DEFAULT_GSD_M)
    if stride <= 0 or crop <= 0:
        raise ManifestBuildError("RS_EVAL_ISAID_CROP/STRIDE must be positive")

    checksums: dict[Path, str] = {}
    samples: list[GeoSequenceSample] = []
    for official, target in SPLIT_MAP.items():
        annotation = raw / official / f"iSAID_{official}.json"
        if not annotation.is_file():
            raise ManifestBuildError(f"missing annotation file: {annotation}")
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        for image in payload["images"]:
            name = image["file_name"]
            stem = Path(name).stem
            image_path = images_root / official / name
            mask_path = masks_root / official / "images" / image["seg_file_name"]
            if not image_path.is_file():
                raise ManifestBuildError(
                    f"missing image {image_path}; extract the DOTA archives into "
                    f"{images_root}/{official}/"
                )
            if not mask_path.is_file():
                raise ManifestBuildError(f"missing semantic mask {mask_path}")
            width, height = png_size(mask_path)
            if mask_path not in checksums:
                checksums[mask_path] = file_sha256(mask_path)
            digest = checksums[mask_path]
            for origin in tiles_covering(width, height, crop, stride):
                window = crop_from_tile(crop, origin)
                samples.append(
                    GeoSequenceSample(
                        task_id=task_spec.task_id,
                        sample_id=f"{stem}_x{origin[0]:05d}_y{origin[1]:05d}",
                        group_id=stem,
                        split=target,
                        image_paths=[image_path.relative_to(data_root).as_posix()],
                        label_path=mask_path.relative_to(data_root).as_posix(),
                        sensor=task_spec.sensor,
                        bands=list(task_spec.bands),
                        timestamps=[0],
                        temporal_mask=[True],
                        crs=None,
                        transform=None,
                        gsd_m=gsd,
                        checksum=digest,
                        window=window,
                        valid_fraction=round(
                            valid_fraction(width, height, crop, stride, origin), 6
                        ),
                    )
                )
    if not samples:
        raise ManifestBuildError("no iSAID tiles produced; are the annotations present?")
    return samples
