"""Landslide4Sense manifest builder for T4 (Landslide Detection).

Layout (``raw/landslide4sense/main``):

- ``images/{train,validation,test}/image_<n>.h5`` holds a 14-band 128x128 patch
  under the ``img`` dataset (bands per TerraTorch: 12 Sentinel-2 bands plus
  slope and DEM);
- ``annotations/{train,validation,test}/mask_<n>.h5`` holds a binary 128x128
  mask under the ``mask`` dataset (0 = non-landslide, 1 = landslide).

The official split is kept as-is: ``train`` / ``validation`` / ``test``. A few
stray ``annotations/train/image_*.h5`` files are not masks and are ignored; the
builder pairs every image with ``mask_<n>.h5`` and fails loudly if one is
missing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample

DATASET_DIR = Path("raw") / "landslide4sense" / "main"
SPLITS = ("train", "validation", "test")
GSD_M = 10.0


class ManifestBuildError(ValueError):
    """Raised when the local dataset cannot produce a complete manifest."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_landslide4sense_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    base = data_root / DATASET_DIR
    samples: list[GeoSequenceSample] = []
    missing: list[str] = []
    for split in SPLITS:
        image_dir = base / "images" / split
        label_dir = base / "annotations" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ManifestBuildError(f"missing directory (run the extract step first): {image_dir}")
        for image_path in sorted(image_dir.glob("image_*.h5")):
            number = image_path.stem.removeprefix("image_")
            label_path = label_dir / f"mask_{number}.h5"
            if not label_path.is_file():
                missing.append(f"{split}/image_{number}")
                continue
            samples.append(
                GeoSequenceSample(
                    task_id=task_spec.task_id,
                    sample_id=f"{split}_image_{number}",
                    group_id=f"{split}_image_{number}",
                    split=split,
                    image_paths=[image_path.relative_to(data_root).as_posix()],
                    label_path=label_path.relative_to(data_root).as_posix(),
                    sensor=task_spec.sensor,
                    bands=list(task_spec.bands),
                    timestamps=[0],
                    temporal_mask=[True],
                    crs=None,
                    transform=None,
                    gsd_m=GSD_M,
                    checksum=file_sha256(image_path),
                )
            )
    if missing:
        raise ManifestBuildError(f"{len(missing)} images missing a mask, e.g. {missing[:3]}")
    if not samples:
        raise ManifestBuildError("no samples found; is the dataset extracted?")
    return samples
