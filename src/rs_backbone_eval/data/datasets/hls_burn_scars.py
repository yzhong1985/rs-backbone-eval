"""HLS Burn Scar manifest builder for T2 (Wildfire Scar Detection).

Layout after the prepare step (``raw/hls_burn_scars/{training,validation}``):

- ``<stem>_merged.tif`` is the 6-band image, ``<stem>.mask.tif`` the label;
- the stem encodes system (HLS.S30 / HLS.L30), tile (e.g. T14SNB) and
  acquisition date as year+day-of-year (e.g. ``2018215``).

Split policy (evaluation design 4.2): the official ``training`` directory is
split deterministically into ``train`` (90%) and ``validation`` (10%); the
official ``validation`` directory becomes ``official_validation`` and is only
reported as such -- it is not an independent test set. ``group_id`` is the HLS
tile so the audit surfaces geographic overlap across splits.
"""

from __future__ import annotations

import hashlib
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample

DATASET_DIR = Path("raw") / "hls_burn_scars"
GSD_M = 30.0
PROJECT_VALIDATION_FRACTION = 0.10
SPLIT_SEED = 17
HLS_STEM = re.compile(r"(T[A-Z0-9]{5})\.(\d{7})")


class ManifestBuildError(ValueError):
    """Raised when the local dataset cannot produce a complete manifest."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def parse_stem(stem: str) -> tuple[str, str, str]:
    """Return (system, tile, ISO date) from an HLS chip stem."""
    match = HLS_STEM.search(stem)
    if match is None:
        raise ManifestBuildError(f"cannot parse HLS chip name: {stem}")
    tile, stamp = match.group(1), match.group(2)
    if "S30" in stem:
        system = "HLS.S30"
    elif "L30" in stem:
        system = "HLS.L30"
    else:
        system = "unknown"
    year, day_of_year = int(stamp[:4]), int(stamp[4:])
    date = (datetime(year, 1, 1) + timedelta(days=day_of_year - 1)).date().isoformat()
    return system, tile, date


def _collect(directory: Path) -> list[str]:
    images = sorted(directory.glob("*_merged.tif"))
    if not images:
        raise ManifestBuildError(f"no *_merged.tif found in {directory}")
    return [path.name.removesuffix("_merged.tif") for path in images]


def build_hls_burn_scars_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    base = data_root / DATASET_DIR
    training_dir, validation_dir = base / "training", base / "validation"
    for directory in (training_dir, validation_dir):
        if not directory.is_dir():
            raise ManifestBuildError(f"missing directory (run the extract step first): {directory}")

    project_train = _collect(training_dir)
    rng = random.Random(SPLIT_SEED)
    shuffled = sorted(project_train)
    rng.shuffle(shuffled)
    n_validation = max(1, int(len(shuffled) * PROJECT_VALIDATION_FRACTION))
    validation = set(shuffled[:n_validation])
    split_of = {stem: ("validation" if stem in validation else "train") for stem in project_train}
    for stem in _collect(validation_dir):
        split_of[stem] = "official_validation"

    samples: list[GeoSequenceSample] = []
    missing: list[str] = []
    for stem, split in sorted(split_of.items()):
        directory = validation_dir if split == "official_validation" else training_dir
        image_path = directory / f"{stem}_merged.tif"
        label_path = directory / f"{stem}.mask.tif"
        if not image_path.is_file() or not label_path.is_file():
            missing.append(stem)
            continue
        _, tile, date = parse_stem(stem)
        samples.append(
            GeoSequenceSample(
                task_id=task_spec.task_id,
                sample_id=stem,
                group_id=tile,
                split=split,
                image_paths=[image_path.relative_to(data_root).as_posix()],
                label_path=label_path.relative_to(data_root).as_posix(),
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                timestamps=[date],
                temporal_mask=[True],
                crs=None,
                transform=None,
                gsd_m=GSD_M,
                checksum=file_sha256(image_path),
            )
        )
    if missing:
        raise ManifestBuildError(
            f"{len(missing)} chips missing a mask or image, e.g. {missing[:3]}"
        )
    if not samples:
        raise ManifestBuildError("no samples found; is the dataset extracted?")
    return samples
