"""US multi-temporal crop classification manifest builder for T5.

Layout after extraction (``raw/multi-temporal-crop-classification``):

- ``training_chips/`` and ``validation_chips/`` each hold
  ``chip_<x>_<y>_merged.tif`` (18 bands = 3 dates x 6 HLS bands, 224x224) and
  ``chip_<x>_<y>.mask.tif`` (13 crop classes; **0 means invalid** and the
  model adapters must map it to the task's ignore index);
- ``chips_df.csv`` gives the tile and the three acquisition dates per chip.

Split policy (evaluation design 4.2): the official training directory is split
deterministically into ``train`` (90%) and ``validation`` (10%); the official
validation directory becomes ``official_validation`` and is only used for the
final report. The single stacked raster is recorded as one ``image_paths``
entry with three ``timestamps`` (see the manifest contract).
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample

DATASET_DIR = Path("raw") / "multi-temporal-crop-classification"
GSD_M = 30.0
PROJECT_VALIDATION_FRACTION = 0.10
SPLIT_SEED = 17
DATES_PER_CHIP = 3


class ManifestBuildError(ValueError):
    """Raised when the local dataset cannot produce a complete manifest."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_chip_metadata(base: Path) -> dict[str, dict]:
    """chip_id -> {tile, dates[]} from the official chips_df.csv."""
    csv_path = base / "main" / "chips_df.csv"
    if not csv_path.is_file():
        raise ManifestBuildError(f"missing chip metadata: {csv_path}")
    table: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dates = [
                row["first_img_date"],
                row["middle_img_date"],
                row["last_img_date"],
            ]
            table[row["chip_id"]] = {
                "tile": row["tile"],
                "dates": [date[:10] for date in dates],
            }
    return table


def _collect(directory: Path) -> list[str]:
    chips = sorted(
        path.name.removesuffix("_merged.tif")
        for path in directory.glob("*_merged.tif")
        if not path.name.startswith("._")  # macOS resource-fork files in the archive
    )
    if not chips:
        raise ManifestBuildError(f"no *_merged.tif found in {directory}")
    return chips


def build_multicrop_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    base = data_root / DATASET_DIR
    training_dir, validation_dir = base / "training_chips", base / "validation_chips"
    for directory in (training_dir, validation_dir):
        if not directory.is_dir():
            raise ManifestBuildError(f"missing directory (run the extract step first): {directory}")
    metadata = load_chip_metadata(base)

    project_train = _collect(training_dir)
    rng = random.Random(SPLIT_SEED)
    shuffled = sorted(project_train)
    rng.shuffle(shuffled)
    n_validation = max(1, int(len(shuffled) * PROJECT_VALIDATION_FRACTION))
    validation = set(shuffled[:n_validation])
    split_of = {chip: ("validation" if chip in validation else "train") for chip in project_train}
    for chip in _collect(validation_dir):
        split_of[chip] = "official_validation"

    samples: list[GeoSequenceSample] = []
    missing: list[str] = []
    unknown: list[str] = []
    for chip, split in sorted(split_of.items()):
        directory = validation_dir if split == "official_validation" else training_dir
        image_path = directory / f"{chip}_merged.tif"
        label_path = directory / f"{chip}.mask.tif"
        if not image_path.is_file() or not label_path.is_file():
            missing.append(chip)
            continue
        info = metadata.get(chip)
        if info is None:
            unknown.append(chip)
            continue
        dates = info["dates"]
        if len(dates) != DATES_PER_CHIP:
            unknown.append(chip)
            continue
        samples.append(
            GeoSequenceSample(
                task_id=task_spec.task_id,
                sample_id=chip,
                group_id=info["tile"],
                split=split,
                # one stacked 18-band raster holds the three dates
                image_paths=[image_path.relative_to(data_root).as_posix()],
                label_path=label_path.relative_to(data_root).as_posix(),
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                timestamps=dates,
                temporal_mask=[True] * DATES_PER_CHIP,
                crs=None,
                transform=None,
                gsd_m=GSD_M,
                checksum=file_sha256(image_path),
            )
        )
    if missing:
        raise ManifestBuildError(f"{len(missing)} chips missing image or mask, e.g. {missing[:3]}")
    if unknown:
        raise ManifestBuildError(f"{len(unknown)} chips lack date metadata, e.g. {unknown[:3]}")
    if not samples:
        raise ManifestBuildError("no samples found; is the dataset extracted?")
    return samples
