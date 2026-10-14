"""Burn scar intensity manifest builder for T3 (Burn Scar Intensity).

Layout after extraction (``raw/burn_intensity/extracted``):

- ``pre/``, ``during/``, ``post/`` each hold ``HLS_<case>_<row>_<col>.tiff``
  (six HLS bands, 224x224) and ``BS_<case>_<row>_<col>.tiff`` (burn severity
  0..4). The label rasters are byte-identical in all three directories, so the
  builder reads them from ``pre/`` exactly like the official terratorch
  ``BurnIntensityNonGeo`` reader does.
- ``BS_files_raw.csv`` lists the 296 official chips (the filtered
  ``BS_files_with_less_than_25_percent_zeros.csv`` is the 49-chip subset that the
  ``use_full_data: false`` setting would use; the official config uses the full
  data, so this builder does too).

Split policy: the HuggingFace release ships no ``train.txt`` / ``val.txt``, so
the official 80/20 split cannot be reproduced byte-for-byte. Following the
evaluation design's anti-leakage stance, the split is made at the **fire case**
level (chips of one burn scar never straddle splits) with the project seed:
80% train, 10% validation for checkpoint selection, 10% test reported once.
The paper's 80/20 was image-level; this deviation is recorded here on purpose.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample

DATASET_DIR = Path("raw") / "burn_intensity"
EXTRACT_DIR = DATASET_DIR / "extracted"
TIME_STEPS = ("pre", "during", "post")
LABEL_DIR = "pre"
GSD_M = 30.0
SPLIT_SEED = 17
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10


class ManifestBuildError(ValueError):
    """Raised when the local dataset cannot produce a complete manifest."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def case_of(stem: str) -> str:
    """Return the fire case: the chip stem without its trailing row/col pair."""
    return stem.rsplit("_", 2)[0]


def _label_stems(base: Path) -> list[str]:
    labels = sorted(
        path.name.removesuffix(".tiff") for path in (base / LABEL_DIR).glob("BS_*.tiff")
    )
    if not labels:
        raise ManifestBuildError(f"no BS_*.tiff found in {base / LABEL_DIR}")
    return [name.removeprefix("BS_") for name in labels]


def _split_cases(cases: list[str]) -> dict[str, str]:
    """Assign every fire case to exactly one split, deterministically."""
    rng = random.Random(SPLIT_SEED)
    shuffled = sorted(cases)
    rng.shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * TRAIN_FRACTION))
    n_validation = max(1, round(len(shuffled) * VALIDATION_FRACTION))
    split_of: dict[str, str] = {}
    for index, case in enumerate(shuffled):
        if index < n_train:
            split_of[case] = "train"
        elif index < n_train + n_validation:
            split_of[case] = "validation"
        else:
            split_of[case] = "test"
    return split_of


def build_burn_intensity_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    base = data_root / EXTRACT_DIR
    for step in TIME_STEPS:
        if not (base / step).is_dir():
            raise ManifestBuildError(
                f"missing directory (run the extract step first): {base / step}"
            )

    stems = _label_stems(base)
    split_of = _split_cases(sorted({case_of(stem) for stem in stems}))

    samples: list[GeoSequenceSample] = []
    missing: list[str] = []
    for stem in stems:
        images = [base / step / f"HLS_{stem}.tiff" for step in TIME_STEPS]
        label = base / LABEL_DIR / f"BS_{stem}.tiff"
        if not all(path.is_file() for path in images) or not label.is_file():
            missing.append(stem)
            continue
        samples.append(
            GeoSequenceSample(
                task_id=task_spec.task_id,
                sample_id=stem,
                group_id=case_of(stem),
                split=split_of[case_of(stem)],
                image_paths=[path.relative_to(data_root).as_posix() for path in images],
                label_path=label.relative_to(data_root).as_posix(),
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                timestamps=list(TIME_STEPS),
                temporal_mask=[True] * len(TIME_STEPS),
                crs=None,
                transform=None,
                gsd_m=GSD_M,
                checksum=file_sha256(images[0]),
            )
        )
    if missing:
        raise ManifestBuildError(
            f"{len(missing)} chips missing an HLS date or label, e.g. {missing[:3]}"
        )
    if not samples:
        raise ManifestBuildError("no samples found; is the dataset extracted?")
    return samples
