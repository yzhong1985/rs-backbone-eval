"""Manifest subsetting for label-budget and geographic-holdout experiments.

Two derivations of an existing manifest, both deterministic and group-aware so
a single scene/tile/case never straddles the derived splits:

- ``few_shot_subset`` keeps a fraction of the training groups (10%/30%/100% for
  the sample-efficiency curve); validation and test are untouched so the
  comparison stays paired.
- ``geographic_holdout`` moves a fraction of the training groups into a new
  holdout split that must not be used for training or hyperparameter search.

Sampling is by ``group_id`` (scene/tile/case), never by pixel or by sample, so
the curve measures transfer from fewer labelled *units*, not fewer pixels. The
caller is still responsible for checking class coverage of the kept groups.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..contracts import ManifestSample
from .manifest import write_manifest

GEOGRAPHIC_HOLDOUT = "geographic_holdout"
TEMPORAL_HOLDOUT = "temporal_holdout"


def read_manifest(path: Path) -> list[ManifestSample]:
    """Load a manifest into memory (validation is the caller's responsibility)."""
    samples: list[ManifestSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(ManifestSample.model_validate(json.loads(line)))
    return samples


def groups_in_split(samples: list[ManifestSample], split: str) -> list[str]:
    return sorted({sample.group_id for sample in samples if sample.split == split})


def choose_groups(groups: list[str], fraction: float, seed: int) -> set[str]:
    """Deterministically keep ``fraction`` of the values (at least one)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    keep = max(1, round(len(shuffled) * fraction))
    return set(shuffled[:keep])


def few_shot_subset(
    samples: list[ManifestSample],
    fraction: float,
    seed: int,
    *,
    source_split: str = "train",
    unit: str = "sample",
) -> list[ManifestSample]:
    """Keep a fraction of ``source_split``; other splits are unchanged.

    ``unit="sample"`` samples labelled chips/images (the usual label-budget
    curve); ``unit="group"`` samples scenes/tiles/cases instead, which is
    stricter and only sensible when a group holds several chips.
    """
    if unit not in {"sample", "group"}:
        raise ValueError("unit must be 'sample' or 'group'")
    if unit == "group":
        keep = choose_groups(groups_in_split(samples, source_split), fraction, seed)
        return [s for s in samples if s.split != source_split or s.group_id in keep]
    ids = sorted({s.sample_id for s in samples if s.split == source_split})
    keep = choose_groups(ids, fraction, seed)
    return [s for s in samples if s.split != source_split or s.sample_id in keep]


def geographic_holdout(
    samples: list[ManifestSample],
    fraction: float,
    seed: int,
    *,
    source_split: str = "train",
    holdout_split: str = GEOGRAPHIC_HOLDOUT,
) -> list[ManifestSample]:
    """Move a fraction of ``source_split`` groups into ``holdout_split``."""
    moved = choose_groups(groups_in_split(samples, source_split), fraction, seed)
    out: list[ManifestSample] = []
    for sample in samples:
        if sample.split == source_split and sample.group_id in moved:
            out.append(sample.model_copy(update={"split": holdout_split}))
        else:
            out.append(sample)
    return out


def write_derived(path: Path, samples: list[ManifestSample]) -> str:
    """Write a derived manifest and return its content hash."""
    return write_manifest(path, samples)
