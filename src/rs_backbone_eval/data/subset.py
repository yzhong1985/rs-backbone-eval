"""Manifest subsetting for label-budget and geographic/temporal-holdout experiments.

Three derivations of an existing manifest, all deterministic and group-aware so
a single scene/tile/case never straddles the derived splits:

- ``few_shot_subset`` keeps a fraction of the training groups (10%/30%/100% for
  the sample-efficiency curve); validation and test are untouched so the
  comparison stays paired.
- ``geographic_holdout`` moves a fraction of the training groups into a new
  holdout split that must not be used for training or hyperparameter search.
- ``temporal_holdout`` moves every training group with an observation at or
  after a cutoff date, so the holdout is a genuine forward-in-time test and no
  post-cutoff observation leaks into training.

Sampling is by ``group_id`` (scene/tile/case), never by pixel or by sample, so
the curve measures transfer from fewer labelled *units*, not fewer pixels. The
caller is still responsible for checking class coverage of the kept groups.

Both holdout tools refuse to leave a held-out region partly in another split: a
group that already appears in more than one split would keep the holdout region
in training or evaluation. The caller must either fix the source manifest or ask
for ``isolate=True``, which moves every row of the selected groups into the
holdout split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np

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


def groups_in_split(samples: Iterable[ManifestSample], split: str) -> list[str]:
    return sorted({sample.group_id for sample in samples if sample.split == split})


def group_splits(samples: Iterable[ManifestSample]) -> dict[str, list[str]]:
    """Map every group to the sorted splits it appears in."""
    mapping: dict[str, set[str]] = {}
    for sample in samples:
        mapping.setdefault(sample.group_id, set()).add(sample.split)
    return {group: sorted(splits) for group, splits in mapping.items()}


def straddling_groups(
    samples: Iterable[ManifestSample], groups: Iterable[str]
) -> dict[str, list[str]]:
    """Selected groups that appear in more than one split (a holdout leak)."""
    per_group = group_splits(samples)
    selected = set(groups)
    return {
        group: per_group[group] for group in sorted(selected) if len(per_group.get(group, ())) > 1
    }


def choose_groups(groups: list[str], fraction: float, seed: int) -> set[str]:
    """Deterministically keep ``fraction`` of the values (at least one)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    keep = max(1, round(len(shuffled) * fraction))
    return set(shuffled[:keep])


def parse_cutoff(cutoff: str | int) -> np.datetime64:
    """Parse a holdout cutoff: a year (``2018``) or an ISO date (``2018-06-01``)."""
    if isinstance(cutoff, int):
        if not 1900 <= cutoff <= 2200:
            raise ValueError(f"integer cutoff must be a year, got {cutoff}")
        return np.datetime64(f"{cutoff}-01-01", "D")
    text = str(cutoff).strip()
    if not text:
        raise ValueError("cutoff must not be empty")
    return np.datetime64(text[:10] if len(text) >= 10 else text, "D")


def observation_date(stamp: str | int) -> np.datetime64:
    """Observation date of one timestamp (integers are days since the epoch)."""
    if isinstance(stamp, int):
        return np.datetime64(stamp, "D")
    return np.datetime64(str(stamp)[:10], "D")


def _assert_isolated(
    samples: Iterable[ManifestSample],
    moved: set[str],
    *,
    isolate: bool,
    hint: str,
) -> None:
    if isolate:
        return
    straddling = straddling_groups(samples, moved)
    if not straddling:
        return
    detail = "; ".join(
        f"{group}: {', '.join(splits)}" for group, splits in list(straddling.items())[:5]
    )
    raise ValueError(
        f"{hint} would leave {len(straddling)} group(s) in another split ({detail}); "
        "fix the source manifest or pass isolate=True to move every row of the "
        "selected groups into the holdout split"
    )


def _apply_holdout(
    samples: list[ManifestSample],
    moved: set[str],
    *,
    source_split: str,
    holdout_split: str,
    isolate: bool,
) -> list[ManifestSample]:
    out: list[ManifestSample] = []
    for sample in samples:
        if sample.group_id in moved and (isolate or sample.split == source_split):
            out.append(sample.model_copy(update={"split": holdout_split}))
        else:
            out.append(sample)
    return out


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
    isolate: bool = False,
) -> list[ManifestSample]:
    """Move a fraction of ``source_split`` groups into ``holdout_split``.

    Raises ``ValueError`` when a selected group also appears in another split,
    unless ``isolate=True`` moves every row of those groups so the holdout
    region is fully excluded from training, validation and the official test.
    """
    moved = choose_groups(groups_in_split(samples, source_split), fraction, seed)
    _assert_isolated(samples, moved, isolate=isolate, hint="geographic_holdout")
    return _apply_holdout(
        samples,
        moved,
        source_split=source_split,
        holdout_split=holdout_split,
        isolate=isolate,
    )


def temporal_holdout(
    samples: list[ManifestSample],
    cutoff: str | int,
    *,
    source_split: str = "train",
    holdout_split: str = TEMPORAL_HOLDOUT,
    isolate: bool = False,
) -> list[ManifestSample]:
    """Move every ``source_split`` group with an observation on/after ``cutoff``.

    The cutoff is a year or an ISO date. A group with any post-cutoff
    observation goes entirely into the holdout, so no post-cutoff observation
    can leak into training even when a group spans the cutoff.
    """
    boundary = parse_cutoff(cutoff)
    moved: set[str] = set()
    for sample in samples:
        if sample.split != source_split:
            continue
        if any(observation_date(stamp) >= boundary for stamp in sample.timestamps):
            moved.add(sample.group_id)
    _assert_isolated(samples, moved, isolate=isolate, hint="temporal_holdout")
    return _apply_holdout(
        samples,
        moved,
        source_split=source_split,
        holdout_split=holdout_split,
        isolate=isolate,
    )


def write_derived(path: Path, samples: list[ManifestSample]) -> str:
    """Write a derived manifest and return its content hash."""
    return write_manifest(path, samples)
