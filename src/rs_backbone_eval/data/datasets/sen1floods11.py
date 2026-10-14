"""Sen1Floods11 v1.1 hand-labeled manifest builder for T1 (Flood Detection).

Layout assumptions (official GCS bucket, synced to ``data/raw/sen1floods11/v1.1``):

- ``splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv`` list
  ``<chip>_S1Hand.tif,<chip>_LabelHand.tif`` pairs; splits are chip-random and
  the same locations appear in train/validation/test by official design;
- ``data/flood_events/HandLabeled/S2Hand/<chip>_S2Hand.tif`` are the 13-band
  S2 L1C TOA chips used by the task protocol (band subset per TaskSpec);
- ``Sen1Floods11_Metadata.geojson`` is event-level: one entry per flood event
  with ``location`` and ``s2_date``.

Chip-to-event mapping uses the location prefix of the chip ID. The bucket
names Cambodia chips ``Mekong_*``; the alias is explicit below. ``group_id``
is the event so split audits can surface the known official cross-split
location mixing, which the evaluation design keeps for B0 reproduction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample

DATASET_VERSION_DIR = Path("raw") / "sen1floods11" / "v1.1"
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "validation": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia_holdout": "flood_bolivia_data.csv",
}
CHIP_GSD_M = 10.0
EVENT_ALIASES = {"Mekong": "Cambodia"}


class ManifestBuildError(ValueError):
    """Raised when the local dataset cannot produce a complete manifest."""


@dataclass(frozen=True)
class FloodEvent:
    event_id: int
    location: str
    s2_date: str
    s1_date: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_events(data_root: Path) -> dict[str, FloodEvent]:
    """Map every event location (and its chip-naming alias) to one event."""
    metadata_path = data_root / DATASET_VERSION_DIR / "Sen1Floods11_Metadata.geojson"
    if not metadata_path.is_file():
        raise ManifestBuildError(f"missing event metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    events: dict[str, FloodEvent] = {}
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        location = properties.get("location")
        raw_date = properties.get("s2_date")
        if not location or not raw_date:
            continue
        parsed = datetime.strptime(raw_date.replace("/", "-"), "%Y-%m-%d").date()
        raw_s1 = properties.get("s1_date")
        s1_date = (
            datetime.strptime(raw_s1.replace("/", "-"), "%Y-%m-%d").date().isoformat()
            if raw_s1
            else None
        )
        event = FloodEvent(
            event_id=int(properties.get("ID", -1)),
            location=location,
            s2_date=parsed.isoformat(),
            s1_date=s1_date,
        )
        for name in (location, *(k for k, v in EVENT_ALIASES.items() if v == location)):
            events[name] = event
    return events


def iter_chips(splits_dir: Path) -> list[tuple[str, str]]:
    """Yield (split, chip_id) pairs in a stable order from the official CSVs."""
    chips: list[tuple[str, str]] = []
    for split in sorted(SPLIT_FILES):
        path = splits_dir / SPLIT_FILES[split]
        if not path.is_file():
            raise ManifestBuildError(f"missing official split file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            chip = line.split(",", 1)[0].removesuffix("_S1Hand.tif")
            chips.append((split, chip))
    return chips


def build_sen1floods11_manifest(
    data_root: Path,
    task_spec: TaskSpec,
    *,
    layer: str = "S2Hand",
    date_field: str = "s2_date",
) -> list[GeoSequenceSample]:
    """Build a T1 manifest from official splits and one hand-labeled layer.

    ``layer="S2Hand"`` uses the optical chips (the Prithvi paper protocol);
    ``layer="S1Hand"`` uses the Sentinel-1 chips, which share the same
    hand-drawn labels and can therefore serve as the SAR counterpart of T1.
    """
    version_root = data_root / DATASET_VERSION_DIR
    hand_root = version_root / "data" / "flood_events" / "HandLabeled"
    splits_dir = version_root / "splits" / "flood_handlabeled"
    events = load_events(data_root)

    samples: list[GeoSequenceSample] = []
    missing: list[str] = []
    unmapped: set[str] = set()
    for split, chip in iter_chips(splits_dir):
        location = chip.split("_", 1)[0]
        event = events.get(location)
        if event is None:
            unmapped.add(location)
            continue
        image_path = hand_root / layer / f"{chip}_{layer}.tif"
        label_path = hand_root / "LabelHand" / f"{chip}_LabelHand.tif"
        if not image_path.is_file() or not label_path.is_file():
            missing.append(chip)
            continue
        samples.append(
            GeoSequenceSample(
                task_id=task_spec.task_id,
                sample_id=chip,
                group_id=event.location,
                split=split,
                image_paths=[image_path.relative_to(data_root).as_posix()],
                label_path=label_path.relative_to(data_root).as_posix(),
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                timestamps=[getattr(event, date_field) or event.s2_date],
                temporal_mask=[True],
                crs=None,
                transform=None,
                gsd_m=CHIP_GSD_M,
                checksum=file_sha256(image_path),
            )
        )

    if unmapped:
        raise ManifestBuildError(f"chips without event metadata: {sorted(unmapped)}")
    if missing:
        raise ManifestBuildError(
            f"{len(missing)} chips missing {layer}/LabelHand files: {missing[:5]}"
        )
    if not samples:
        raise ManifestBuildError("no samples found; is the dataset synced?")
    return samples
