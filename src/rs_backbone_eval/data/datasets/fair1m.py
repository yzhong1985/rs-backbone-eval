"""FAIR1M detection manifest: whole-image tiling over rotated-box annotations (T10).

FAIR1M ships very large RGB scenes (1 000–10 000 px) plus one XML per scene whose
objects are quadrilaterals. A training sample is therefore a **tile**, not an
image, and the manifest row carries:

* ``group_id`` = source image stem, so ``holdout-manifest`` can carve a
  validation split without putting two tiles of one scene on both sides;
* ``window`` = ``[x, y, w, h]`` in source pixels, with the **clamped** edge rule
  (``data.tiling.tile_origins_clamped``): detection must not pad a band of empty
  pixels next to real objects (FAIR1M plan §5.2);
* ``annotation_path`` = one derived JSON per source image holding every object;
* ``instance_count`` / ``ignored_instance_count`` = how many objects are visible
  enough to train on in this tile, and how many were cut by the tile edge and
  must be treated as ignore rather than background.

Data layout (overridable, because the mirror layout is not the official one)::

    raw/fair1m/images/{train,val}/<stem>.png|.tif     # or a flat images/
    raw/fair1m/labelXml[/<split>]/<stem>.xml
    derived/fair1m/obb/<stem>.json                    # written by this builder

The official 2.0 train set is identical to the 1.0 train set, so the mirror that
carries 1.0 train is usable; the manifest still records what it actually read via
``source_notes`` in the task config and the run tags. Official 2.0 val/test are
not required for the internal ``train-holdout`` protocol.

Regression guard: FAIR1M polygons are closed rings of **five** points (the first
vertex repeated). All 14 756 published train annotations were checked; the
closing duplicate is dropped and anything that is not a quadrilateral after that
is counted and skipped, never silently accepted.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from ...detection.classes import FAIR1M_CLASSES, UnknownClassError, class_id
from ...detection.geometry import Instance, visible_fraction
from ...registry import TaskSpec
from ..image_header import ImageHeaderError, image_size
from ..sequence import GeoSequenceSample
from ..tiling import tile_origins_clamped, valid_fraction
from .sen1floods11 import ManifestBuildError, file_sha256

DEFAULT_CROP = 1024
# 200 px overlap, as locked in the plan; the clamped rule makes the interior
# stride 824 and the last tile flush with the edge.
DEFAULT_STRIDE = 824
DEFAULT_KEEP_RATIO = 0.7
# FAIR1M scenes are 0.3–0.8 m. A single declared value is required because the
# annotations carry no per-image GSD; 0.5 is a declared choice, not a measurement,
# and every run records gsd_source=declared (plan §6.1).
DEFAULT_GSD_M = 0.5
SPLIT_MAP = {"train": "train", "val": "test"}
IMAGE_SUFFIXES = (".png", ".PNG", ".tif", ".tiff", ".TIF", ".TIFF", ".jpg")


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


@dataclass
class BuildStats:
    """Counts that make the conversion auditable without opening annotations."""

    images: int = 0
    malformed_objects: int = 0
    unknown_classes: list[str] = field(default_factory=list)
    instances: int = 0
    tiles: int = 0
    kept_assignments: int = 0
    ignored_assignments: int = 0
    empty_tiles: int = 0
    stale_annotations: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "images": self.images,
            "instances": self.instances,
            "tiles": self.tiles,
            "kept_instance_assignments": self.kept_assignments,
            "ignored_instance_assignments": self.ignored_assignments,
            "empty_tiles": self.empty_tiles,
            "stale_annotations_rewritten": self.stale_annotations,
            "malformed_objects": self.malformed_objects,
            "unknown_classes": sorted(set(self.unknown_classes)),
        }


def parse_fair1m_xml(path: Path) -> tuple[list[Instance], int]:
    """Read one annotation file, returning ``(instances, malformed_count)``.

    A quadrilateral is stored as five points with the first repeated; the
    duplicate is dropped. Objects that are not quadrilaterals after that are
    counted and skipped — the plan requires reporting them, not repairing them
    silently.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ManifestBuildError(f"{path}: unreadable XML ({error})") from error
    instances: list[Instance] = []
    malformed = 0
    for obj in root.iter("object"):
        name = (obj.findtext("possibleresult/name") or "").strip()
        points: list[tuple[float, float]] = []
        for point in obj.findall("points/point"):
            text = (point.text or "").strip()
            try:
                x, y = (float(value) for value in text.split(","))
            except ValueError:
                points = []
                break
            points.append((x, y))
        if (
            len(points) == 5
            and abs(points[0][0] - points[-1][0]) < 1e-6
            and abs(points[0][1] - points[-1][1]) < 1e-6
        ):
            points = points[:4]
        if len(points) != 4:
            malformed += 1
            continue
        instances.append(Instance(class_name=name, points=tuple(points)))
    return instances, malformed


def annotation_path_for(derived_root: Path, stem: str) -> Path:
    return derived_root / "obb" / f"{stem}.json"


def write_annotation(
    path: Path, *, image: str, width: int, height: int, instances: list[Instance]
) -> None:
    """Persist the per-image objects the workers filter per tile."""
    payload = {
        "image": image,
        "width": width,
        "height": height,
        "classes": list(FAIR1M_CLASSES),
        "instances": [
            {
                "class_id": class_id(instance.class_name),
                "class_name": instance.class_name,
                "points": [[round(x, 3), round(y, 3)] for x, y in instance.points],
            }
            for instance in instances
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _split_dirs(images_root: Path) -> list[tuple[str, Path]]:
    found = [(name, images_root / name) for name in SPLIT_MAP if (images_root / name).is_dir()]
    return found or [("train", images_root)]


def _xml_for(xml_root: Path, official: str, stem: str) -> Path | None:
    for candidate in (xml_root / official / f"{stem}.xml", xml_root / f"{stem}.xml"):
        if candidate.is_file():
            return candidate
    return None


def _tile_counts(
    instances: list[Instance],
    origins_x: list[int],
    origins_y: list[int],
    crop: int,
    keep_ratio: float,
) -> tuple[dict[tuple[int, int], list[int]], int]:
    """Count kept/ignored objects per tile origin.

    Returns ``{(x, y): [kept, ignored]}`` plus the total ignored count. Only tiles
    whose window can intersect the object's bounding box are evaluated, which is
    what keeps the build linear in the number of objects rather than in
    objects × tiles.
    """
    counts: dict[tuple[int, int], list[int]] = {}
    ignored_total = 0
    for instance in instances:
        x0, y0, x1, y1 = instance.bbox
        polygon = instance.polygon
        for x in origins_x:
            if x > x1 or x + crop < x0:
                continue
            for y in origins_y:
                if y > y1 or y + crop < y0:
                    continue
                ratio = visible_fraction(polygon, x, y, x + crop, y + crop)
                if ratio <= 0:
                    continue
                bucket = counts.setdefault((x, y), [0, 0])
                if ratio >= keep_ratio:
                    bucket[0] += 1
                else:
                    bucket[1] += 1
                    ignored_total += 1
    return counts, ignored_total


def build_fair1m_manifest(
    data_root: Path, task_spec: TaskSpec, stats: BuildStats | None = None
) -> list[GeoSequenceSample]:
    """Build one manifest row per tile for a FAIR1M split tree."""
    images_root = Path(
        os.environ.get("RS_EVAL_FAIR1M_IMAGES", data_root / "raw" / "fair1m" / "images")
    )
    xml_root = Path(os.environ.get("RS_EVAL_FAIR1M_XML", data_root / "raw" / "fair1m" / "labelXml"))
    derived_root = Path(os.environ.get("RS_EVAL_FAIR1M_DERIVED", data_root / "derived" / "fair1m"))
    crop = _env_int("RS_EVAL_FAIR1M_CROP", DEFAULT_CROP)
    stride = _env_int("RS_EVAL_FAIR1M_STRIDE", DEFAULT_STRIDE)
    keep_ratio = _env_float("RS_EVAL_FAIR1M_KEEP_RATIO", DEFAULT_KEEP_RATIO)
    gsd = _env_float("RS_EVAL_FAIR1M_GSD", DEFAULT_GSD_M)
    rebuild = os.environ.get("RS_EVAL_FAIR1M_REBUILD", "0") not in ("", "0", "false", "False")
    if crop <= 0 or stride <= 0 or not (0 < keep_ratio <= 1):
        raise ManifestBuildError("FAIR1M crop/stride must be positive and keep ratio in (0, 1]")
    if not images_root.is_dir():
        raise ManifestBuildError(f"missing FAIR1M images: {images_root}")
    if not xml_root.is_dir():
        raise ManifestBuildError(f"missing FAIR1M annotations: {xml_root}")

    stats = stats if stats is not None else BuildStats()
    samples: list[GeoSequenceSample] = []
    for official, split in ((name, SPLIT_MAP[name]) for name, _ in _split_dirs(images_root)):
        images = sorted(
            path
            for path in (images_root / official).iterdir()
            if path.is_file() and path.suffix in IMAGE_SUFFIXES
        )
        if not images:
            continue
        for image_path in images:
            stem = image_path.stem
            xml_path = _xml_for(xml_root, official, stem)
            if xml_path is None:
                raise ManifestBuildError(
                    f"missing annotation for {image_path.name} (looked in {xml_root})"
                )
            try:
                width, height = image_size(image_path)
            except ImageHeaderError as error:
                raise ManifestBuildError(str(error)) from error
            instances, malformed = parse_fair1m_xml(xml_path)
            for instance in instances:
                try:
                    class_id(instance.class_name)
                except UnknownClassError:
                    stats.unknown_classes.append(instance.class_name)
            stats.images += 1
            stats.malformed_objects += malformed
            stats.instances += len(instances)

            annotation = annotation_path_for(derived_root, stem)
            stale = False
            if not rebuild and annotation.is_file():
                stored = json.loads(annotation.read_text(encoding="utf-8"))
                stale = (int(stored.get("width", 0)), int(stored.get("height", 0))) != (
                    width,
                    height,
                )
            if rebuild or stale or not annotation.is_file():
                write_annotation(
                    annotation,
                    image=image_path.name,
                    width=width,
                    height=height,
                    instances=instances,
                )
                if stale:
                    # The image behind an id was replaced (val pairing fix), so the
                    # cached size no longer describes it. Repairing here keeps the
                    # file consistent with the pixels; the worker also cross-checks
                    # the size against valid_fraction and would refuse a stale file
                    # (review 2026-10-29).
                    stats.stale_annotations += 1
            digest = file_sha256(annotation)
            origins_x = tile_origins_clamped(width, crop, stride)
            origins_y = tile_origins_clamped(height, crop, stride)
            counts, _ = _tile_counts(instances, origins_x, origins_y, crop, keep_ratio)
            for origin_y in origins_y:
                for origin_x in origins_x:
                    kept, ignored = counts.get((origin_x, origin_y), (0, 0))
                    stats.tiles += 1
                    stats.kept_assignments += kept
                    stats.ignored_assignments += ignored
                    if kept == 0:
                        stats.empty_tiles += 1
                    samples.append(
                        GeoSequenceSample(
                            task_id=task_spec.task_id,
                            sample_id=f"{stem}_x{origin_x:05d}_y{origin_y:05d}",
                            group_id=stem,
                            split=split,
                            image_paths=[image_path.relative_to(data_root).as_posix()],
                            annotation_path=annotation.relative_to(data_root).as_posix(),
                            instance_count=kept,
                            ignored_instance_count=ignored,
                            sensor=task_spec.sensor,
                            bands=list(task_spec.bands),
                            timestamps=[0],
                            temporal_mask=[True],
                            crs=None,
                            transform=None,
                            gsd_m=gsd,
                            checksum=digest,
                            window=[origin_x, origin_y, crop, crop],
                            valid_fraction=round(
                                valid_fraction(width, height, crop, stride, (origin_x, origin_y)), 6
                            ),
                        )
                    )
    if not samples:
        raise ManifestBuildError("no FAIR1M tiles produced; are images and annotations present?")
    return samples
