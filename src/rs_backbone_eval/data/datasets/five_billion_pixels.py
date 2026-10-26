"""Five-Billion-Pixels (GF-2) land-cover manifest: an out-of-domain T8 task.

The dataset holds 150 Gaofen-2 multispectral scenes (4 m, 4 bands B/G/R/NIR, 16 bit)
with 24 land-cover classes; ``0`` is unlabeled and maps to the ignore index.
The paper's own test list (30 scenes) is used as the test split; the remaining
scenes are split into train and a project validation subset.

Two properties make this a *cross-domain* test rather than a like-for-like
comparison, and every run records them:

* the sensor is 4 m with no SWIR, finer than the 10-30 m S2/HLS pretraining;
* only four bands exist, so Prithvi (six bands) and AnySat (ten bands) need an
  explicit adapter, while Clay can be told the four wavelengths.

The scenes are huge (roughly 6900x7300 px), so workers must read windows
instead of whole rasters.

GSD: these are the multispectral (MSS) products. The paper describes them as
4 m; 0.8 m is the GF-2 panchromatic band, which this release does not carry.
The RPC ground extent (~32 km over ~6900 rows) agrees with a few metres.

Groups: scenes are grouped by footprint overlap (RPC bounding boxes from
``Coordinate_files/*.rpb``, single-linkage at >5% of the smaller box). Many
of the paper's test scenes revisit a training footprint on another date, so
the official test is not geographically independent; the grouping makes that
measurable instead of hiding it behind one dataset-wide group.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample
from .sen1floods11 import ManifestBuildError, file_sha256

DEFAULT_ROOT = "/geodata/Five-Billion-Pixels"
IMAGE_DIR = "Image_16bit_BGRNir"
LABEL_DIR = "Annotation__index"
LABEL_SUFFIX = "_24label.png"
IMAGE_SUFFIX = ".tiff"
VALIDATION_FRACTION = 0.1
# Band order of the 16-bit product, as documented in the readme (B, G, R, NIR).
# The 8-bit NIR/R/G/B product is RGBA with a constant alpha band and carries no
# usable NIR, so the 16-bit product is used instead.
BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
GSD_M = 4.0
COORDINATE_DIR = "Coordinate_files"
# Fraction of the smaller RPC box two scenes must share to be one footprint.
OVERLAP_THRESHOLD = 0.05


def dataset_root() -> Path:
    return Path(os.environ.get("RS_EVAL_FBP_ROOT", DEFAULT_ROOT)).expanduser().resolve()


def _test_scene_names(root: Path) -> set[str]:
    """Scene stems listed under 'Test images for baseline' in the readme."""
    readme = next(iter(root.glob("readme*.txt")), None)
    if readme is None:
        raise ManifestBuildError(f"readme not found under {root}")
    names: set[str] = set()
    listing = False
    for line in readme.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("* test images"):
            listing = True
            continue
        if listing:
            if stripped.startswith("*"):
                break
            # The readme lists the 8-bit names (".tif") while the 16-bit product
            # is stored as ".tiff"; strip either suffix.
            for suffix in (IMAGE_SUFFIX, ".tif"):
                if stripped.endswith(suffix):
                    names.add(stripped.removesuffix(suffix))
                    break
    return names


def rpc_box(path: Path) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) of an RPB file's normalisation domain."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    def value(key: str) -> float:
        match = re.search(key + r"\s*=\s*([-+\d.eE]+)", text)
        if match is None:
            raise ManifestBuildError(f"{key} missing in {path}")
        return float(match.group(1))

    lat, lon = value("latOffset"), value("longOffset")
    lat_scale, lon_scale = value("latScale"), value("longScale")
    return (lat - lat_scale, lat + lat_scale, lon - lon_scale, lon + lon_scale)


def overlap_fraction(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Shared area as a fraction of the smaller box."""
    height = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    width = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    smaller = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
    return height * width / smaller if smaller > 0 else 0.0


def footprint_groups(
    boxes: dict[str, tuple[float, float, float, float]], threshold: float = OVERLAP_THRESHOLD
) -> dict[str, str]:
    """Single-linkage groups of overlapping footprints; the id is the first scene."""
    parent = {name: name for name in boxes}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    names = sorted(boxes)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if overlap_fraction(boxes[first], boxes[second]) > threshold:
                root_a, root_b = find(first), find(second)
                if root_a != root_b:
                    parent[max(root_a, root_b)] = min(root_a, root_b)
    return {name: f"fbp-{find(name)}" for name in names}


def project_validation(
    images: list[str], test_names: set[str], group_of: dict[str, str]
) -> set[str]:
    """Deterministic ~10% validation carve-out made of whole footprint groups.

    Only groups without a test scene are eligible, so validation never shares a
    footprint with train or test. The official test list is left untouched.
    """
    members: dict[str, list[str]] = {}
    for scene in images:
        if scene not in test_names:
            members.setdefault(group_of[scene], []).append(scene)
    tainted = {group_of[scene] for scene in images if scene in test_names}
    eligible = sorted(group for group in members if group not in tainted)
    target = round(VALIDATION_FRACTION * sum(len(scenes) for scenes in members.values()))
    chosen: set[str] = set()
    for index, group in enumerate(eligible):
        if len(chosen) >= target:
            break
        if index % 3 == 0 and len(chosen) + len(members[group]) <= target:
            chosen.update(members[group])
    return chosen


def build_five_billion_pixels_manifest(
    data_root: Path, task_spec: TaskSpec
) -> list[GeoSequenceSample]:
    """Build the T8 manifest. ``data_root`` is ignored; the scene root comes from env."""
    root = dataset_root()
    image_dir = root / IMAGE_DIR
    label_dir = root / LABEL_DIR
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ManifestBuildError(f"Five-Billion-Pixels not found at {root}")

    test_names = _test_scene_names(root)
    images = sorted(path.stem for path in image_dir.glob(f"*{IMAGE_SUFFIX}"))
    if not images:
        raise ManifestBuildError(f"no GF-2 images under {image_dir}")

    coordinate_dir = root / COORDINATE_DIR
    missing_rpc = [name for name in images if not (coordinate_dir / f"{name}.rpb").is_file()]
    if missing_rpc:
        raise ManifestBuildError(
            f"{len(missing_rpc)} scenes lack RPC files for footprint grouping: {missing_rpc[:5]}"
        )
    group_of = footprint_groups({name: rpc_box(coordinate_dir / f"{name}.rpb") for name in images})

    validation_names = project_validation(images, test_names, group_of)
    split_of: dict[str, str] = {}
    for scene in images:
        if scene in test_names:
            split_of[scene] = "test"
        elif scene in validation_names:
            split_of[scene] = "validation"
        else:
            split_of[scene] = "train"

    samples: list[GeoSequenceSample] = []
    missing: list[str] = []
    for name in images:
        image_path = image_dir / f"{name}{IMAGE_SUFFIX}"
        label_path = label_dir / f"{name}{LABEL_SUFFIX}"
        if not label_path.is_file():
            missing.append(name)
            continue
        samples.append(
            GeoSequenceSample(
                task_id=task_spec.task_id,
                sample_id=name,
                group_id=group_of[name],
                split=split_of[name],
                image_paths=[image_path.relative_to(root).as_posix()],
                label_path=label_path.relative_to(root).as_posix(),
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                timestamps=["2016-01-01"],
                temporal_mask=[True],
                crs=None,
                transform=None,
                gsd_m=GSD_M,
                checksum=file_sha256(image_path),
            )
        )
    if missing:
        raise ManifestBuildError(f"{len(missing)} scenes missing labels: {missing[:5]}")
    if not samples:
        raise ManifestBuildError("no samples found")
    return samples
