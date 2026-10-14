"""Five-Billion-Pixels (GF-2) land-cover manifest: an out-of-domain T8 task.

The dataset holds 150 Gaofen-2 scenes (about 0.8 m, 4 bands B/G/R/NIR, 16 bit)
with 24 land-cover classes; ``0`` is unlabeled and maps to the ignore index.
The paper's own test list (30 scenes) is used as the test split; the remaining
scenes are split into train and a project validation subset.

Two properties make this a *cross-domain* test rather than a like-for-like
comparison, and every run records them:

* the sensor is 0.8 m with no SWIR, far from the 10-30 m S2/HLS pretraining;
* only four bands exist, so Prithvi (six bands) and AnySat (ten bands) need an
  explicit adapter, while Clay can be told the four wavelengths.

The scenes are huge (roughly 6900x7300 px), so workers must read windows
instead of whole rasters.
"""

from __future__ import annotations

import os
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
GSD_M = 0.8


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

    train_names = [scene for scene in images if scene not in test_names]
    # Deterministic 10% project validation carve-out from the training scenes.
    step = max(1, round(1 / VALIDATION_FRACTION))
    validation_names = set(train_names[::step])
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
                group_id=name.split("_", 1)[0],
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
