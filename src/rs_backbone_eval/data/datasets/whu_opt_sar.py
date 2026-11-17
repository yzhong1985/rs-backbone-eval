"""WHU-OPT-SAR optical-branch manifest (T13): GF-1 RGB+NIR land cover, Hubei.

Layout under ``data_root/raw/whu-opt-sar`` (see
``docs/evaluation/WHU_OPT_SAR_接入说明_2026-11-16.md``)::

    optical/<sheet>.tif                 4 x uint8, band order R, G, B, NIR
    lbl/<sheet>.tif                     uint8, values {0, 10, ..., 70}
    The division of the dataset.txt     upstream 80 train / 20 test sheets

Every sheet is 5556 x 3704 px. The rasters carry no georeference; it is decoded
from the 1:50,000 sheet code (``data.whu_sheets``) and written into each row as
a CGCS2000 geographic affine, so a worker can sample AEF on the tile footprint.

Rows are 512 px tiles at stride 384 with the last tile clamped to the edge (no
padding): ``group_id`` is the sheet, so a split never puts two tiles of one
sheet on both sides. The upstream ``test`` sheets are the test split; a fixed
set of ``VALIDATION_SHEETS`` train sheets (drawn once with ``VALIDATION_SEED``)
becomes the validation split. Adjacent sheets share edges, so this is
sheet-level isolation, not spatial independence.

Band order was checked from class means (forest has the highest 4th band, water
the lowest), not taken from metadata: the tif colour interpretation only says
red/green/blue/undefined.
"""

from __future__ import annotations

import random
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample
from ..tiling import crop_from_tile, tiles_clamped
from ..whu_sheets import CRS, TILE_HEIGHT_PX, TILE_WIDTH_PX, decode
from .fair1m import tile_transform
from .sen1floods11 import ManifestBuildError, file_sha256

RAW_DIR = "whu-opt-sar"
DIVISION_FILE = "The division of the dataset.txt"
CROP = 512
STRIDE = 384
GSD_M = 5.0
VALIDATION_SHEETS = 8
VALIDATION_SEED = 17


def read_division(path: Path) -> dict[str, str]:
    """``{sheet: "train" | "test"}`` from the upstream division file."""
    if not path.is_file():
        raise ManifestBuildError(f"missing split file {path}")
    split: str | None = None
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":"):
            split = line[:-1].strip().lower()
            if split not in {"train", "test"}:
                raise ManifestBuildError(f"{path.name}: unknown section {line!r}")
            continue
        if split is None:
            raise ManifestBuildError(f"{path.name}: sheet {line!r} before any section")
        sheet = decode(line).name
        if sheet in out:
            raise ManifestBuildError(f"{path.name}: sheet {sheet} listed twice")
        out[sheet] = split
    return out


def validation_sheets(train: list[str]) -> set[str]:
    """Fixed validation sheets carved from the upstream train list."""
    if len(train) <= VALIDATION_SHEETS:
        raise ManifestBuildError(f"only {len(train)} train sheets, cannot hold out validation")
    return set(random.Random(VALIDATION_SEED).sample(sorted(train), VALIDATION_SHEETS))


def build_whu_opt_sar_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    raw = data_root / "raw" / RAW_DIR
    division = read_division(raw / DIVISION_FILE)
    held_out = validation_sheets([s for s, split in division.items() if split == "train"])
    samples: list[GeoSequenceSample] = []
    for sheet_name, upstream in sorted(division.items()):
        image = raw / "optical" / f"{sheet_name}.tif"
        label = raw / "lbl" / f"{sheet_name}.tif"
        for path in (image, label):
            if not path.is_file():
                raise ManifestBuildError(f"missing {path}")
        split = "validation" if sheet_name in held_out else upstream
        sheet = decode(sheet_name)
        digest = file_sha256(label)
        for x, y in tiles_clamped(TILE_WIDTH_PX, TILE_HEIGHT_PX, CROP, STRIDE):
            samples.append(
                GeoSequenceSample(
                    task_id=task_spec.task_id,
                    sample_id=f"{sheet_name}_x{x:04d}_y{y:04d}",
                    group_id=sheet_name,
                    split=split,
                    image_paths=[image.relative_to(data_root).as_posix()],
                    label_path=label.relative_to(data_root).as_posix(),
                    sensor=task_spec.sensor,
                    bands=list(task_spec.bands),
                    timestamps=[0],
                    temporal_mask=[True],
                    crs=CRS,
                    transform=tile_transform(list(sheet.transform()), x, y),
                    gsd_m=GSD_M,
                    checksum=digest,
                    window=crop_from_tile(CROP, (x, y)),
                )
            )
    return samples
