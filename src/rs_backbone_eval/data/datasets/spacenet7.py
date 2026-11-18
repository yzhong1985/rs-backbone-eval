"""SpaceNet 7 (MUDS) change manifest (T14): consecutive Planet monthly pairs.

Layout under ``data_root/raw/spacenet7`` (the extracted train archive)::

    train/<AOI>/images/global_monthly_<YYYY>_<MM>_mosaic_<AOI>.tif   RGBA uint8 (alpha = validity)
    train/<AOI>/labels_match_pix/..._Buildings.geojson               polygons, pixel coordinates
    test_public/<AOI>/images_masked/                                 no labels, not used here

Each AOI contributes one sample per pair of consecutive months (t1, t2). The
change label is derived from the two building footprints and written by the
label step (``derived/spacenet7-change/<AOI>/<t1>_<t2>.tif``); this builder only
records the path. ``test_public`` has no labels, so validation and test are
AOI-level holdouts carved from the train archive with ``SPLIT_SEED``.

The rasters are parsed from their TIFF tags (no rasterio in the control plane).
Their georeference is EPSG:3857 with a north-up affine.
"""

from __future__ import annotations

import hashlib
import random
import re
import struct
from pathlib import Path

from ...registry import TaskSpec
from ..sequence import GeoSequenceSample
from .fair1m import pixel_size_m
from .sen1floods11 import ManifestBuildError, file_sha256

RAW_DIR = "spacenet7"
TRAIN_DIR = "train"
LABEL_DIR = "derived/spacenet7-change"
VALIDATION_AOIS = 6
TEST_AOIS = 6
SPLIT_SEED = 17
GSD_M = 4.78  # declared Planet monthly basemap pixel size in EPSG:3857 metres
_NAME = re.compile(r"^global_monthly_(\d{4})_(\d{2})_mosaic_(.+)\.tif$")

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d"}
_GEO_CRS_KEYS = (3072, 2048)  # ProjectedCSTypeGeoKey, GeographicTypeGeoKey


def read_tags(path: Path) -> dict[int, tuple[int, tuple]]:
    """``{tag: (type, values)}`` from the first IFD of a classic little/big-endian TIFF."""
    with path.open("rb") as handle:
        head = handle.read(8)
        if head[:2] == b"II":
            order = "<"
        elif head[:2] == b"MM":
            order = ">"
        else:
            raise ManifestBuildError(f"{path.name}: not a TIFF")
        magic, offset = struct.unpack(order + "HI", head[2:8])
        if magic != 42:
            raise ManifestBuildError(f"{path.name}: BigTIFF or unknown TIFF variant ({magic})")
        handle.seek(offset)
        (count,) = struct.unpack(order + "H", handle.read(2))
        tags: dict[int, tuple[int, tuple]] = {}
        for _ in range(count):
            tag, kind, n, raw = struct.unpack(order + "HHI4s", handle.read(12))
            if kind not in _TYPE_SIZE or kind not in _TYPE_FMT:
                continue
            fmt = _TYPE_FMT[kind]
            size = _TYPE_SIZE[kind] * n
            if size <= 4:
                data = raw[:size]
            else:
                (where,) = struct.unpack(order + "I", raw)
                pos = handle.tell()
                handle.seek(where)
                data = handle.read(size)
                handle.seek(pos)
            tags[tag] = (kind, struct.unpack(order + fmt * n, data))
    return tags


def read_grid(path: Path) -> tuple[tuple[int, int], str, list[float]]:
    """``((width, height), "EPSG:<code>", affine)`` from a Planet monthly raster."""
    tags = read_tags(path)
    for needed in (256, 257, 33550, 33922, 34735):
        if needed not in tags:
            raise ManifestBuildError(f"{path.name}: missing TIFF tag {needed}")
    width, height = tags[256][1][0], tags[257][1][0]
    scale = tags[33550][1]
    tie = tags[33922][1]
    if tie[0] != 0 or tie[1] != 0 or len(scale) < 2:
        raise ManifestBuildError(f"{path.name}: expects a single tie point at the origin")
    keys = tags[34735][1]
    epsg = None
    for index in range(keys[3]):
        key_id, location, _count, value = keys[4 + 4 * index : 8 + 4 * index]
        if location == 0 and key_id in _GEO_CRS_KEYS:
            epsg = value
    if epsg != 3857:
        raise ManifestBuildError(f"{path.name}: expected EPSG:3857, found {epsg}")
    affine = [scale[0], 0.0, tie[3], 0.0, -scale[1], tie[4]]
    return (width, height), f"EPSG:{epsg}", affine


def same_grid(a: tuple, b: tuple) -> bool:
    """Equal size and CRS, affine equal to 1e-6 m (writer rounding differs by ~1e-12)."""
    return (
        a[0] == b[0]
        and a[1] == b[1]
        and all(abs(x - y) <= 1e-6 for x, y in zip(a[2], b[2], strict=True))
    )


def month_index(year: int, month: int) -> int:
    return year * 12 + month - 1


def aoi_split(aois: list[str]) -> dict[str, str]:
    """AOI-level split: ``VALIDATION_AOIS`` + ``TEST_AOIS`` drawn once, the rest train."""
    if len(aois) < VALIDATION_AOIS + TEST_AOIS + 1:
        raise ManifestBuildError(f"only {len(aois)} AOIs, cannot hold out validation and test")
    chosen = random.Random(SPLIT_SEED).sample(sorted(aois), VALIDATION_AOIS + TEST_AOIS)
    split = {aoi: "train" for aoi in aois}
    for aoi in chosen[:VALIDATION_AOIS]:
        split[aoi] = "validation"
    for aoi in chosen[VALIDATION_AOIS:]:
        split[aoi] = "test"
    return split


def aoi_frames(aoi_dir: Path, data_root: Path) -> list[tuple[int, str, Path]]:
    """``[(month_index, YYYY-MM, path)]`` for one AOI, sorted by month."""
    aoi = aoi_dir.name
    frames: list[tuple[int, str, Path]] = []
    for path in sorted((aoi_dir / "images").glob("*.tif")):
        match = _NAME.match(path.name)
        if match is None or match.group(3) != aoi:
            raise ManifestBuildError(f"{aoi}: unexpected image name {path.name}")
        year, month = int(match.group(1)), int(match.group(2))
        frames.append((month_index(year, month), f"{year:04d}-{month:02d}", path))
    frames.sort(key=lambda item: item[0])
    return frames


def build_spacenet7_manifest(data_root: Path, task_spec: TaskSpec) -> list[GeoSequenceSample]:
    train_root = data_root / "raw" / RAW_DIR / TRAIN_DIR
    if not train_root.is_dir():
        raise ManifestBuildError(f"missing {train_root}")
    aoi_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    split = aoi_split([p.name for p in aoi_dirs])
    digests: dict[Path, str] = {}
    samples: list[GeoSequenceSample] = []
    for aoi_dir in aoi_dirs:
        frames = aoi_frames(aoi_dir, data_root)
        if len(frames) < 2:
            raise ManifestBuildError(f"{aoi_dir.name}: fewer than two monthly images")
        grids = {path: read_grid(path) for _, _, path in frames}
        first = next(iter(grids.values()))
        if any(not same_grid(grid, first) for grid in grids.values()):
            raise ManifestBuildError(f"{aoi_dir.name}: monthly rasters do not share one grid")
        (width, height), crs, affine = first
        ground = pixel_size_m(crs, affine, height)
        for (i1, t1, p1), (i2, t2, p2) in zip(frames, frames[1:], strict=False):
            if i2 - i1 != 1:
                # Coverage has gaps (about a third of the AOIs). A pair across a gap
                # would not be a one-month change, so it is not a sample.
                continue
            for path in (p1, p2):
                if path not in digests:
                    digests[path] = file_sha256(path)
            rel = [p.relative_to(data_root).as_posix() for p in (p1, p2)]
            label = f"{LABEL_DIR}/{aoi_dir.name}/{t1}_{t2}.tif"
            samples.append(
                GeoSequenceSample(
                    task_id=task_spec.task_id,
                    sample_id=f"{aoi_dir.name}_{t1}_{t2}",
                    group_id=aoi_dir.name,
                    split=split[aoi_dir.name],
                    image_paths=rel,
                    label_path=label,
                    sensor=task_spec.sensor,
                    bands=list(task_spec.bands),
                    timestamps=[t1, t2],
                    temporal_mask=[True, True],
                    crs=crs,
                    transform=list(affine),
                    gsd_m=GSD_M,
                    gsd_measured_m=ground,
                    checksum=file_sha256_pair(digests[p1], digests[p2]),
                    window=[0, 0, width, height],
                )
            )
    return samples


def file_sha256_pair(first: str, second: str) -> str:
    """Fingerprint of a pair: sha256 over the two monthly image digests in order."""

    return hashlib.sha256(f"{first}|{second}".encode("ascii")).hexdigest()
