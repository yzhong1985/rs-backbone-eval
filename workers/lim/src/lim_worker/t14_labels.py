"""T14 (SpaceNet 7) change masks: a pixel is 1 when a building covers it at t2 but not at t1.

The footprints are the ``*_Buildings.geojson`` files in each AOI's
``labels_match_pix`` directory. The cloud/unusable (UDM) polygons are the
``*_UDM.geojson`` files in the AOI's ``labels`` directory, one per monthly image.
Footprint coordinates are already in pixel units of the monthly raster, so they
are rasterised with an identity transform. UDM coordinates are lon/lat in CRS84
(``urn:ogc:def:crs:OGC:1.3:CRS84``). They are projected to EPSG:3857 and then
converted to pixels with the row's ``transform``, so the UDM must be in CRS84 or
the file is rejected. A pixel is cloud when it lies in the UDM of t1 or of t2;
cloud pixels are written as 255 (``CLOUD_VALUE``) even where a building is
present. A missing UDM file counts as no cloud and is reported in the
``udm_missing`` count. A UDM polygon whose bounding box misses the image window
is reported in ``udm_outside``, and a non-empty UDM with every polygon outside
the window logs a warning. The masks are written as uint8 GeoTIFFs at
``row["label_path"]`` under the data root, with the same grid and CRS as the
image.

Run in the LIM worker env (it needs rasterio), not in the control plane::

    uv run --project workers/lim python -m lim_worker.t14_labels \\
        --data-root DATA_ROOT --manifest MANIFEST.jsonl

Existing masks are kept unless ``--force`` is given. Masks written before the
UDM step have no 255 values and need ``--force`` to pick up the cloud pixels.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

RAW_TRAIN = Path("raw/spacenet7/train")
LABEL_PREFIX = "derived/spacenet7-change/"
TASK_ID = "T14"
CLOUD_VALUE = 255  # ignored by the loss and the metrics (t14.IGNORE_INDEX)
UDM_CRS = "urn:ogc:def:crs:OGC:1.3:CRS84"
EARTH_RADIUS_M = 6378137.0  # sphere radius used by EPSG:3857 (spherical Web Mercator)

logger = logging.getLogger(__name__)


def footprint_path(data_root: Path, row: dict, image_path: str) -> Path:
    """Buildings geojson that matches one monthly image of the row's AOI."""
    stem = Path(image_path).stem
    return (
        data_root / RAW_TRAIN / row["group_id"] / "labels_match_pix" / f"{stem}_Buildings.geojson"
    )


def udm_path(data_root: Path, row: dict, image_path: str) -> Path:
    """UDM (cloud/unusable) geojson that matches one monthly image of the row's AOI."""
    stem = Path(image_path).stem
    return data_root / RAW_TRAIN / row["group_id"] / "labels" / f"{stem}_UDM.geojson"


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    """EPSG:3857 (spherical Web Mercator) coordinates in metres of a lon/lat point in degrees.

    ``x = R * lon`` and ``y = R * ln(tan(pi/4 + lat/2))``, with the angles in radians and
    ``R = 6378137``. This is the EPSG:3857 projection, so it agrees with pyproj's
    ``OGC:CRS84 -> EPSG:3857`` transform to well under a millimetre.
    """
    x = EARTH_RADIUS_M * math.radians(lon)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def lonlat_to_pixel(lon: float, lat: float, transform: list[float]) -> tuple[float, float]:
    """Fractional (col, row) of a lon/lat point on a north-up grid with affine ``transform``.

    ``transform`` is ``[a, b, c, d, e, f]`` in EPSG:3857 metres, the row's ``transform`` field.
    """
    a, b, c, d, e, f = transform
    if b != 0 or d != 0:
        raise ValueError(f"rotated grids are not supported (transform {transform})")
    x, y = lonlat_to_web_mercator(lon, lat)
    return (x - c) / a, (y - f) / e


def _flat_positions(coords: list) -> Iterator[tuple[float, float]]:
    """Every ``(x, y)`` position of a nested GeoJSON coordinate list."""
    if coords and isinstance(coords[0], (int, float)):
        yield coords[0], coords[1]
        return
    for item in coords:
        yield from _flat_positions(item)


def _map_positions(coords: list, fn) -> list:
    """Nested GeoJSON coordinate list with each ``(x, y)`` position replaced by ``fn(x, y)``."""
    if coords and isinstance(coords[0], (int, float)):
        return list(fn(coords[0], coords[1]))
    return [_map_positions(item, fn) for item in coords]


def _geometries(path: Path) -> list[dict]:
    """Geometries of the features in a geojson file (features without geometry are skipped)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [f["geometry"] for f in data.get("features", []) if f.get("geometry")]


def _rasterize(geometries: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of geometries given in pixel coordinates; no geometry gives all False."""
    if not geometries:
        return np.zeros(shape, dtype=bool)
    burned = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape,
        transform=Affine.identity(),
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)


def _udm_pixel_geometries(path: Path, transform: list[float]) -> list[dict]:
    """UDM polygons of a geojson, converted from CRS84 lon/lat to pixel coordinates.

    A file with no geometry needs no CRS and gives an empty list. A file with geometry must
    declare CRS84; any other CRS, or none, raises ``ValueError``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    features = [f for f in data.get("features", []) if f.get("geometry")]
    if not features:
        return []
    name = ((data.get("crs") or {}).get("properties") or {}).get("name")
    if name != UDM_CRS:
        raise ValueError(f"UDM {path} has crs {name!r}; expected {UDM_CRS} (lon/lat)")

    def to_pixel(lon: float, lat: float) -> tuple[float, float]:
        return lonlat_to_pixel(lon, lat, transform)

    geometries = []
    for feature in features:
        geom = feature["geometry"]
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"UDM {path} has unsupported geometry {geom.get('type')!r}")
        geometries.append(
            {"type": geom["type"], "coordinates": _map_positions(geom["coordinates"], to_pixel)}
        )
    return geometries


def _outside_window(geom: dict, shape: tuple[int, int]) -> bool:
    """True when the bounding box of a pixel-coordinate geometry misses the image window."""
    height, width = shape
    positions = list(_flat_positions(geom["coordinates"]))
    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    return max(xs) <= 0 or min(xs) >= width or max(ys) <= 0 or min(ys) >= height


def footprint_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of building pixels; an empty footprint file gives all False."""
    if not path.is_file():
        raise FileNotFoundError(f"missing building footprints {path}")
    return _rasterize(_geometries(path), shape)


def cloud_mask(data_root: Path, row: dict, shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """Boolean mask of ``t1 UDM | t2 UDM``, the absent UDM count, and the outside polygon count.

    A missing UDM file adds no cloud pixels. A UDM polygon whose bounding box misses the
    image window adds no pixels either, and is counted as outside. The caller reports both
    counts. A non-empty UDM with every polygon outside logs a warning.
    """
    cloud = np.zeros(shape, dtype=bool)
    missing = 0
    outside = 0
    for image_path in row["image_paths"]:
        path = udm_path(data_root, row, image_path)
        if not path.is_file():
            missing += 1
            continue
        polygons = _udm_pixel_geometries(path, row["transform"])
        n_outside = sum(_outside_window(geom, shape) for geom in polygons)
        outside += n_outside
        if polygons and n_outside == len(polygons):
            logger.warning(
                "T14 UDM %s: all %d polygon(s) lie outside the image window %s",
                path,
                len(polygons),
                shape,
            )
        cloud |= _rasterize(polygons, shape)
    return cloud, missing, outside


def change_mask(data_root: Path, row: dict) -> tuple[np.ndarray, int, int]:
    """uint8 mask: 1 where a building is present at t2 and absent at t1, 255 on cloud.

    Returns ``(mask, udm_missing, udm_outside)``, the absent UDM file count and the count of
    UDM polygons outside the image window.
    """
    _, _, width, height = (int(v) for v in row["window"])
    shape = (height, width)
    t1, t2 = row["image_paths"]
    before = footprint_mask(footprint_path(data_root, row, t1), shape)
    after = footprint_mask(footprint_path(data_root, row, t2), shape)
    cloud, missing, outside = cloud_mask(data_root, row, shape)
    mask = (after & ~before).astype(np.uint8)
    mask[cloud] = CLOUD_VALUE
    return mask, missing, outside


def write_mask(data_root: Path, row: dict, mask: np.ndarray) -> None:
    """Write one-band uint8 GeoTIFF on the image grid (CRS and affine from the row)."""
    out = data_root / row["label_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    a, b, c, d, e, f = row["transform"]
    height, width = mask.shape
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=row["crs"],
        transform=Affine(a, b, c, d, e, f),
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)


def build_masks(data_root: Path, manifest: Path, force: bool = False) -> dict[str, int]:
    """Write every missing change mask named by the T14 rows of ``manifest``.

    ``udm_missing`` counts absent UDM files over the written masks (one per
    image that has no UDM). ``udm_outside`` counts UDM polygons whose bounding box misses
    the image window. ``cloud_px`` counts the 255 pixels written.
    """
    counts = {"written": 0, "skipped": 0, "udm_missing": 0, "udm_outside": 0, "cloud_px": 0}
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") != TASK_ID or row["label_path"] in seen:
                continue
            seen.add(row["label_path"])
            if (data_root / row["label_path"]).exists() and not force:
                counts["skipped"] += 1
                continue
            mask, missing, outside = change_mask(data_root, row)
            write_mask(data_root, row, mask)
            counts["written"] += 1
            counts["udm_missing"] += missing
            counts["udm_outside"] += outside
            counts["cloud_px"] += int((mask == CLOUD_VALUE).sum())
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="rewrite existing masks")
    args = parser.parse_args(argv)
    counts = build_masks(args.data_root, args.manifest, force=args.force)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
