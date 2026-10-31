#!/usr/bin/env python3
"""Read one AlphaEarth Foundations (AEF) annual embedding window and report it.

AEF is not an encoder: the public product is a precomputed 64-band int8 COG at
10 m on UTM grids. This probe answers the P1 question from
``docs/models/AEF与LIM下游任务比较可行性_2026-10-30.md`` — can we resolve a point
to the right COG, read the window, dequantize it and record enough identity to
reuse it — without committing to a task or a model.

Needs rasterio/GDAL, so run it in a worker environment:

    workers/prithvi/.venv/bin/python scripts/aef_probe.py --tif /geodata/.../799.tif --year 2021

The script never writes into the repository: outputs go to ``--out`` (default
``/tmp/opencode/aef``) and carry the source URI, CRS, transform and window so a
later manifest can hash them.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

BUCKET = "https://storage.googleapis.com/alphaearth_foundations"
PREFIX = "satellite_embedding/v1/annual"
NODATA = -128
CELL_M = 10.0


def utm_zone(lon: float, lat: float) -> tuple[int, str]:
    zone = int((lon + 180.0) // 6.0) + 1
    return zone, ("N" if lat >= 0 else "S")


def zone_epsg(zone: int, hemisphere: str) -> str:
    return f"EPSG:{32600 + zone if hemisphere == 'N' else 32700 + zone}"


def list_cogs(year: int, zone_name: str) -> list[str]:
    """Every COG key under one year/zone, following the XML listing pages."""
    prefix = f"{PREFIX}/{year}/{zone_name}/"
    keys: list[str] = []
    marker: str | None = None
    for _ in range(50):
        url = f"{BUCKET}?prefix={prefix}&max-keys=1000"
        if marker:
            url += "&marker=" + urllib.parse.quote(marker)
        with urllib.request.urlopen(url, timeout=60) as response:
            xml = response.read().decode("utf-8", "replace")
        page = re.findall(r"<Key>(.*?)</Key>", xml)
        keys.extend(key for key in page if key.endswith(".tiff"))
        if "<IsTruncated>true" not in xml:
            break
        marker = page[-1]
    return keys


def _bounds_of(key: str) -> tuple[str, tuple[float, float, float, float]] | None:
    """Normalised bounds from the transform.

    AEF COGs are stored bottom-up (positive y scale), and ``dataset.bounds``
    assumes a north-up transform, so it reports bottom > top. Recompute from the
    affine and take min/max instead of trusting the property.
    """
    try:
        with rasterio.open("/vsicurl/" + f"{BUCKET}/{key}") as dataset:
            transform = dataset.transform
            xs = [transform.c, transform.c + transform.a * dataset.width]
            ys = [transform.f, transform.f + transform.e * dataset.height]
            return key, (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None


def resolve_cog(keys: list[str], x: float, y: float, workers: int = 8) -> tuple[str, tuple]:
    """The COG whose bounds contain ``(x, y)`` in the zone CRS."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_bounds_of, keys):
            if result is None:
                continue
            key, (left, bottom, right, top) = result
            if left <= x <= right and bottom <= y <= top:
                return key, (left, bottom, right, top)
    raise SystemExit(f"no COG covers ({x:.1f}, {y:.1f}) among {len(keys)} candidates")


def target_point(args) -> tuple[float, float, str]:
    """Lon/lat from ``--tif`` (centre pixel) or from ``--lon/--lat``."""
    if args.tif:
        with rasterio.open(args.tif) as dataset:
            if dataset.crs is None:
                raise SystemExit(f"{args.tif} has no CRS")
            centre = dataset.transform * (dataset.width / 2, dataset.height / 2)
            lon, lat = warp_transform(dataset.crs, "EPSG:4326", [centre[0]], [centre[1]])
        return float(lon[0]), float(lat[0]), str(dataset.crs)
    return args.lon, args.lat, "EPSG:4326"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", type=Path, help="a georeferenced image; its centre is used")
    parser.add_argument("--lon", type=float, default=139.6920)
    parser.add_argument("--lat", type=float, default=35.5203)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--zone", help="override the resolved UTM zone, e.g. 54N")
    parser.add_argument("--radius-m", type=float, default=320.0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/opencode/aef"))
    args = parser.parse_args()

    started = time.time()
    lon, lat, source_crs = target_point(args)
    zone, hemisphere = utm_zone(lon, lat)
    zone_name = args.zone or f"{zone}{hemisphere}"
    epsg = zone_epsg(int(re.sub(r"\D", "", zone_name)), zone_name[-1])
    easting, northing = warp_transform("EPSG:4326", epsg, [lon], [lat])
    easting, northing = float(easting[0]), float(northing[0])
    print(
        f"point lon={lon:.5f} lat={lat:.5f} (from {source_crs}) -> {epsg} "
        f"E={easting:.1f} N={northing:.1f}, zone {zone_name}",
        flush=True,
    )

    keys = list_cogs(args.year, zone_name)
    print(f"{len(keys)} COGs under {args.year}/{zone_name}", flush=True)
    key, bounds = resolve_cog(keys, easting, northing)
    print(f"resolved {key} bounds={bounds}", flush=True)

    with rasterio.open("/vsicurl/" + f"{BUCKET}/{key}") as dataset:
        row, col = dataset.index(easting, northing)
        half = max(1, int(round(args.radius_m / CELL_M)))
        window = rasterio.windows.Window(
            max(col - half, 0), max(row - half, 0), 2 * half + 1, 2 * half + 1
        )
        raw = dataset.read(window=window)
        transform = dataset.window_transform(window)
        print(f"window {raw.shape} at row/col {row}/{col}, crs {dataset.crs}", flush=True)

    valid = raw != NODATA
    values = np.where(valid, raw.astype(np.float32), np.nan)
    values = np.sign(values) * (values / 127.5) ** 2
    values = np.nan_to_num(values, nan=0.0)
    valid_share = float(valid.mean())
    centre = raw[:, raw.shape[1] // 2, raw.shape[2] // 2]

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.year}_{zone_name}_{row}_{col}"
    np.savez_compressed(
        args.out / f"{stem}.npz",
        features=values.astype(np.float32),
        valid=valid,
        transform=np.array(list(transform)[:6], dtype=np.float64),
    )
    report = {
        "source_uri": f"gs://alphaearth_foundations/{key}",
        "https_uri": f"{BUCKET}/{key}",
        "year": args.year,
        "zone": zone_name,
        "crs": str(epsg),
        "bounds": bounds,
        "window_transform": list(transform)[:6],
        "window_shape": list(raw.shape),
        "window_valid_share": valid_share,
        "centre_raw": [int(v) for v in centre],
        "centre_dequantized": [
            round(float(v), 6) for v in values[:, raw.shape[1] // 2, raw.shape[2] // 2]
        ],
        "point": {"lon": lon, "lat": lat, "easting": easting, "northing": northing},
        "seconds": round(time.time() - started, 1),
        "note": "Dequantised with sign(v) * (v / 127.5)^2 after masking -128; "
        "rows are stored bottom-up (positive y scale), so row/col come from "
        "rasterio.index(), not from hand arithmetic.",
    }
    (args.out / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: report[k] for k in ("window_shape", "window_valid_share", "centre_raw", "seconds")},
            ensure_ascii=False,
        )
    )
    print("centre dequantized (first 8):", report["centre_dequantized"][:8])
    print("wrote", args.out / f"{stem}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
