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
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from rs_backbone_eval.data.aef_index import (
    BUCKET,
    CELL_M,
    NODATA,
    AefIndexError,
    build_index,
    find_cog,
    index_path,
    load_index,
)


def utm_zone(lon: float, lat: float) -> tuple[int, str]:
    zone = int((lon + 180.0) // 6.0) + 1
    return zone, ("N" if lat >= 0 else "S")


def zone_epsg(zone: int, hemisphere: str) -> str:
    return f"EPSG:{32600 + zone if hemisphere == 'N' else 32700 + zone}"


def resolve(index_dir: Path, year: int, zone_name: str, x: float, y: float) -> dict:
    """Tile entry for ``(x, y)``; builds the index once if it is missing."""
    path = index_path(index_dir, year, zone_name)
    try:
        index = load_index(path)
    except AefIndexError:
        print(
            f"no index at {path}; scanning once (slow) — build it with scripts/aef_index.py",
            flush=True,
        )
        index = build_index(year, zone_name, index_dir)
    entry = find_cog(index, x, y)
    if entry is None:
        raise SystemExit(
            f"no COG covers ({x:.1f}, {y:.1f}) among {index.get('cog_count', 0)} in {path}"
        )
    return entry


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
    parser.add_argument(
        "--index", type=Path, default=Path("/geodata/rs-eval/data/derived/aef/index")
    )
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

    entry = resolve(args.index, args.year, zone_name, easting, northing)
    key, bounds = entry["key"], entry["bounds"]
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
