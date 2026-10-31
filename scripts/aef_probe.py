#!/usr/bin/env python3
"""Read one AlphaEarth Foundations (AEF) annual embedding window and report it.

AEF is not an encoder: the public product is a precomputed 64-band int8 COG at
10 m on UTM grids, with the de-quantised vector at every pixel normalised to
length 1. This probe answers P1 of the AEF plan: resolve a point through the
official index, read the window, de-quantise it, check the norm, and record the
identity (object path, ETag, CRS, transform) needed to reuse it.

Needs rasterio (and pyarrow for the official index), so run it in a worker
environment:

    workers/prithvi/.venv/bin/python scripts/aef_probe.py \
        --tif /geodata/rs-eval/data/raw/fair1m/images/val/799.tif --year 2021

The year is chosen for the engineering test; it is **not** an acquisition year
(FAIR1M's capture date is unknown and the XML research/time may not be used for
this). Outputs go to ``/home/yzhong/rs-eval/audits/aef-<date>/`` so they survive a
restart; nothing is written into the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from rs_backbone_eval.data.aef_index import (
    BUCKET,
    CELL_M,
    NODATA,
    AefIndexError,
    find_cog,
    index_path,
    load_index,
    official_slice,
)

DEFAULT_PARQUET = Path("/geodata/rs-eval/data/derived/aef/aef_index.parquet")
DEFAULT_OUT = Path(f"/home/yzhong/rs-eval/audits/aef-{date.today():%Y%m%d}")


def utm_zone(lon: float, lat: float) -> tuple[int, str]:
    zone = int((lon + 180.0) // 6.0) + 1
    return zone, ("N" if lat >= 0 else "S")


def zone_epsg(zone: int, hemisphere: str) -> str:
    return f"EPSG:{32600 + zone if hemisphere == 'N' else 32700 + zone}"


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


def resolve(args, year: int, zone_name: str, x: float, y: float) -> dict:
    """Tile entry from the official slice, else the converted JSON slice."""
    try:
        index = official_slice(args.parquet, year, zone_name)
    except AefIndexError:
        path = index_path(args.index, year, zone_name)
        index = load_index(path)
    entry = find_cog(index, x, y)
    if entry is None:
        raise SystemExit(f"no COG covers ({x:.1f}, {y:.1f}) in {year}/{zone_name}")
    return entry


def https_url(path: str) -> str:
    """``gs://bucket/key`` or a bare key -> an HTTPS object URL."""
    if path.startswith("gs://"):
        return "https://storage.googleapis.com/" + path[len("gs://") :]
    return f"{BUCKET}/{path}"


def object_identity(path: str) -> dict:
    """ETag/generation of the COG, so a later run can prove it read the same object."""
    url = https_url(path)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {
                "etag": response.headers.get("ETag", "").strip('"'),
                "generation": response.headers.get("x-goog-generation"),
            }
    except Exception as error:  # identity is best effort, never fatal
        return {"etag": None, "generation": None, "head_error": type(error).__name__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", type=Path, help="a georeferenced image; its centre is used")
    parser.add_argument("--lon", type=float, default=139.6920)
    parser.add_argument("--lat", type=float, default=35.5203)
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="AEF year to read; arbitrary for an engineering test, "
        "not the image's acquisition year",
    )
    parser.add_argument("--zone", help="override the resolved UTM zone, e.g. 54N")
    parser.add_argument("--radius-m", type=float, default=320.0)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--index", type=Path, default=Path("/geodata/rs-eval/data/derived/aef/index")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
        f"E={easting:.1f} N={northing:.1f}, zone {zone_name}, year {args.year} "
        f"(arbitrary, not the acquisition year)",
        flush=True,
    )

    entry = resolve(args, args.year, zone_name, easting, northing)
    print(f"resolved {entry['key']} bounds={entry['bounds']}", flush=True)

    with rasterio.open("/vsicurl/" + https_url(entry["path"])) as dataset:
        row, col = dataset.index(easting, northing)
        half = max(1, int(round(args.radius_m / CELL_M)))
        window = rasterio.windows.Window(
            max(col - half, 0), max(row - half, 0), 2 * half + 1, 2 * half + 1
        )
        raw = dataset.read(window=window)
        transform = dataset.window_transform(window)
        flipped = bool(dataset.transform.e > 0)
        if flipped:
            # AEF COGs are stored bottom-up: row 0 is the southern edge. Flip so the
            # array is north-up like every other raster we hand to a model.
            raw = raw[:, ::-1, :]
            north = transform.f + transform.e * raw.shape[1]
            transform = rasterio.Affine(
                transform.a, transform.b, transform.c, -transform.d, -transform.e, north
            )
        print(
            f"window {raw.shape} at row/col {row}/{col}, crs {dataset.crs}, "
            f"flipped_to_north_up={flipped}",
            flush=True,
        )

    valid = raw != NODATA
    values = np.where(valid, raw.astype(np.float32), np.nan)
    values = np.sign(values) * (values / 127.5) ** 2
    values = np.nan_to_num(values, nan=0.0)
    norms = np.linalg.norm(values, axis=0)
    valid_norms = norms[valid[0]]
    norm_ok = bool(valid_norms.size and np.allclose(valid_norms, 1.0, atol=2e-2))

    identity = object_identity(entry["path"])
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.year}_{zone_name}_{row}_{col}"
    np.savez_compressed(
        args.out / f"{stem}.npz",
        features=values.astype(np.float32),
        valid=valid,
        transform=np.array(list(transform)[:6], dtype=np.float64),
    )
    report = {
        "source_uri": entry["path"],
        "https_uri": https_url(entry["path"]),
        "object": identity,
        "year": args.year,
        "year_note": "arbitrary engineering test year; not the image acquisition year",
        "zone": zone_name,
        "crs": str(epsg),
        "bounds": entry["bounds"],
        "window_transform": list(transform)[:6],
        "flipped_to_north_up": flipped,
        "window_shape": list(raw.shape),
        "window_valid_share": float(valid.mean()),
        "norm_check": {
            "all_close_to_one": norm_ok,
            "min": float(valid_norms.min()) if valid_norms.size else None,
            "max": float(valid_norms.max()) if valid_norms.size else None,
        },
        "point": {"lon": lon, "lat": lat, "easting": easting, "northing": northing},
        "seconds": round(time.time() - started, 1),
        "note": "De-quantised with sign(v) * (v / 127.5)^2 after masking -128; "
        "rows flipped when the COG's y scale is positive.",
    }
    (args.out / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "window_shape",
                    "window_valid_share",
                    "flipped_to_north_up",
                    "norm_check",
                    "seconds",
                )
            },
            ensure_ascii=False,
        )
    )
    print("wrote", args.out / f"{stem}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
