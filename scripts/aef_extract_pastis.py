#!/usr/bin/env python3
"""Extract AEF 2019 features for every PASTIS-HD patch (T11 input).

For each patch the 10 m sample grid comes from the SPOT affine (pixel centres at
5 + 10k), not from the patch bbox: the 24-patch QC showed this convention lands
on (0,0) for 17/24 patches and within one cell for 21/24. Reads are boundless so
a patch that straddles a COG edge is padded with NODATA rather than clipped.

Run in the LIM worker environment:

    uv run --project workers/lim --with pyarrow --with pyproj \\
        python scripts/aef_extract_pastis.py --year 2019 --workers 6

Output: ``<out>/<year>/<patch>.npz`` (features float16 [64,128,128], valid
bool [128,128]) and ``<patch>.json`` (COG, window, valid share, dates, fold).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

AUDIT = Path("/geodata/rs-eval/data/derived/aef-audit/PASTIS-HD")
INDEX = Path("/geodata/rs-eval/data/derived/aef/aef_index.parquet")
PASTIS = Path("/home/yzhong/rs-eval/data-aef/PASTIS-HD")
COGS = Path("/home/yzhong/rs-eval/data-aef/aef-cogs")
OUT = Path("/home/yzhong/rs-eval/data-aef/features")
NODATA = -128
_local = threading.local()


def _dataset(path: Path):
    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = {}
    if path not in cache:
        cache[path] = rasterio.open(path)
    return cache[path]


def load_index(year: int) -> list[dict]:
    rows = pq.read_table(
        INDEX,
        columns=["year", "utm_zone", "path", "utm_west", "utm_south", "utm_east", "utm_north"],
    ).to_pylist()
    return [row for row in rows if row["year"] == year]


def cog_for(rows: list[dict], x: float, y: float) -> dict | None:
    """First local COG containing the point, else the nearest local one."""
    for row in rows:
        if row["utm_west"] <= x <= row["utm_east"] and row["utm_south"] <= y <= row["utm_north"]:
            return row
    if not rows:
        return None

    def distance(row: dict) -> float:
        dx = max(row["utm_west"] - x, 0, x - row["utm_east"])
        dy = max(row["utm_south"] - y, 0, y - row["utm_north"])
        return float(np.hypot(dx, dy))

    return min(rows, key=distance)


def centre_utm(patch: dict) -> tuple[float, float, str]:
    poly = np.asarray(patch["geometry"]["coordinates"][0], dtype=float).reshape(-1, 2)
    cx, cy = poly.mean(axis=0)
    lon, lat = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(cx, cy)
    zone = int((lon + 180) // 6) + 1
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone}", always_xy=True).transform(
        lon, lat
    )
    return x, y, f"{zone}N"


def extract(patch: dict, year: int, rows: list[dict], out_dir: Path) -> dict:
    pid = int(patch["properties"]["ID_PATCH"])
    target = out_dir / f"{pid}.npz"
    if target.is_file():
        return {"patch": pid, "skipped": True}
    spot = next((PASTIS / "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019").glob(f"*_{pid}.tif"), None)
    if spot is None:
        return {"patch": pid, "error": "no SPOT"}
    x, y, zone = centre_utm(patch)

    with rasterio.open(spot) as spot_ds:
        cc, rr = np.meshgrid(5 + 10 * np.arange(128), 5 + 10 * np.arange(128))
        xs, ys = spot_ds.transform * (cc, rr)

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    candidates = [
        row
        for row in rows
        if not (
            row["utm_east"] < xmin
            or row["utm_west"] > xmax
            or row["utm_north"] < ymin
            or row["utm_south"] > ymax
        )
    ]
    if not candidates:
        return {"patch": pid, "error": f"no COG for {zone}"}

    started = time.time()
    feats = np.full((64, 128, 128), NODATA, dtype=np.int16)
    covered = np.zeros((128, 128), dtype=bool)
    used: list[str] = []
    for row in candidates:
        inside = (
            (xs >= row["utm_west"])
            & (xs <= row["utm_east"])
            & (ys >= row["utm_south"])
            & (ys <= row["utm_north"])
        )
        if not inside.any():
            continue
        path = COGS / str(year) / row["path"].rsplit("/", 1)[-1]
        if not path.is_file():
            continue
        ds = _dataset(path)
        rws, cls = rasterio.transform.rowcol(ds.transform, np.ravel(xs), np.ravel(ys))
        rws = np.asarray(rws).reshape(xs.shape)
        cls = np.asarray(cls).reshape(xs.shape)
        c0, c1 = int(cls[inside].min()), int(cls[inside].max()) + 1
        r0, r1 = int(rws[inside].min()), int(rws[inside].max()) + 1
        block = ds.read(
            window=Window(c0, r0, c1 - c0, r1 - r0),
            boundless=True,
            fill_value=NODATA,
        ).astype(np.int16)
        feats[:, inside] = block[:, (rws - r0)[inside], (cls - c0)[inside]]
        covered |= inside
        used.append(path.name)
    if not used:
        return {"patch": pid, "error": f"no local COG for {zone}"}

    valid = covered & (feats != NODATA).all(axis=0)
    np.savez(target, features=feats.astype(np.float16), valid=valid)

    dates = patch["properties"]["dates-S2"]
    meta = {
        "patch": pid,
        "tile": patch["properties"]["TILE"],
        "fold": patch["properties"]["Fold"],
        "year": year,
        "cogs": used,
        "grid_bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
        "valid_share": round(float(valid.mean()), 4),
        "dates_s2": [int(v) for v in dates.values()],
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / f"{pid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {
        "patch": pid,
        "valid": meta["valid_share"],
        "cogs": len(used),
        "seconds": meta["seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    meta = json.loads((AUDIT / "metadata.geojson").read_text(encoding="utf-8"))
    patches = meta["features"]
    if args.limit:
        patches = patches[: args.limit]
    year_dir = COGS / str(args.year)
    rows = [
        row
        for row in load_index(args.year)
        if (year_dir / row["path"].rsplit("/", 1)[-1]).is_file()
    ]
    out_dir = args.out / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda patch: extract(patch, args.year, rows, out_dir), patches):
            if result.get("error"):
                errors += 1
                print("ERR", result, flush=True)
            else:
                done += 1
                if done % 200 == 0:
                    print(f"{done}/{len(patches)} {time.time() - started:.0f}s", flush=True)
    print(f"done {done}, errors {errors}, {time.time() - started:.0f}s -> {out_dir}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
