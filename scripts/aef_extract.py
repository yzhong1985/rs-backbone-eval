#!/usr/bin/env python3
"""Sample AEF embeddings onto a FAIR1M tile's /8 grid (P2 input for fusion).

AEF is a precomputed 64-band field on a 10 m UTM grid; a FAIR1M tile is arbitrary
pixels at 0.2-0.8 m. For a *context* input we do not need AEF at pixel
resolution: this script samples the nearest AEF cell at the centre of every 8x8
pixel block, producing a `[64, H/8, W/8]` map that lines up with the detector's
/8 FPN level.

Run in a worker environment (rasterio + pyarrow):

    workers/prithvi/.venv/bin/python scripts/aef_extract.py \
        --year 2021 --limit 5 --out /home/yzhong/rs-eval/aef-features

Writes one ``<sample_id>.npz`` (features, valid) and one ``.json`` (year, AEF
COG, ETag, CRS, residual) per tile. The year is an engineering choice, not the
image acquisition year.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from rs_backbone_eval.data.aef_index import NODATA, find_cog, official_slice

MANIFEST = Path("/geodata/rs-eval/data/manifests/T10/holdout_1024_s824_seed17.jsonl")
DATA_ROOT = Path("/geodata/rs-eval/data")
PARQUET = Path("/geodata/rs-eval/data/derived/aef/aef_index.parquet")
BLOCK = 8


def utm_zone(lon: float, lat: float) -> tuple[int, str]:
    return int((lon + 180.0) // 6.0) + 1, ("N" if lat >= 0 else "S")


def read_aef_window(cog_path: str, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, dict]:
    """Nearest AEF cell for every (x, y) in the COG's UTM frame."""
    https = cog_path.replace("gs://", "https://storage.googleapis.com/")
    with rasterio.open("/vsicurl/" + https) as dataset:
        rows, cols = rasterio.transform.rowcol(dataset.transform, xs, ys)
        rows = np.clip(rows, 0, dataset.height - 1)
        cols = np.clip(cols, 0, dataset.width - 1)
        row0, row1 = int(rows.min()), int(rows.max()) + 1
        col0, col1 = int(cols.min()), int(cols.max()) + 1
        window = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        block = dataset.read(window=window).astype(np.int16)
        transform = dataset.window_transform(window)
        flipped = bool(dataset.transform.e > 0)
    if flipped:
        block = block[:, ::-1, :]
        north = transform.f + transform.e * block.shape[1]
        transform = rasterio.Affine(
            transform.a, transform.b, transform.c, -transform.d, -transform.e, north
        )
        rows = block.shape[1] - 1 - (rows - row0)
    else:
        rows = rows - row0
    cols = cols - col0
    sampled = block[:, rows, cols]  # [64, N]
    valid = sampled != NODATA
    values = np.where(valid, sampled.astype(np.float32), np.nan)
    values = np.sign(values) * (values / 127.5) ** 2
    values = np.nan_to_num(values, nan=0.0)
    meta = {
        "cog": cog_path,
        "crs": str(dataset.crs) if dataset.crs else None,
        "window_transform": list(transform)[:6],
        "valid_share": float(valid.mean()),
    }
    return values.astype(np.float16), meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("/home/yzhong/rs-eval/aef-features"))
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    rows = [row for row in rows if row["split"] == args.split][: args.limit]
    if not rows:
        raise SystemExit("no rows")
    args.out.mkdir(parents=True, exist_ok=True)

    zone_cache: dict[str, dict] = {}
    for row in rows:
        started = time.time()
        with rasterio.open(DATA_ROOT / row["image_paths"][0]) as image:
            if image.crs is None:
                print("skip (no CRS)", row["sample_id"])
                continue
            x, y, width, height = (int(v) for v in row["window"])
            valid_w = min(width, image.width - x)
            valid_h = min(height, image.height - y)
            # centre of each 8x8 block inside the tile's valid region
            steps_x = np.arange(0, valid_w, BLOCK) + BLOCK / 2
            steps_y = np.arange(0, valid_h, BLOCK) + BLOCK / 2
            grid_x, grid_y = np.meshgrid(x + steps_x, y + steps_y)
            px = image.transform * (grid_x.ravel(), grid_y.ravel())
            lon, lat = warp_transform(image.crs, "EPSG:4326", px[0], px[1])
        zone, hemisphere = utm_zone(float(np.mean(lon)), float(np.mean(lat)))
        zone_name = f"{zone}{hemisphere}"
        epsg = f"EPSG:{32600 + zone if hemisphere == 'N' else 32700 + zone}"
        east, north = warp_transform("EPSG:4326", epsg, lon, lat)
        index = zone_cache.get(zone_name) or official_slice(args.parquet, args.year, zone_name)
        zone_cache[zone_name] = index
        entry = find_cog(index, float(np.mean(east)), float(np.mean(north)))
        if entry is None:
            print("skip (no AEF tile)", row["sample_id"])
            continue
        features, meta = read_aef_window(entry["path"], np.asarray(east), np.asarray(north))
        shape = (len(steps_y), len(steps_x))
        tensor = features.reshape(64, *shape)
        valid = tensor.any(axis=0)
        np.savez_compressed(
            args.out / f"{row['sample_id']}.npz", features=tensor.astype(np.float16), valid=valid
        )
        report = {
            "sample_id": row["sample_id"],
            "year": args.year,
            "year_note": "engineering choice, not the acquisition year",
            "grid": list(shape),
            "block_px": BLOCK,
            "zone": zone_name,
            "crs": epsg,
            **meta,
            "seconds": round(time.time() - started, 1),
        }
        (args.out / f"{row['sample_id']}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: report[k] for k in ("sample_id", "grid", "valid_share", "seconds")},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
