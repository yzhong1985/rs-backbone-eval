#!/usr/bin/env python3
"""Spot-check the AEF/PASTIS 10 m grid alignment on a per-tile sample.

For each patch: build the 10 m sample grid from the SPOT affine, read the 2019
AEF window from the local COGs, compute PC1, and correlate it with the S2 NDVI
of the acquisition closest to 2019-07-15 (the pre-declared summer anchor). The
best shift within +/-2 cells is reported for diagnosis only; it is never used to
move the data. Run in the LIM worker environment with pyarrow/pyproj:

    uv run --project workers/lim --with pyarrow --with pyproj \
        python scripts/aef_align_qc.py

Writes alignment_batch.csv under the audit directory.
"""

import csv
import glob
import json
import statistics
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

ROOT = Path("/home/yzhong/rs-eval/data-aef/PASTIS-HD")
COGDIR = Path("/home/yzhong/rs-eval/data-aef/aef-cogs/2019")
AUD = Path("/home/yzhong/rs-eval/audits/aef-20261103")
AUD.mkdir(parents=True, exist_ok=True)
META_PATH = "/geodata/rs-eval/data/derived/aef-audit/PASTIS-HD/metadata.geojson"
with open(META_PATH, encoding="utf-8") as handle:
    META = json.load(handle)
BY = {f["properties"]["ID_PATCH"]: f for f in META["features"]}
idx = pq.read_table(
    "/geodata/rs-eval/data/derived/aef/aef_index.parquet",
    columns=["year", "utm_zone", "path", "utm_west", "utm_south", "utm_east", "utm_north"],
)
COGS = [r for r in idx.to_pylist() if r["year"] == 2019]
PICKS = [
    10000,
    10107,
    10213,
    10323,
    10432,
    10540,
    20000,
    20128,
    20256,
    20387,
    20522,
    20657,
    30000,
    30146,
    30292,
    30440,
    30589,
    30737,
    40000,
    40111,
    40223,
    40337,
    40451,
    40562,
]


def cog_for(pid):
    f = BY[pid]
    c = np.asarray(f["geometry"]["coordinates"][0], dtype=float).reshape(-1, 2)
    x, y = c.mean(axis=0)
    # Lambert-93 -> WGS84 -> local UTM
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(x, y)
    zone = int((lon + 180) // 6) + 1
    ex, ny = Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone}", always_xy=True).transform(
        lon, lat
    )
    for r in COGS:
        if (
            r["utm_zone"] == f"{zone}N"
            and r["utm_west"] <= ex <= r["utm_east"]
            and r["utm_south"] <= ny <= r["utm_north"]
        ):
            return r, f"{zone}N"
    return None, f"{zone}N"


def pc1_and_shift(pid):
    r, zone = cog_for(pid)
    if r is None:
        return None
    name = r["path"].rsplit("/", 1)[-1]
    path = COGDIR / name
    spotp = glob.glob(str(ROOT / f"DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019/*_{pid}.tif"))[0]
    with rasterio.open(spotp) as ds:
        cc, rr = np.meshgrid(5 + 10 * np.arange(128), 5 + 10 * np.arange(128))
        xs, ys = ds.transform * (cc, rr)
        spot = np.transpose(ds.read(), (1, 2, 0)).astype(np.float32)
    with rasterio.open(path) as ds:
        rowflat, colflat = rasterio.transform.rowcol(ds.transform, np.ravel(xs), np.ravel(ys))
        rows = np.asarray(rowflat).reshape(xs.shape)
        cols = np.asarray(colflat).reshape(xs.shape)
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        blk = ds.read(
            window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True, fill_value=-128
        ).astype(np.float32)
        feats = blk[:, rows - r0, cols - c0]
    valid = (feats != -128).all(axis=0)
    # PC1 is computed on valid pixels only: zero-padding the missing regions
    # would shift the mean / axes and inject a constant block into every
    # correlation (review 2026-11-03, P2). Gaps can only come from the
    # boundless read of an edge-straddling window.
    x = np.where(feats == -128, 0.0, np.sign(feats) * (feats / 127.5) ** 2)
    X = x.reshape(64, -1).T
    flat_valid = valid.reshape(-1)
    Xv = X[flat_valid]
    Xv = Xv - Xv.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xv, full_matrices=False)
    pc_flat = np.zeros(X.shape[0], dtype=np.float32)
    pc_flat[flat_valid] = Xv @ vt[0]
    pc = pc_flat.reshape(128, 128)

    s2 = np.load(ROOT / f"DATA_S2/S2_{pid}.npy")
    order = sorted(BY[pid]["properties"]["dates-S2"].items(), key=lambda kv: int(kv[0]))
    didx = min(order, key=lambda kv: abs(int(kv[1]) - 20190715))[0]
    frame = s2[int(didx)]
    ndvi = (frame[6].astype(np.float32) - frame[2]) / np.maximum(frame[6] + frame[2], 1)
    spot10 = spot.reshape(128, 10, 128, 10, 3).mean(axis=(1, 3))
    bright = spot10.mean(-1)

    r0_ndvi = float(np.corrcoef(pc[valid], ndvi[valid])[0, 1])
    r0_spot = float(np.corrcoef(pc[valid], bright[valid])[0, 1])
    best = max(
        (
            (
                abs(
                    float(
                        np.corrcoef(
                            np.roll(pc, (dy, dx), axis=(0, 1))[
                                np.roll(valid, (dy, dx), axis=(0, 1))
                            ],
                            ndvi[np.roll(valid, (dy, dx), axis=(0, 1))],
                        )[0, 1]
                    )
                ),
                float(
                    np.corrcoef(
                        np.roll(pc, (dy, dx), axis=(0, 1))[np.roll(valid, (dy, dx), axis=(0, 1))],
                        ndvi[np.roll(valid, (dy, dx), axis=(0, 1))],
                    )[0, 1]
                ),
                dx,
                dy,
            )
            for dy in range(-2, 3)
            for dx in range(-2, 3)
        ),
        key=lambda t: t[0],
    )
    return {
        "patch": pid,
        "tile": BY[pid]["properties"]["TILE"],
        "zone": zone,
        "fold": BY[pid]["properties"]["Fold"],
        "date": int(order[int(didx)][1]),
        "cog": name,
        "valid": round(float(valid.mean()), 3),
        "r_ndvi": round(r0_ndvi, 3),
        "r_spot": round(r0_spot, 3),
        "best_r": round(best[1], 3),
        "best_dx": best[2],
        "best_dy": best[3],
        "shift_px": round(float(np.hypot(best[2], best[3])), 2),
    }


rows = []
for pid in PICKS:
    try:
        row = pc1_and_shift(pid)
        if row:
            rows.append(row)
            print(row, flush=True)
    except Exception as e:
        print("ERR", pid, type(e).__name__, str(e)[:80], flush=True)
out = AUD / "alignment_batch.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", out, len(rows))
zero = [r for r in rows if (r["best_dx"], r["best_dy"]) == (0, 0)]
print("at (0,0):", len(zero), "/", len(rows))
print(
    "shift dist",
    {
        k: sum(1 for r in rows if (r["best_dx"], r["best_dy"]) == k)
        for k in sorted({(r["best_dx"], r["best_dy"]) for r in rows})
    },
)
print("mean |r| ndvi", round(statistics.mean(abs(r["r_ndvi"]) for r in rows), 3))
