#!/usr/bin/env python3
"""Extract AEF 2019 features for every PASTIS-HD patch (T11 input).

For each patch the 10 m sample grid comes from the SPOT affine (pixel centres at
5 + 10k), not from the patch bbox: the 24-patch QC showed this convention lands
on (0,0) for 17/24 patches and within one cell for 21/24. Reads are boundless so
a patch that straddles a COG edge is padded with NODATA rather than clipped.

Run in the LIM worker environment (roots honour ``RS_EVAL_AEF_DATA_ROOT``):

    uv run --project workers/lim --with pyarrow --with pyproj \\
        python scripts/aef_extract_pastis.py --year 2019 --workers 6

Output: ``<out>/<year>/<patch>.npz`` (features float16 [64,128,128], valid
bool [128,128]) and ``<patch>.json`` (COG, window, valid share, dates, fold).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

_LOCAL = Path(os.environ.get("RS_EVAL_AEF_DATA_ROOT", "/home/yzhong/rs-eval/data-aef"))
AUDIT = Path(
    os.environ.get("RS_EVAL_AEF_AUDIT_META", "/geodata/rs-eval/data/derived/aef-audit/PASTIS-HD")
)
INDEX = Path(
    os.environ.get("RS_EVAL_AEF_INDEX", "/geodata/rs-eval/data/derived/aef/aef_index.parquet")
)
PASTIS = Path(os.environ.get("RS_EVAL_AEF_PASTIS_ROOT", _LOCAL / "PASTIS-HD"))
COGS = Path(os.environ.get("RS_EVAL_AEF_COGS_ROOT", _LOCAL / "aef-cogs"))
OUT = Path(os.environ.get("RS_EVAL_AEF_FEATURES_ROOT", _LOCAL / "features"))
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
    import pyarrow.parquet as pq  # needed only here, not by extract()/tests

    rows = pq.read_table(
        INDEX,
        columns=["year", "utm_zone", "path", "utm_west", "utm_south", "utm_east", "utm_north"],
    ).to_pylist()
    return [row for row in rows if row["year"] == year]


def centre_utm(patch: dict) -> tuple[float, float, str]:
    from pyproj import Transformer  # pyproj is only needed by this helper

    poly = np.asarray(patch["geometry"]["coordinates"][0], dtype=float).reshape(-1, 2)
    cx, cy = poly.mean(axis=0)
    lon, lat = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(cx, cy)
    zone = int((lon + 180) // 6) + 1
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone}", always_xy=True).transform(
        lon, lat
    )
    return x, y, f"{zone}N"


def _cache_ok(target: Path) -> bool:
    """A cached patch counts as extracted only if both artefacts are usable.

    The archive is fully read (not just keys: numpy keeps lazy arrays until a
    member is touched), the shapes/dtypes must match the declared AEF encoding
    and the sidecar must parse with the identity fields (review 2026-11-03,
    round 5, P2: an empty-shaped npz with a non-JSON sidecar used to pass).
    """
    sidecar = target.with_suffix(".json")
    if not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not all(meta.get(key) is not None for key in ("patch", "year", "quantization")):
        return False
    try:
        with np.load(target) as archive:
            features = archive["features"]
            valid = archive["valid"]
            if features.dtype != np.float16 or features.ndim != 3:
                return False
            height, width = features.shape[1:]
            if features.shape[1] != features.shape[2] or width <= 0 or height <= 0:
                return False
            if valid.shape != (height, width) or valid.dtype != np.bool_:
                return False
            # force the lazy members so truncation surfaces here
            np.asarray(features).max()
            np.asarray(valid).sum()
    except Exception:
        return False
    return True


def _cog_sha256(path: Path) -> str:
    """Per-COG provenance hash, cached once per process (review 2026-11-03)."""
    cache = getattr(_local, "cog_hashes", None)
    if cache is None:
        cache = _local.cog_hashes = {}
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _script_commit() -> str:
    """HEAD, suffixed with ``-dirty`` for uncommitted tracked changes."""
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    commit = head.stdout.strip() if head.returncode == 0 else "unknown"
    status = subprocess.run(
        ["git", "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False
    )
    if status.returncode == 0 and status.stdout.strip():
        commit = f"{commit}-dirty"
    return commit


def extract(patch: dict, year: int, rows: list[dict], out_dir: Path) -> dict:
    pid = int(patch["properties"]["ID_PATCH"])
    target = out_dir / f"{pid}.npz"
    if target.is_file():
        # A stale crash can leave a truncated .npz with a missing sidecar, so
        # "exists" alone must not count as done (review 2026-11-03, P2).
        if _cache_ok(target):
            return {"patch": pid, "skipped": True}
        target.unlink(missing_ok=True)
    spot = next((PASTIS / "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019").glob(f"*_{pid}.tif"), None)
    if spot is None:
        return {"patch": pid, "error": "no SPOT"}
    zone = centre_utm(patch)[2]

    with rasterio.open(spot) as spot_ds:
        cc, rr = np.meshgrid(5 + 10 * np.arange(128), 5 + 10 * np.arange(128))
        xs, ys = spot_ds.transform * (cc, rr)

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    # Zone gate first (review 2026-11-03, P1): UTM coordinates are numerically
    # overlapping across zones, so a bbox match in the wrong zone would read
    # confidently-nonsense data (valid_share stays 1.0). PASTIS spans 30N/31N/32N.
    candidates = [
        row
        for row in rows
        if row["utm_zone"] == zone
        and not (
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
        zone_eps = 32600 + int(zone[:2]) + (100 if zone.endswith("S") else 0)
        if ds.crs is None or ds.crs.to_epsg() != zone_eps:
            return {"patch": pid, "error": f"COG crs {ds.crs} is not UTM {zone}"}
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
        # Raw AEF int8 values (payload [-127, 127], -128 = NODATA) are stored
        # as float16 **without** de-quantization; the representation consumed
        # downstream is sign(x) * (x / 127.5)**2 masked by `valid`
        # (declared here after review 2026-11-03).
        "quantization": "raw_aef_int8_as_float16",
        # Provenance for the manifest/audit layer (review 2026-11-03).
        "script_commit": _script_commit(),
        "index_sha256": _sha256_file(INDEX),
        "cog_sha256": {name: _cog_sha256(COGS / str(year) / name) for name in used},
        "seconds": round(time.time() - started, 2),
    }
    # Sidecar first, then the payload, both via rename: an interrupted run can
    # never leave a bare .npz that the next pass treats as done
    # (review 2026-11-03, P2), and _cache_ok() rejects any npz without a
    # readable sidecar.
    sidecar = out_dir / f"{pid}.json"
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, sidecar)
    npz_tmp = target.with_suffix(".npz.tmp")
    # np.savez appends ".npz" to a bare path, so "10000.npz.tmp" would
    # materialise as "10000.npz.tmp.npz" and the following rename would fail
    # (review 2026-11-03, round 5, P1). Writing through the file handle keeps
    # the tmp name exact.
    with npz_tmp.open("wb") as handle:
        np.savez(handle, features=feats.astype(np.float16), valid=valid)
    os.replace(npz_tmp, target)
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
