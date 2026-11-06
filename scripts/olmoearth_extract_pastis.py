#!/usr/bin/env python3
"""Extract OlmoEarth v1.3 embeddings for PASTIS-HD patches (T11 input).

The 10 m grid comes from the AEF sidecar's ``grid_bbox`` (SPOT affine,
``5 + 10k`` sample points): we read the nearest OlmoEarth pixel for each
sample. Output mirrors the AEF feature cache:

    <out>/<patch>.npz   features int8 [dims,128,128], valid bool [128,128]
    <out>/<patch>.json  source, window, valid share, dequantization, hashes

Run with the extra dependencies:

    uv run --python 3.12 --with 'zarr>=3' --with obstore --with numpy \
        python scripts/olmoearth_extract_pastis.py --dims 64 --limit 5

The remote store is public/anonymous. Reads are retried; ``--workers`` opens
one Zarr handle per thread because the synchronous client is not guaranteed to
be re-entrant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from obstore.store import S3Store

STORE_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
ENDPOINT = "https://data.source.coop"
YEAR = 2019
DIMS = 64
PATCH_SIZE = 128
AEF_META = Path("/home/yzhong/rs-eval/data-aef/features/2019")
OUT = Path("/home/yzhong/rs-eval/data-olmoearth/features/2019")
_local = threading.local()


def _root():
    root = getattr(_local, "root", None)
    if root is None:
        s3 = S3Store.from_url(STORE_URL, endpoint=ENDPOINT, skip_signature=True)
        store = zarr.storage.ObjectStore(s3, read_only=True)
        zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
        root = zarr.open_group(store, mode="r")
        _local.root = root
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_patch(patch: int, year: int, dims: int) -> dict:
    meta = json.loads((AEF_META / f"{patch}.json").read_text(encoding="utf-8"))
    zone_no = int(meta["tile"][1:3])
    root = _root()
    zone = root[f"utm{zone_no:02d}"]
    a, _, c, _, e, f = zone.attrs["spatial:transform"]
    xmin, ymin, xmax, ymax = meta["grid_bbox"]
    # Nearest pixel for the top-left sample point; the 10 m grid is contiguous.
    col0 = int(round((xmin - c) / a))
    row0 = int(round((ymax - f) / e))
    times = zone["time"][:]
    matches = np.where(times == year)[0]
    if not len(matches):
        raise ValueError(f"year {year} not in zone time axis {times.tolist()}")
    t = int(matches[0])

    features = np.full((dims, PATCH_SIZE, PATCH_SIZE), -128, dtype=np.int8)
    attempts = 4
    for attempt in range(attempts):
        try:
            window = zone["embeddings"][
                t, 0:dims, row0 : row0 + PATCH_SIZE, col0 : col0 + PATCH_SIZE
            ]
            break
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))
    window = np.asarray(window, dtype=np.int8)
    h = min(PATCH_SIZE, window.shape[1])
    w = min(PATCH_SIZE, window.shape[2])
    features[:, :h, :w] = window[:, :h, :w]
    valid = ~np.all(features == -128, axis=0)
    return {
        "features": features,
        "valid": valid,
        "meta": {
            "patch": patch,
            "tile": meta["tile"],
            "fold": meta["fold"],
            "year": year,
            "dims": dims,
            "source": STORE_URL,
            "zone": f"utm{zone_no:02d}",
            "rowcol": [row0, col0],
            "grid_bbox": meta["grid_bbox"],
            "valid_share": round(float(valid.mean()), 4),
            "quantization": "olmoearth_int8",
            "dequantization": "(abs(v)/127.5)**2 * sign(v); -128 -> nodata",
            "dates_s2": meta.get("dates_s2", []),
            "aef_sidecar": str(AEF_META / f"{patch}.json"),
        },
    }


def extract(patch: int, year: int, dims: int, out_dir: Path) -> dict:
    target = out_dir / f"{patch}.npz"
    sidecar = out_dir / f"{patch}.json"
    if target.is_file() and sidecar.is_file():
        return {"patch": patch, "skipped": True}
    payload = _read_patch(patch, year, dims)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = payload["meta"]
    meta["sha256"] = None
    tmp_side = sidecar.with_suffix(".json.tmp")
    tmp_npz = target.with_name(target.name + ".tmp")
    tmp_side.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with tmp_npz.open("wb") as handle:
        np.savez(handle, features=payload["features"], valid=payload["valid"])
    meta["sha256"] = _sha256(tmp_npz)
    tmp_side.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_side, sidecar)
    os.replace(tmp_npz, target)
    return {"patch": patch, "valid": meta["valid_share"], "skipped": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--dims", type=int, choices=(64, 128), default=DIMS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--patch-ids", type=str, default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.patch_ids:
        patches = [int(value) for value in args.patch_ids.split(",") if value.strip()]
    else:
        patches = sorted(int(path.stem) for path in AEF_META.glob("*.json"))
        if args.limit:
            patches = patches[: args.limit]
    print(f"{len(patches)} patches, year {args.year}, dims {args.dims}, out {args.out}", flush=True)
    started = time.time()
    ok = errors = skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract, patch, args.year, args.dims, args.out) for patch in patches]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result.get("skipped"):
                    skipped += 1
                    print(f"[{index}/{len(patches)}] {result['patch']} skipped", flush=True)
                else:
                    ok += 1
                    print(
                        f"[{index}/{len(patches)}] {result['patch']} valid {result['valid']}",
                        flush=True,
                    )
            except Exception as error:  # noqa: BLE001 - report and continue
                errors += 1
                print(f"[{index}/{len(patches)}] ERROR {type(error).__name__}: {error}", flush=True)
    print(
        f"done ok={ok} skipped={skipped} errors={errors} in {round(time.time() - started, 1)} s",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
