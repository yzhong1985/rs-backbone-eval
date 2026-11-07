#!/usr/bin/env python3
"""List the OlmoEarth v1.3 shards that cover a PASTIS-HD feature set.

The store is a Zarr v3 array over (year, dim, y, x) with outer shards of
``(1, 128, 2048, 2048)``; one shard is one storage object, so a 128x128 patch
window pulls in every shard it touches. This script maps each patch's
``grid_bbox`` from the AEF sidecar to that shard grid and prints the exact
subset (135 shards for the 2019 PASTIS-HD patches), so a mirror download is
reproducible without guessing a bounding box.

    uv run python scripts/olmoearth_shard_list.py \
        --aef-dir /home/yzhong/rs-eval/data-aef/features/2019 \
        --zarr-root /home/yzhong/rs-eval/data-olmoearth/cache/embeddings.zarr \
        --year 2019

Pass ``--urls`` to emit the Source Cooperative object URLs (one per line).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SHARD = 2048
GRID = 128
BASE = "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"


def shards_for_bbox(
    bbox: tuple[float, float, float, float],
    transform: tuple[float, float, float, float, float, float],
) -> set[tuple[int, int]]:
    """Shard (row, col) cells touched by a patch window, from its bbox.

    The affine origin is a pixel corner, so ``floor`` gives the containing
    pixel (nearest centre) -- the same rule as the extractor. ``round`` shifted
    boundary windows one column east and could miss the last shard on an edge
    (review 2026-11-07, second round).
    """
    a, _, c, _, e, f = transform
    xmin, _, _, ymax = bbox
    col0 = int((xmin - c) // a)
    row0 = int((ymax - f) // e)
    rows = {row0 // SHARD, (row0 + GRID - 1) // SHARD}
    cols = {col0 // SHARD, (col0 + GRID - 1) // SHARD}
    return {(row, col) for row in rows for col in cols}


def required_shards(
    sidecars: list[dict], transforms: dict[int, tuple[float, ...]]
) -> dict[int, set[tuple[int, int]]]:
    by_zone: dict[int, set[tuple[int, int]]] = {}
    for meta in sidecars:
        zone = int(meta["tile"][1:3])
        by_zone.setdefault(zone, set()).update(
            shards_for_bbox(tuple(meta["grid_bbox"]), transforms[zone])
        )
    return by_zone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aef-dir", type=Path, required=True)
    parser.add_argument("--zarr-root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--urls", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    sidecars = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.aef_dir.glob("*.json"), key=lambda p: int(p.stem))
    ]
    transforms = {}
    for zone in sorted({int(meta["tile"][1:3]) for meta in sidecars}):
        meta = json.loads(
            (args.zarr_root / f"utm{zone:02d}" / "zarr.json").read_text(encoding="utf-8")
        )
        transforms[zone] = tuple(meta["attributes"]["spatial:transform"])

    by_zone = required_shards(sidecars, transforms)
    counts = Counter()
    total = 0
    for zone, shards in sorted(by_zone.items()):
        time_index = year_index(args.year)
        for row, col in sorted(shards):
            counts[zone] += 1
            total += 1
            if args.urls:
                print(f"{BASE}/utm{zone:02d}/embeddings/c/{time_index}/0/{row}/{col}")
    print(f"patches={len(sidecars)} shards={total} per_zone={dict(counts)}")
    if args.json_out is not None:
        payload = {
            str(zone): sorted([row, col] for row, col in shards) for zone, shards in by_zone.items()
        }
        args.json_out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


def year_index(year: int) -> int:
    """Position of *year* on the zone time axis (2017-2025, verified locally).

    All three UTM zones carry the same nine years; the shard path uses
    ``c/<index>/<dim_half>/<row>/<col>``.
    """
    if not 2017 <= year <= 2025:
        raise SystemExit(f"year {year} outside the 2017-2025 time axis")
    return year - 2017


if __name__ == "__main__":
    raise SystemExit(main())
