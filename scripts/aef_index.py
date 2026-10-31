#!/usr/bin/env python3
"""Build an AEF tile slice from the official Parquet (or verify it by scanning).

    # official index -> dependency-free slice (needs pyarrow, i.e. a worker venv)
    workers/prithvi/.venv/bin/python scripts/aef_index.py --year 2021 --zone 54N

    # cross-check a slice against the COG headers (665 s for 54N/2021)
    workers/prithvi/.venv/bin/python scripts/aef_index.py --year 2021 --zone 54N --check-scan

The official file is `satellite_embedding/v1/annual/aef_index.parquet`
(302,466 rows, 69,600,467 bytes, sha256 5d216665…cb947). The scan path is kept
only to prove a slice matches the files; it is not the production route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rs_backbone_eval.data.aef_index import (
    AefIndexError,
    compare_indexes,
    official_slice,
    scan_index,
    write_slice,
)

DEFAULT_PARQUET = Path("/geodata/rs-eval/data/derived/aef/aef_index.parquet")
DEFAULT_OUT = Path("/geodata/rs-eval/data/derived/aef/index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, nargs="+", default=[2021])
    parser.add_argument("--zone", nargs="+", default=["54N"])
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-scan", action="store_true")
    args = parser.parse_args()

    if not args.parquet.is_file():
        raise SystemExit(f"missing official index: {args.parquet}")
    print(
        json.dumps(
            {
                "parquet": str(args.parquet),
                "bytes": args.parquet.stat().st_size,
                "sha256": sha256(args.parquet),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for year in args.year:
        for zone in args.zone:
            index = official_slice(args.parquet, year, zone)
            target = write_slice(index, args.out)
            print(
                json.dumps(
                    {
                        "slice": str(target),
                        "year": year,
                        "zone": zone,
                        "cogs": index["cog_count"],
                        "crs": index["crs"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.check_scan:
                continue
            scanned = scan_index(year, zone, args.out / "scan", workers=8)
            problems = compare_indexes(index, scanned)
            print(
                json.dumps(
                    {
                        "cross_check": "ok" if not problems else "mismatch",
                        "problems": problems[:10],
                        "scanned": scanned["cog_count"],
                        "seconds": scanned["seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AefIndexError as error:
        raise SystemExit(str(error)) from error
