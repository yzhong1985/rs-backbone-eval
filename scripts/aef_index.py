#!/usr/bin/env python3
"""Build the AEF bounding-box index for one (year, zone).

    workers/prithvi/.venv/bin/python scripts/aef_index.py --year 2021 --zone 54N

Scanning opens every COG once over HTTPS (4 min for 54N/2021). The resulting
JSON makes ``scripts/aef_probe.py`` resolve a point in milliseconds, which is the
difference between a probe and a dataset path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rs_backbone_eval.data.aef_index import build_index, index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, nargs="+", default=[2021])
    parser.add_argument("--zone", nargs="+", default=["54N"])
    parser.add_argument("--out", type=Path, default=Path("/geodata/rs-eval/data/derived/aef/index"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for year in args.year:
        for zone in args.zone:
            index = build_index(year, zone, args.out, workers=args.workers)
            print(
                json.dumps(
                    {
                        "index": str(index_path(args.out, year, zone)),
                        "year": year,
                        "zone": zone,
                        "cogs": index["cog_count"],
                        "unreadable": len(index["unreadable"]),
                        "crs": index["crs"],
                        "seconds": index["seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
