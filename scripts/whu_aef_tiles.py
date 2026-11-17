#!/usr/bin/env python3
"""List the AEF annual tiles that cover the 100 WHU-OPT-SAR sheets.

Writes ``tiles.txt`` (one gs:// URL per line, input of whu_aef_download.sh) and
``tiles.json`` (``sheet_tiles``: sheet -> covering URLs, input of
whu_aef_warp.py). For 2019 this is 32 tiles (22 in 49N, 10 in 50N); 51 sheets
sit in one tile, 42 in two, 7 in three or four.

    uv run --with pyarrow --with shapely python scripts/whu_aef_tiles.py \\
        --optical /home/yzhong/rs-eval-data/whu-opt-sar/optical \\
        --out /home/yzhong/rs-eval-cache/whu-aef2019 --year 2019
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from shapely import box, wkb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rs_backbone_eval.data.whu_sheets import decode  # noqa: E402

DEFAULT_INDEX = Path("/geodata/rs-eval/data/derived/aef/aef_index.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args(argv)

    sheets = [decode(p.stem) for p in sorted(args.optical.glob("*.tif"))]
    west = min(s.west for s in sheets) - 1
    east = max(s.east for s in sheets) + 1
    south = min(s.south for s in sheets) - 1
    north = max(s.north for s in sheets) + 1
    rows = [
        r
        for r in pq.read_table(args.index, filters=[("year", "=", args.year)]).to_pylist()
        if r["wgs84_east"] > west
        and r["wgs84_west"] < east
        and r["wgs84_north"] > south
        and r["wgs84_south"] < north
    ]
    geoms = [(r["path"], wkb.loads(r["geom"])) for r in rows]
    cover = {
        s.name: sorted(path for path, geom in geoms if geom.intersects(box(*s.bounds)))
        for s in sheets
    }
    empty = [name for name, paths in cover.items() if not paths]
    if empty:
        raise SystemExit(f"no AEF {args.year} tile covers: {empty}")
    tiles = sorted({p for paths in cover.values() for p in paths})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tiles.txt").write_text("".join(f"{t}\n" for t in tiles), encoding="utf-8")
    (args.out / "tiles.json").write_text(
        json.dumps({"year": args.year, "tiles": tiles, "sheet_tiles": cover}, indent=1),
        encoding="utf-8",
    )
    print(f"{len(sheets)} sheets -> {len(tiles)} AEF {args.year} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
