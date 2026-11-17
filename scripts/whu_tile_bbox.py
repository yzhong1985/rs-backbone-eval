#!/usr/bin/env python3
"""Decode the WHU-OPT-SAR file names into geographic bounds.

WHU-OPT-SAR tifs carry no CRS, geotransform, GCPs or RPCs — ``rasterio`` opens
them with an identity transform. The georeference lives in the file name, which
is the GB/T 13989-2012 1:50,000 sheet code, e.g. ``NH49E004021.tif``:

* ``N``      northern hemisphere
* ``H``      row letter of the 1:1,000,000 sheet: A = 0-4 N, B = 4-8 N, ...
* ``49``     column of the 1:1,000,000 sheet, counted from 180 W (1 = 180-174 W,
             so 49 = 108-114 E)
* ``E``      1:50,000 (the 1:100,000 prefix is ``D``, 1:25,000 is ``F``)
* ``004``    row inside the 1:1M sheet, 001 = northernmost, 24 rows of 10'
* ``021``    column inside the 1:1M sheet, 001 = westernmost, 24 columns of 15'

Every sheet is therefore exactly 15' x 10' = 0.25 deg x 0.1666667 deg, and the
rasters are 5556 x 3704 px, i.e. 4.49964e-5 deg/px on *both* axes. That equality
is the check that the raster fills the sheet bounds without margins: a 5 m
projected grid would not have equal degrees per pixel in x and y.

Note that the east-west pixel is ~4.33 m on the ground at 30.5 N while the
north-south pixel is ~5.0 m, so "5 m resolution" holds only in latitude.

Usage::

    python scripts/whu_tile_bbox.py NH49E004021
    python scripts/whu_tile_bbox.py --check "The division of the dataset.txt"
    python scripts/whu_tile_bbox.py NH49E004021 --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rs_backbone_eval.data.whu_sheets import (  # noqa: F401  re-exported for tests
    DEG_PER_PX_X,
    DEG_PER_PX_Y,
    SHEET_HEIGHT_DEG,
    SHEET_WIDTH_DEG,
    TILE_HEIGHT_PX,
    TILE_WIDTH_PX,
    Sheet,
    decode,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="sheet file name, e.g. NH49E004021.tif")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--check",
        metavar="TXT",
        help="validate every sheet name in a file (one per line, e.g. the upstream split)",
    )
    args = parser.parse_args(argv)

    if args.check:
        names = [
            line.strip()
            for line in Path(args.check).read_text(encoding="utf-8").splitlines()
            if line.strip().lower().endswith(".tif")
        ]
        sheets = [decode(name) for name in names]
        unique = {sheet.name for sheet in sheets}
        print(f"{len(sheets)} names, {len(unique)} unique sheets")
        lat_s = min(s.south for s in sheets)
        lat_n = max(s.north for s in sheets)
        lon_w = min(s.west for s in sheets)
        lon_e = max(s.east for s in sheets)
        print(f"lat {lat_s:.4f}..{lat_n:.4f}  lon {lon_w:.4f}..{lon_e:.4f}")
        return 0

    if not args.name:
        parser.error("give a sheet name or --check")
    sheet = decode(args.name)
    if args.json:
        print(json.dumps({**asdict(sheet), "bounds": sheet.bounds}, ensure_ascii=False))
    else:
        print(
            f"{sheet.name}: lat {sheet.south:.4f}..{sheet.north:.4f}, "
            f"lon {sheet.west:.4f}..{sheet.east:.4f}"
        )
        print(f"pixel grid {DEG_PER_PX_X:.8e} deg/px (x) x {DEG_PER_PX_Y:.8e} deg/px (y)")
        print(f"affine a,b,c,d,e,f = {sheet.transform()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
