#!/usr/bin/env python3
"""Warp AEF annual embeddings onto each WHU-OPT-SAR sheet grid (T13 teacher).

The WHU rasters have no georeference; their grid is the 1:50,000 sheet in
CGCS2000 geographic coordinates (``rs_backbone_eval.data.whu_sheets``). AEF is
stored as 10 m UTM COGs (49N / 50N, south-up). For every sheet this script
writes one int8 GeoTIFF on the sheet's own grid at ``FACTOR`` image pixels per
AEF cell (2 -> 2778 x 1852 cells, ~8.7 m x 10 m), so a 512 px image tile maps to
an exact 256 x 256 teacher window with no further resampling at train time.

* resampling is nearest: AEF values are quantised int8 (decode
  ``sign(q) * (|q| / 127.5) ** 2``); averaging raw codes would not commute with
  that decode;
* ``-128`` is NoData and stays NoData; the fill fraction (cells where every band
  is -128) is written to ``summary.json`` per sheet;
* source tiles are passed to one gdalwarp call, so a sheet crossing two AEF
  tiles or the 49N/50N boundary is stitched by GDAL (which also handles the
  south-up geotransform).

Usage (needs GDAL CLI and rasterio, e.g. the LIM worker env)::

    workers/lim/.venv/bin/python scripts/whu_aef_warp.py \\
        --tiles /home/yzhong/rs-eval-cache/whu-aef2019/tiles \\
        --index /home/yzhong/rs-eval-cache/whu-aef2019/tiles.json \\
        --out /home/yzhong/rs-eval-cache/whu-aef2019/sheets --workers 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rs_backbone_eval.data.whu_sheets import CRS, TILE_HEIGHT_PX, TILE_WIDTH_PX, decode  # noqa: E402

FACTOR = 2
NODATA = -128


def local_name(url: str) -> str:
    """``gs://.../2019/49N/<id>.tiff`` -> ``49N_<id>.tiff`` (download.sh naming)."""
    parts = url.rstrip("/").split("/")
    return f"{parts[-2]}_{parts[-1]}"


def warp_sheet(sheet_name: str, sources: list[Path], out: Path) -> dict:
    sheet = decode(sheet_name)
    target = out / f"{sheet_name}.tif"
    if not target.exists():
        missing = [str(s) for s in sources if not s.is_file()]
        if missing:
            return {"sheet": sheet_name, "status": "missing_tiles", "missing": missing}
        tmp = target.with_suffix(".tmp.tif")
        subprocess.run(
            [
                "gdalwarp",
                "-q",
                "-overwrite",
                "-t_srs",
                CRS,
                "-te",
                *(f"{v:.10f}" for v in (sheet.west, sheet.south, sheet.east, sheet.north)),
                "-ts",
                str(TILE_WIDTH_PX // FACTOR),
                str(TILE_HEIGHT_PX // FACTOR),
                "-r",
                "near",
                "-srcnodata",
                str(NODATA),
                "-dstnodata",
                str(NODATA),
                "-ot",
                "Int8",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "TILED=YES",
                "-co",
                "INTERLEAVE=BAND",
                "-wo",
                "NUM_THREADS=4",
                *map(str, sources),
                str(tmp),
            ],
            check=True,
        )
        tmp.rename(target)
    with rasterio.open(target) as src:
        data = src.read()
        shape = (src.count, src.height, src.width)
    fill = float(np.mean((data == NODATA).all(axis=0)))
    return {"sheet": sheet_name, "status": "ok", "shape": shape, "fill": round(fill, 6)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="tiles.json (sheet_tiles)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sheets", nargs="*", help="limit to these sheets")
    args = parser.parse_args(argv)

    index = json.loads(args.index.read_text(encoding="utf-8"))["sheet_tiles"]
    names = sorted(args.sheets or index)
    args.out.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> dict:
        sources = [args.tiles / local_name(url) for url in index[name]]
        return warp_sheet(name, sources, args.out)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, names))
    for row in results:
        print(json.dumps(row))
    summary = args.out / "summary.json"
    previous = json.loads(summary.read_text()) if summary.exists() else {}
    previous.update({row["sheet"]: row for row in results})
    summary.write_text(json.dumps(previous, indent=1, sort_keys=True), encoding="utf-8")
    bad = [r for r in results if r["status"] != "ok"]
    print(f"{len(results) - len(bad)} ok, {len(bad)} not ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
