#!/usr/bin/env python3
"""Wrap a GF-2 GeoTIFF plus its ``.rpb`` sidecar into an RPC-carrying VRT.

GDAL does not read the GF-2 ``.rpb`` files on its own: ``gdalinfo`` shows no
RPC domain and ``gdalwarp -rpc`` fails with "Unable to compute a RPC based
transformation". Wrapping both files in a VRT whose metadata holds the 16 RPC
tags that GDAL expects makes ``gdalwarp -rpc`` work unchanged::

    python scripts/fbp_rpc_vrt.py SCENE.tiff SCENE.rpb SCENE_rpc.vrt
    gdalwarp -rpc -t_srs EPSG:32650 -tr 10 10 -tap -to RPC_HEIGHT=<HEIGHT_OFF> \
        SCENE_rpc.vrt SCENE_utm50.tif

Two details matter for the FBP scenes and are why this is a script and not a
one-liner:

* GDAL's RPC transformer uses **height 0** when no ``RPC_DEM``/``RPC_HEIGHT``
  is given; it does not fall back to ``HEIGHT_OFF``. For the 4249 m scene that
  is a 197 m (19.7 px at 10 m) shift, so pass ``RPC_HEIGHT`` explicitly or a DEM.
* the caller must warp onto the common 10 m grid (``-tr 10 10 -tap``): AEF and
  OlmoEarth tiles are both 10 m rasters whose pixel edges sit on multiples of
  10 in the same UTM zone, so ``-tap`` puts all three on identical pixel edges.

The size/band/dtype lookup uses ``gdalinfo -json`` (``gdal-bin`` must be
installed); the RPC parsing itself is dependency-free so it can be tested
without GDAL.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# GF-2 .rpb scalar name -> GDAL RPC metadata tag.
SCALARS = {
    "errBias": "ERR_BIAS",
    "errRand": "ERR_RAND",
    "lineOffset": "LINE_OFF",
    "sampOffset": "SAMP_OFF",
    "latOffset": "LAT_OFF",
    "longOffset": "LONG_OFF",
    "heightOffset": "HEIGHT_OFF",
    "lineScale": "LINE_SCALE",
    "sampScale": "SAMP_SCALE",
    "latScale": "LAT_SCALE",
    "longScale": "LONG_SCALE",
    "heightScale": "HEIGHT_SCALE",
}
# GF-2 .rpb tuple name -> GDAL RPC coefficient tag (20 coefficients each).
COEFFICIENTS = {
    "lineNumCoef": "LINE_NUM_COEFF",
    "lineDenCoef": "LINE_DEN_COEFF",
    "sampNumCoef": "SAMP_NUM_COEFF",
    "sampDenCoef": "SAMP_DEN_COEFF",
}
NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")


def parse_rpb(text: str) -> dict[str, list[float]]:
    """Parse a GF-2 ``.rpb`` file into ``name -> numbers``.

    Scalars appear as ``key = 1.0;`` and coefficient groups as
    ``key = (a, b, ... 20 values ...);``; both are flattened to a list so the
    caller can check the length.
    """
    values: dict[str, list[float]] = {}
    for name, body in re.findall(r"(\w+)\s*=\s*\(([^)]*)\)", text, re.S):
        values[name] = [float(item) for item in NUMBER.findall(body)]
    for name, body in re.findall(r"^\s*(\w+)\s*=\s*([^;\n(]+);", text, re.M):
        values.setdefault(name, [float(item) for item in NUMBER.findall(body)])
    return values


def rpc_metadata(values: dict[str, list[float]]) -> dict[str, str]:
    """Map parsed RPB values onto the 16 GDAL RPC tags."""
    meta: dict[str, str] = {}
    for source, tag in SCALARS.items():
        meta[tag] = f"{values[source][0]:.10f}"
    for source, tag in COEFFICIENTS.items():
        coefficients = values[source]
        if len(coefficients) != 20:
            raise ValueError(f"{source} has {len(coefficients)} coefficients, expected 20")
        meta[tag] = " ".join(f"{item:.15e}" for item in coefficients)
    return meta


def build_vrt(
    meta: dict[str, str], source: Path, width: int, height: int, bands: int, dtype: str
) -> str:
    """Render a VRT that reads ``source`` and carries the RPC metadata domain."""
    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        '  <Metadata domain="RPC">',
    ]
    for tag, value in meta.items():
        lines.append(f'    <MDI key="{tag}">{value}</MDI>')
    lines.append("  </Metadata>")
    for band in range(1, bands + 1):
        lines += [
            f'  <VRTRasterBand dataType="{dtype}" band="{band}">',
            "    <SimpleSource>",
            f'      <SourceFilename relativeToVRT="0">{source}</SourceFilename>',
            f"      <SourceBand>{band}</SourceBand>",
            "    </SimpleSource>",
            "  </VRTRasterBand>",
        ]
    lines.append("</VRTDataset>")
    return "\n".join(lines) + "\n"


def raster_shape(path: Path) -> tuple[int, int, int, str]:
    """``(width, height, bands, dtype)`` from ``gdalinfo -json``."""
    output = subprocess.run(
        ["gdalinfo", "-json", str(path)], check=True, capture_output=True, text=True
    ).stdout
    info = json.loads(output)
    width, height = info["size"]
    bands = info["bands"]
    return width, height, len(bands), bands[0]["type"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("rpb", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    meta = rpc_metadata(parse_rpb(args.rpb.read_text(encoding="utf-8", errors="ignore")))
    width, height, bands, dtype = raster_shape(args.image)
    args.output.write_text(
        build_vrt(meta, args.image.resolve(), width, height, bands, dtype), encoding="utf-8"
    )
    print(f"wrote {args.output} ({width}x{height}, {bands} bands, {len(meta)} RPC tags)")


if __name__ == "__main__":
    main()
