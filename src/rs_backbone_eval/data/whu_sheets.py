"""WHU-OPT-SAR sheet names -> geographic bounds (GB/T 13989-2012 1:50,000).

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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The georeference is CGCS2000 geographic; the sheet grid is defined in it.
CRS = "EPSG:4490"
SHEET_WIDTH_DEG = 0.25  # 15'
SHEET_HEIGHT_DEG = 1 / 6  # 10'
TILE_WIDTH_PX = 5556
TILE_HEIGHT_PX = 3704
DEG_PER_PX_X = SHEET_WIDTH_DEG / TILE_WIDTH_PX
DEG_PER_PX_Y = SHEET_HEIGHT_DEG / TILE_HEIGHT_PX

_NAME = re.compile(r"^N([A-V])(\d{2})E(\d{3})(\d{3})(?:\.tif)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Sheet:
    name: str
    north: float
    south: float
    west: float
    east: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def pixel_centre(self, col: int, row: int) -> tuple[float, float]:
        """Lon/lat of the centre of pixel ``(col, row)`` (row 0 = northern edge)."""
        if not (0 <= col < TILE_WIDTH_PX and 0 <= row < TILE_HEIGHT_PX):
            raise ValueError(f"pixel ({col}, {row}) outside {TILE_WIDTH_PX}x{TILE_HEIGHT_PX}")
        return (
            self.west + (col + 0.5) * DEG_PER_PX_X,
            self.north - (row + 0.5) * DEG_PER_PX_Y,
        )

    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL/``rasterio.Affine`` order: (a, b, c, d, e, f) with x=a*col+b*row+c."""
        return (DEG_PER_PX_X, 0.0, self.west, 0.0, -DEG_PER_PX_Y, self.north)


def _clean(name: str) -> str:
    stem = Path(name.strip()).name
    if stem.endswith(".tif"):
        stem = stem[:-4]
    # The public release carries a "(1)" duplicate of every sheet.
    return re.sub(r"\(\d+\)$", "", stem)


def decode(name: str) -> Sheet:
    """Decode a WHU-OPT-SAR sheet file name into its geographic bounds."""
    stem = _clean(name)
    match = _NAME.match(stem)
    if not match:
        raise ValueError(f"not a WHU-OPT-SAR 1:50,000 sheet name: {name!r}")
    row_letter, column_1m, row_50k, column_50k = (
        match.group(1).upper(),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )
    if not (1 <= row_50k <= 24 and 1 <= column_50k <= 24):
        raise ValueError(
            f"{name!r}: 1:50,000 index must be 001..024, got {row_50k:03d}{column_50k:03d}"
        )
    if not (1 <= column_1m <= 60):
        raise ValueError(f"{name!r}: 1:1,000,000 column must be 01..60")

    north = (ord(row_letter) - ord("A") + 1) * 4.0 - (row_50k - 1) * SHEET_HEIGHT_DEG
    west = (column_1m - 1) * 6.0 - 180.0 + (column_50k - 1) * SHEET_WIDTH_DEG
    return Sheet(
        name=stem,
        north=north,
        south=north - SHEET_HEIGHT_DEG,
        west=west,
        east=west + SHEET_WIDTH_DEG,
    )
