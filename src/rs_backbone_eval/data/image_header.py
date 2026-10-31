"""Image header readers for the manifest builders (no Pillow, no rasterio).

The control plane must never grow a raster dependency just to learn how large a
source image is: FAIR1M ships PNG or TIFF depending on the mirror, iSAID ships
PNG, and both are read here from the file header. Workers keep using rasterio.

Only geometry is parsed — allow-listed fields, no payload.
"""

from __future__ import annotations

import struct
from pathlib import Path


class ImageHeaderError(ValueError):
    """The file is not an image whose size can be read from its header."""


def png_size(path: Path) -> tuple[int, int]:
    """Width/height from a PNG header.

    PNG puts the two integers at a fixed offset in the IHDR chunk.
    """
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ImageHeaderError(f"{path}: not a PNG with an IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ImageHeaderError(f"{path}: invalid PNG size {width}x{height}")
    return int(width), int(height)


def tiff_size(path: Path) -> tuple[int, int]:
    """Width/height by walking IFD0 of a classic (not Big) TIFF.

    Reads tag 256 (ImageWidth) and 257 (ImageLength); both may be SHORT or LONG.
    """
    with Path(path).open("rb") as handle:
        head = handle.read(8)
        if head[:2] == b"II":
            endian = "<"
        elif head[:2] == b"MM":
            endian = ">"
        else:
            raise ImageHeaderError(f"{path}: not a TIFF (bad byte order mark)")
        magic = struct.unpack(endian + "H", head[2:4])[0]
        if magic == 43:
            raise ImageHeaderError(f"{path}: BigTIFF is not supported")
        if magic != 42:
            raise ImageHeaderError(f"{path}: not a TIFF (magic {magic})")
        offset = struct.unpack(endian + "I", head[4:8])[0]
        handle.seek(offset)
        count = struct.unpack(endian + "H", handle.read(2))[0]
        entries = handle.read(12 * count)

    sizes: dict[int, int] = {}
    for index in range(count):
        block = entries[index * 12 : index * 12 + 12]
        if len(block) < 12:
            break
        tag, field_type, fields = struct.unpack(endian + "HHI", block[:8])
        if tag not in (256, 257):
            continue
        if field_type == 3:  # SHORT, inline in the first two bytes
            sizes[tag] = struct.unpack(endian + "H", block[8:10])[0]
        elif field_type == 4:  # LONG, inline
            sizes[tag] = struct.unpack(endian + "I", block[8:12])[0]
    width, height = sizes.get(256), sizes.get(257)
    if not width or not height:
        raise ImageHeaderError(f"{path}: TIFF IFD0 has no ImageWidth/ImageLength")
    return int(width), int(height)


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` for a PNG or classic TIFF, sniffing the header."""
    with Path(path).open("rb") as handle:
        head = handle.read(4)
    if head[:4] == b"\x89PNG":
        return png_size(path)
    if head[:2] in (b"II", b"MM"):
        return tiff_size(path)
    raise ImageHeaderError(f"{path}: expected PNG or TIFF, got {head[:4]!r}")


# GeoTIFF tags that describe where the raster sits on the ground. Only these
# four are read: the manifest needs a CRS and an affine transform, and the
# control plane must not grow rasterio to learn them.
_GEO_PIXEL_SCALE = 33550
_GEO_TIEPOINT = 33922
_GEO_TRANSFORM = 34264
_GEO_KEYS = 34735
_TYPE_BYTES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
_TYPE_FORMAT = {1: "B", 3: "H", 4: "I", 11: "f", 12: "d"}
# GeoKeyDirectory entries: 3072 ProjectedCSTypeGeoKey, 2048 GeographicTypeGeoKey.
_PROJECTED_EPSG = 3072
_GEOGRAPHIC_EPSG = 2048


def _tiff_geo_values(path: Path) -> dict[int, list[float]]:
    """Raw values of the four GeoTIFF tags in IFD0 (empty when absent)."""
    endian = "<"
    with Path(path).open("rb") as handle:
        head = handle.read(8)
        if head[:2] == b"II":
            endian = "<"
        elif head[:2] == b"MM":
            endian = ">"
        else:
            raise ImageHeaderError(f"{path}: not a TIFF (bad byte order mark)")
        magic = struct.unpack(endian + "H", head[2:4])[0]
        if magic == 43:
            raise ImageHeaderError(f"{path}: BigTIFF is not supported")
        if magic != 42:
            raise ImageHeaderError(f"{path}: not a TIFF (magic {magic})")
        offset = struct.unpack(endian + "I", head[4:8])[0]
        handle.seek(offset)
        count = struct.unpack(endian + "H", handle.read(2))[0]
        entries = handle.read(12 * count)

        values: dict[int, list[float]] = {}
        for index in range(count):
            block = entries[index * 12 : index * 12 + 12]
            if len(block) < 12:
                break
            tag, field_type, fields = struct.unpack(endian + "HHI", block[:8])
            if tag not in (_GEO_PIXEL_SCALE, _GEO_TIEPOINT, _GEO_TRANSFORM, _GEO_KEYS):
                continue
            width = _TYPE_BYTES.get(field_type)
            fmt = _TYPE_FORMAT.get(field_type)
            if width is None or fmt is None:
                continue
            total = width * fields
            if total <= 4:
                payload = block[8 : 8 + total]
            else:
                position = struct.unpack(endian + "I", block[8:12])[0]
                here = handle.tell()
                handle.seek(position)
                payload = handle.read(total)
                handle.seek(here)
            if len(payload) < total:
                continue
            values[tag] = list(struct.unpack(endian + fmt * fields, payload))
    return values


def tiff_geo_reference(path: Path) -> tuple[str | None, list[float] | None]:
    """``(crs, affine)`` from a GeoTIFF's IFD0, or ``(None, None)`` when absent.

    ``affine`` is the GDAL six-tuple ``[a, b, c, d, e, f]`` mapping pixel
    ``(col, row)`` to ``(x, y)``. The CRS string is ``EPSG:<code>`` when the
    GeoKey directory carries one (projected keys win over geographic ones); a
    mirror without GeoKeys keeps ``None`` rather than guessing.
    """
    values = _tiff_geo_values(path)
    affine: list[float] | None = None
    if _GEO_TRANSFORM in values and len(values[_GEO_TRANSFORM]) >= 16:
        matrix = values[_GEO_TRANSFORM]
        affine = [matrix[0], matrix[1], matrix[3], matrix[4], matrix[5], matrix[7]]
    elif (
        _GEO_PIXEL_SCALE in values
        and _GEO_TIEPOINT in values
        and len(values[_GEO_PIXEL_SCALE]) >= 2
        and len(values[_GEO_TIEPOINT]) >= 6
    ):
        scale_x, scale_y = values[_GEO_PIXEL_SCALE][0], values[_GEO_PIXEL_SCALE][1]
        tie = values[_GEO_TIEPOINT]
        i, j, x, y = tie[0], tie[1], tie[3], tie[4]
        affine = [
            scale_x,
            0.0,
            x - i * scale_x,
            0.0,
            -scale_y,
            y + j * scale_y,
        ]
    code: int | None = None
    keys = values.get(_GEO_KEYS)
    if keys:
        for index in range(4, len(keys) - 3, 4):
            key, location, _count, value = (
                int(keys[index]),
                int(keys[index + 1]),
                int(keys[index + 2]),
                int(keys[index + 3]),
            )
            if location != 0 or value in (0, 32767):
                continue
            if key == _PROJECTED_EPSG or (key == _GEOGRAPHIC_EPSG and code is None):
                code = value
    return (f"EPSG:{code}" if code else None), affine
