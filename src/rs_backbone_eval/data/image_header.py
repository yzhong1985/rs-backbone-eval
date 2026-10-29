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
