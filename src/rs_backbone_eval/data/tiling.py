"""Whole-image tiling shared by the manifest builders and the evaluators.

Two rules the iSAID protocol depends on (review 2026-10-29):

* every pixel is covered — positions run ``ceil((dim - crop) / stride) + 1`` and
  the last tile is padded, never dropped. An earlier version skipped images
  whose short side was below the crop (418/1411 on iSAID train) and stopped at
  the last whole stride, leaving ~20% of the pixels unseen;
* a tile may be smaller than ``crop`` because the source image is — it is padded
  whole, and ``valid_fraction`` says how much of it is real data.

The functions live here (not in the builder) so the builder, the P0-a audit and
the sliding-window evaluator all use one implementation.
"""

from __future__ import annotations

import math

import numpy as np


def tile_origins(length: int, crop: int, stride: int) -> list[int]:
    """Top-left offsets along one axis, padding the final tile."""
    if length <= 0 or crop <= 0 or stride <= 0:
        return []
    if length <= crop:
        return [0]
    count = math.ceil((length - crop) / stride) + 1
    return [index * stride for index in range(count)]


def tiles_padded(width: int, height: int, crop: int, stride: int) -> int:
    """Number of tiles needed to cover the whole image."""
    return len(tile_origins(width, crop, stride)) * len(tile_origins(height, crop, stride))


def tiles_covering(width: int, height: int, crop: int, stride: int) -> list[tuple[int, int]]:
    """``(x, y)`` top-left corners in row-major order."""
    return [
        (x, y)
        for y in tile_origins(height, crop, stride)
        for x in tile_origins(width, crop, stride)
    ]


def valid_fraction(
    width: int, height: int, crop: int, stride: int, origin: tuple[int, int]
) -> float:
    """Share of a tile that comes from the source image rather than padding."""
    x, y = origin
    real_w = min(crop, max(0, width - x))
    real_h = min(crop, max(0, height - y))
    return (real_w * real_h) / float(crop * crop)


def crop_from_tile(crop: int, origin: tuple[int, int]) -> list[int]:
    """``[x, y, w, h]`` window for a manifest row."""
    return [origin[0], origin[1], crop, crop]


def tile_bounds(
    window: list[int] | tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clip a ``[x, y, w, h]`` window to the source image.

    The last tile of an image normally runs past its right/bottom edge; the
    worker reads only the clipped region and pads the rest.
    """
    if len(window) != 4:
        raise ValueError(f"window must be [x, y, w, h], got {window!r}")
    x, y, w, h = (int(value) for value in window)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"invalid window {window!r}")
    return x, y, min(x + w, width), min(y + h, height)


def pad_tile(array: np.ndarray, crop: int, fill: int | float = 0) -> np.ndarray:
    """Pad a tile to ``crop x crop`` at the bottom-right with ``fill``.

    Workers pass ``fill=ignore_index`` for labels and ``fill=0`` for imagery,
    so a padded pixel never enters the loss or the mIoU (review 2026-10-29).
    """
    values = np.asarray(array)
    if values.ndim < 2:
        raise ValueError(f"tile must be at least 2-D, got shape {values.shape}")
    height, width = values.shape[:2]
    if height > crop or width > crop:
        raise ValueError(f"tile {values.shape} is larger than the {crop}px crop")
    if (height, width) == (crop, crop):
        return values
    padded = np.full((crop, crop, *values.shape[2:]), fill, dtype=values.dtype)
    padded[:height, :width] = values
    return padded
