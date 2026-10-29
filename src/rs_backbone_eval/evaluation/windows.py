"""Whole-image sliding-window inference for tile-level datasets.

iSAID's official mIoU is computed over every pixel of the source image, so test
time must cover the image completely — scoring the manifest tiles as they come
would count border pixels once per overlapping tile and skip nothing but count
some twice, and would score padding. The rules (review 2026-10-29):

* the same tile grid as the manifest (``data.tiling``), so train and test see
  the same geometry;
* overlapping logits are **averaged before the argmax**, not voted per tile —
  a pixel covered by four tiles gets one prediction, not four;
* padding is dropped *before* the confusion matrix: a 387x278 image scores
  exactly 387*278 pixels, not 1024*1024.

The module is numpy-only on purpose. ``predict`` returns probabilities/logits
for one tile, so the caller can keep torch/rasterio inside the worker and the
control plane can test the geometry without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from ..data.tiling import tile_origins
from .confusion import ConfusionMatrix

# ``predict(window)`` returns ``[num_classes, crop, crop]`` values for the tile
# whose top-left corner is ``window[:2]``. The tile may extend past the image;
# the callback pads the read with zeros (padding is a valid model context, it
# is just never scored).
TilePredictor = Callable[[tuple[int, int, int, int]], np.ndarray]


def _finalise(logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if not np.all(counts > 0):
        raise ValueError("sliding window left pixels uncovered; check crop/stride")
    return np.argmax(logits / counts[None, :, :], axis=0)


def predict_bands(
    width: int,
    height: int,
    *,
    num_classes: int,
    crop: int,
    stride: int,
    predict: TilePredictor,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(row_start, labels[rows, width])``, each row exactly once.

    Tiles are processed one horizontal band at a time. A band buffer holds rows
    ``[y, y + crop)``; after switching to the next band the rows no longer
    touched by any tile are averaged and emitted, then the buffer is shifted.
    Peak memory is ``num_classes * crop * width`` floats, independent of the
    image height.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image must be non-empty, got {width}x{height}")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if crop <= 0 or stride <= 0 or stride > crop:
        raise ValueError("crop/stride must be positive and stride <= crop")

    xs = tile_origins(width, crop, stride)
    ys = tile_origins(height, crop, stride)
    canvas = np.zeros((num_classes, crop, width), dtype=np.float32)
    counts = np.zeros((crop, width), dtype=np.float32)

    previous = ys[0]
    for y in ys:
        if y > previous:
            rows = min(y, height) - previous
            yield previous, _finalise(canvas[:, :rows], counts[:rows])
            shift = y - previous
            if shift >= crop:
                canvas.fill(0.0)
                counts.fill(0.0)
            else:
                canvas[:, : crop - shift] = canvas[:, shift:]
                canvas[:, crop - shift :] = 0.0
                counts[: crop - shift] = counts[shift:]
                counts[crop - shift :] = 0.0
            previous = y

        for x in xs:
            logits = np.asarray(predict((x, y, crop, crop)), dtype=np.float32)
            if logits.shape != (num_classes, crop, crop):
                raise ValueError(
                    f"predict returned {logits.shape}, expected {(num_classes, crop, crop)}"
                )
            span = min(crop, width - x)
            canvas[:, :, x : x + span] += logits[:, :, :span]
            counts[:, x : x + span] += 1.0

    rows = height - previous
    yield previous, _finalise(canvas[:, :rows], counts[:rows])


def predict_image(
    width: int,
    height: int,
    *,
    num_classes: int,
    crop: int,
    stride: int,
    predict: TilePredictor,
) -> np.ndarray:
    """Assemble the streamed bands into one ``[height, width]`` label image."""
    labels = np.empty((height, width), dtype=np.int64)
    for start, rows in predict_bands(
        width, height, num_classes=num_classes, crop=crop, stride=stride, predict=predict
    ):
        labels[start : start + rows.shape[0]] = rows
    return labels


def evaluate_image(
    label: np.ndarray,
    *,
    num_classes: int,
    crop: int,
    stride: int,
    predict: TilePredictor,
    ignore_index: int | None = None,
    confusion: ConfusionMatrix | None = None,
) -> ConfusionMatrix:
    """Score one whole image into a confusion matrix, padding excluded.

    ``label`` is the full-resolution ground truth; ``ignore_index`` removes any
    deliberately unlabelled pixels (iSAID has none — padding lives outside the
    image and is never part of this array).
    """
    truth = np.asarray(label)
    if truth.ndim != 2:
        raise ValueError(f"label must be [height, width], got shape {truth.shape}")
    height, width = truth.shape
    matrix = confusion if confusion is not None else ConfusionMatrix(num_classes)
    if matrix.num_classes != num_classes:
        raise ValueError(f"confusion has {matrix.num_classes} classes, expected {num_classes}")
    for start, rows in predict_bands(
        width, height, num_classes=num_classes, crop=crop, stride=stride, predict=predict
    ):
        matrix.update(truth[start : start + rows.shape[0]], rows, ignore_index=ignore_index)
    return matrix
