"""Sliding-window evaluation tests (P2, review 2026-10-29).

Two properties the protocol depends on and that are easy to lose:

* every pixel of the source image is scored exactly once — iSAID's official
  mIoU is over the whole image, so a 387x278 image must contribute 107,466
  pixels, not the 1,048,576 of its padded 1024 tile;
* overlapping tiles are averaged before the argmax, and padding never changes
  a real pixel's vote.
"""

from __future__ import annotations

import numpy as np

from rs_backbone_eval.data.tiling import pad_tile, tile_bounds, tile_origins
from rs_backbone_eval.evaluation.windows import evaluate_image, predict_bands, predict_image


def constant_predictor(num_classes: int, crop: int, klass: int = 0):
    def predict(window: tuple[int, int, int, int]) -> np.ndarray:
        logits = np.zeros((num_classes, crop, crop), dtype=np.float32)
        logits[klass] = 1.0
        return logits

    return predict


def field_predictor(field: np.ndarray, crop: int):
    """Slice a pre-computed ``[C, H, W]`` field, zero-padding outside."""

    def predict(window: tuple[int, int, int, int]) -> np.ndarray:
        x, y, _, _ = window
        classes, height, width = field.shape
        tile = np.zeros((classes, crop, crop), dtype=np.float32)
        span_y = min(crop, height - y)
        span_x = min(crop, width - x)
        tile[:, :span_y, :span_x] = field[:, y : y + span_y, x : x + span_x]
        return tile

    return predict


def reference_labels(field: np.ndarray, crop: int, stride: int) -> np.ndarray:
    """Brute-force whole-image average over a full canvas."""
    classes, height, width = field.shape
    canvas = np.zeros((classes, height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for y in tile_origins(height, crop, stride):
        for x in tile_origins(width, crop, stride):
            span_y = min(crop, height - y)
            span_x = min(crop, width - x)
            canvas[:, y : y + span_y, x : x + span_x] += field[:, y : y + span_y, x : x + span_x]
            counts[y : y + span_y, x : x + span_x] += 1
    return np.argmax(canvas / counts, axis=0)


def test_pad_tile_fills_with_ignore_index() -> None:
    tile = np.full((300, 200, 3), 7, dtype=np.uint8)
    padded = pad_tile(tile, 1024, fill=255)
    assert padded.shape == (1024, 1024, 3)
    assert padded[299, 199].tolist() == [7, 7, 7]
    assert padded[300, 0].tolist() == [255, 255, 255]
    assert padded[1023, 1023].tolist() == [255, 255, 255]


def test_tile_bounds_clip_the_last_tile() -> None:
    assert tile_bounds([512, 512, 1024, 1024], 600, 550) == (512, 512, 600, 550)


def test_small_image_scores_every_pixel_once() -> None:
    """A 387x278 image padded to 1024 must count 107,466 pixels, not 1,048,576."""
    height, width = 278, 387
    truth = np.zeros((height, width), dtype=np.int64)
    matrix = evaluate_image(
        truth,
        num_classes=2,
        crop=1024,
        stride=512,
        predict=constant_predictor(2, 1024),
    )
    assert matrix.total == height * width


def test_overlapping_logits_are_averaged_before_argmax() -> None:
    """Two tiles vote differently; the mean decides, not the last tile."""
    width, height, crop, stride = 768, 512, 512, 256

    def predict(window: tuple[int, int, int, int]) -> np.ndarray:
        x = window[0]
        logits = np.zeros((2, crop, crop), dtype=np.float32)
        if x == 0:
            logits[0] = 1.0  # confident class 0
        else:
            logits[1] = 0.4  # weaker class 1
        return logits

    labels = predict_image(width, height, num_classes=2, crop=crop, stride=stride, predict=predict)
    # the overlap is (0.5, 0.2) -> class 0; a last-wins implementation says class 1
    assert labels[:, 300].tolist() == [0] * height
    assert labels[:, 100].tolist() == [0] * height


def test_padding_cannot_flip_a_real_pixel() -> None:
    """Logits produced over the padded region are dropped, not averaged in."""
    width, height, crop, stride = 700, 500, 512, 256

    def predict(window: tuple[int, int, int, int]) -> np.ndarray:
        x, y, _, _ = window
        logits = np.zeros((2, crop, crop), dtype=np.float32)
        rows = np.arange(y, y + crop)[:, None]
        cols = np.arange(x, x + crop)[None, :]
        outside = (rows >= height) | (cols >= width)
        logits[1][outside] = 1e6
        return logits

    labels = predict_image(width, height, num_classes=2, crop=crop, stride=stride, predict=predict)
    assert labels.shape == (height, width)
    assert labels.tolist() == [[0] * width for _ in range(height)]


def test_streamed_bands_match_a_full_canvas() -> None:
    """The band shift must not drop or double-count the overlap rows."""
    rng = np.random.default_rng(20261029)
    width, height, crop, stride, classes = 2000, 1700, 1024, 512, 3
    field = rng.random((classes, height, width), dtype=np.float32)

    streamed = predict_image(
        width,
        height,
        num_classes=classes,
        crop=crop,
        stride=stride,
        predict=field_predictor(field, crop),
    )
    expected = reference_labels(field, crop, stride)
    assert streamed.shape == expected.shape
    assert np.array_equal(streamed, expected)


def test_evaluate_image_skips_ignored_labels() -> None:
    height, width = 600, 900
    truth = np.zeros((height, width), dtype=np.int64)
    truth[:, :200] = 255  # e.g. a no-data strip the dataset declares
    matrix = evaluate_image(
        truth,
        num_classes=2,
        crop=512,
        stride=256,
        predict=constant_predictor(2, 512),
        ignore_index=255,
    )
    assert matrix.total == height * (width - 200)


def test_bands_cover_each_row_exactly_once() -> None:
    width, height, crop, stride = 1500, 1300, 1024, 512
    covered = 0
    for start, rows in predict_bands(
        width,
        height,
        num_classes=2,
        crop=crop,
        stride=stride,
        predict=constant_predictor(2, crop),
    ):
        assert start == covered
        covered += rows.shape[0]
    assert covered == height
