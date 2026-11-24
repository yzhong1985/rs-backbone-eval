"""CPU tests for the T13 Boundary IoU helper (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_boundary.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
from lim_worker import boundary  # noqa: E402


def _box(h: int, w: int, top: int, left: int, size: int, cls: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.int64)
    out[top : top + size, left : left + size] = cls
    return out


def test_band_width_is_two_percent_of_the_diagonal() -> None:
    assert boundary.band_width(1000, 0) == 20
    assert boundary.band_width(64, 64) == 2
    assert boundary.band_width(1, 1) == 1  # never below one pixel


def test_identical_prediction_has_boundary_iou_one() -> None:
    label = _box(200, 200, 40, 40, 80, 1)
    counts = boundary.boundary_counts(label, label, [1])
    inter, union = counts[1]
    assert inter == union > 0


def test_disjoint_prediction_has_boundary_iou_zero() -> None:
    label = _box(200, 200, 10, 10, 40, 1)
    pred = _box(200, 200, 120, 120, 40, 1)
    inter, _ = boundary.boundary_counts(label, pred, [1])[1]
    assert inter == 0


def test_background_pixels_are_ignored() -> None:
    label = _box(200, 200, 40, 40, 80, 1)
    pred = label.copy()
    pred[0:10, 0:10] = 1  # predicted class inside a background-only corner
    counts = boundary.boundary_counts(label, pred, [1])
    inter, union = counts[1]
    assert inter == union  # the corner is outside the labelled region


def test_small_shift_costs_some_but_not_all_boundary_overlap() -> None:
    label = _box(400, 400, 100, 100, 200, 1)
    shifted = _box(400, 400, 103, 103, 200, 1)  # 3 px shift, band is 2% of ~566 px = 11 px
    inter, union = boundary.boundary_counts(label, shifted, [1])[1]
    assert 0 < inter / union < 1


def test_mean_over_classes_and_empty_totals() -> None:
    total: dict[int, list[int]] = {}
    boundary.add_counts(total, {1: (5, 10), 2: (0, 0)})
    boundary.add_counts(total, {1: (5, 10)})
    assert boundary.mean_boundary_iou(total) == pytest.approx(0.5)
    assert boundary.mean_boundary_iou({}) == 0.0
