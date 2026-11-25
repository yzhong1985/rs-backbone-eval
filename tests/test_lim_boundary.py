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


def _scene(h: int, w: int, top: int, left: int, size: int, cls: int) -> np.ndarray:
    """A labelled background (class 2) with one object, so no boundary touches unlabelled 0."""
    out = np.full((h, w), 2, dtype=np.int64)
    out[top : top + size, left : left + size] = cls
    return out


def test_band_is_two_pixels_as_in_plan_v3() -> None:
    assert boundary.BAND_PX == 2


def test_full_sheet_of_one_class_has_no_boundary_from_the_sheet_edge() -> None:
    label = np.ones((200, 200), dtype=np.int64)
    assert boundary.boundary_counts(label, label, [1])[1] == (0, 0)


def test_object_touching_unlabelled_pixels_is_not_scored() -> None:
    label = _box(200, 200, 40, 40, 80, 1)  # surrounded by unlabelled 0
    assert boundary.boundary_counts(label, label, [1])[1] == (0, 0)


def test_predicted_class_on_unlabelled_pixels_is_not_scored() -> None:
    label = _box(200, 200, 40, 40, 80, 1)
    pred = _box(200, 200, 40, 40, 80, 1)
    pred[0:30, 0:30] = 1  # predicted mask spilling onto unlabelled pixels, far from the object
    assert boundary.boundary_counts(label, pred, [1])[1] == (0, 0)


def test_identical_prediction_has_boundary_iou_one() -> None:
    label = _scene(200, 200, 40, 40, 80, 1)
    counts = boundary.boundary_counts(label, label, [1])
    inter, union = counts[1]
    assert inter == union > 0


def test_disjoint_prediction_has_boundary_iou_zero() -> None:
    label = _scene(200, 200, 10, 10, 40, 1)
    pred = _scene(200, 200, 120, 120, 40, 1)
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
    label = _scene(400, 400, 100, 100, 200, 1)
    shifted = _scene(400, 400, 101, 101, 200, 1)  # 1 px shift, band is 2 px
    inter, union = boundary.boundary_counts(label, shifted, [1])[1]
    assert 0 < inter / union < 1


def test_mean_over_classes_and_empty_totals() -> None:
    total: dict[int, list[int]] = {}
    boundary.add_counts(total, {1: (5, 10), 2: (0, 0)})
    boundary.add_counts(total, {1: (5, 10)})
    assert boundary.mean_boundary_iou(total) == pytest.approx(0.5)
    assert boundary.mean_boundary_iou({}) == 0.0


def _plugin(**overrides) -> dict:
    plugin = {
        "format": "t13-fusion-plugin-v1",
        "protocol": "whu_ft_nir",
        "baseline_run": "lim_mit_huge-T13-whu_ft_b0-train-17",
        "baseline_checkpoint_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "config": {"nir": True, "sar": False},
    }
    plugin.update(overrides)
    return plugin


_PIN = {"run_id": "lim_mit_huge-T13-whu_ft_b0-train-17", "checkpoint_sha256": "a" * 64}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "other"}, "format"),
        ({"protocol": "whu_ft_b0"}, "trained for"),
        ({"baseline_run": "lim_mit_huge-T13-whu_ft_b0-train-42"}, "baseline is"),
        ({"baseline_checkpoint_sha256": "c" * 64}, "checkpoint SHA-256"),
        ({"manifest_sha256": "d" * 64}, "another manifest"),
        ({"config": {"nir": False}}, "recipe"),
    ],
)
def test_plugin_must_match_protocol_baseline_and_manifest(overrides, message) -> None:
    from lim_worker.t13_fusion_eval import check_plugin

    with pytest.raises(RuntimeError, match=message):
        check_plugin(
            _plugin(**overrides),
            pin=_PIN,
            manifest_sha="b" * 64,
            protocol_id="whu_ft_nir",
            use_nir=True,
        )


def test_matching_plugin_passes_the_check() -> None:
    from lim_worker.t13_fusion_eval import check_plugin

    check_plugin(_plugin(), pin=_PIN, manifest_sha="b" * 64, protocol_id="whu_ft_nir", use_nir=True)


def test_dilate_matches_a_brute_force_square_neighbourhood() -> None:
    import torch

    gen = torch.Generator().manual_seed(0)
    mask = torch.rand(23, 31, generator=gen) < 0.05
    for radius in (1, 3, 7):
        fast = boundary._dilate(mask, radius)
        brute = torch.zeros_like(mask)
        for i, j in zip(*torch.nonzero(mask, as_tuple=True), strict=True):
            brute[max(0, i - radius) : i + radius + 1, max(0, j - radius) : j + radius + 1] = True
        assert torch.equal(fast, brute), radius


def test_dilate_is_fast_on_a_full_size_sheet() -> None:
    import time

    import torch

    mask = torch.zeros(5555, 5555, dtype=torch.bool)
    mask[2000:2010, 2000:2010] = True
    started = time.time()
    boundary._dilate(mask, 157)  # 2% of a 5555 px sheet diagonal
    assert time.time() - started < 10.0
