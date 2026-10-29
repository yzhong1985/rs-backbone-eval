"""Matching, COCO AP and rotated NMS: the numbers a FAIR1M cell is judged by.

Hand-computed expectations, because an AP implementation that is merely
self-consistent can still differ from the published numbers by several points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rs_backbone_eval.detection.ap import Predictions, Targets, average_precision, evaluate
from rs_backbone_eval.detection.geometry import obb2poly, polygon_iou
from rs_backbone_eval.detection.nms import Detection, merge_tiles, rotated_nms


def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_average_precision_of_a_perfect_and_a_half_recall_curve() -> None:
    # one positive, one true detection: every recall level is reachable at 1.0
    assert average_precision(np.array([True]), np.array([False]), 1) == pytest.approx(1.0)
    # two positives, one hit then a false positive: recall 0.5 at precision 1,
    # then precision falls to 0.5 -- the 101 levels split 51/50
    value = average_precision(np.array([True, False]), np.array([False, True]), 2)
    assert value == pytest.approx(51 / 101)
    # no positives at all is "not applicable", not zero
    assert average_precision(np.array([True]), np.array([False]), 0) is None


def test_perfect_predictions_score_one() -> None:
    targets = {"scene": Targets()}
    targets["scene"].append(rect(0, 0, 10, 10), 0)
    targets["scene"].append(rect(20, 20, 30, 30), 1)
    predictions = {"scene": Predictions()}
    predictions["scene"].append(rect(0, 0, 10, 10), 0, 0.9)
    predictions["scene"].append(rect(20, 20, 30, 30), 1, 0.8)

    result = evaluate(predictions, targets, num_classes=2)
    assert result["map"] == pytest.approx(1.0)
    assert result["map_50"] == pytest.approx(1.0)
    assert result["map_75"] == pytest.approx(1.0)
    assert result["classes_with_targets"] == 2


def test_a_missed_object_costs_recall_not_the_other_class() -> None:
    targets = {"scene": Targets()}
    targets["scene"].append(rect(0, 0, 10, 10), 0)
    targets["scene"].append(rect(20, 20, 30, 30), 0)
    predictions = {"scene": Predictions()}
    predictions["scene"].append(rect(0, 0, 10, 10), 0, 0.9)

    result = evaluate(predictions, targets, num_classes=1)
    # recall 0.5 at precision 1.0 -> 51 of 101 levels
    assert result["map_50"] == pytest.approx(51 / 101)


def test_a_wrong_class_is_a_false_positive_and_a_miss() -> None:
    targets = {"scene": Targets()}
    targets["scene"].append(rect(0, 0, 10, 10), 0)
    predictions = {"scene": Predictions()}
    predictions["scene"].append(rect(0, 0, 10, 10), 1, 0.9)

    result = evaluate(predictions, targets, num_classes=2)
    assert result["map_50"] == pytest.approx(0.0)
    # the class with no ground truth is excluded, not averaged in as a zero
    assert result["classes_with_targets"] == 1


def test_ignored_ground_truth_absorbs_detections_without_penalty() -> None:
    targets = {"scene": Targets()}
    targets["scene"].append(rect(0, 0, 10, 10), 0)  # trainable
    targets["scene"].append(rect(50, 50, 60, 60), 0, ignored=True)  # cut by a tile edge
    predictions = {"scene": Predictions()}
    predictions["scene"].append(rect(0, 0, 10, 10), 0, 0.9)
    predictions["scene"].append(rect(50, 50, 60, 60), 0, 0.8)  # lands on the ignored object

    result = evaluate(predictions, targets, num_classes=1)
    assert result["map_50"] == pytest.approx(1.0)


def test_size_bins_count_recall_at_iou_half() -> None:
    targets = {"scene": Targets()}
    targets["scene"].append(rect(0, 0, 10, 10), 0)  # 100 px² -> small
    targets["scene"].append(rect(0, 0, 200, 200), 0)  # 40000 px² -> large
    predictions = {"scene": Predictions()}
    predictions["scene"].append(rect(0, 0, 10, 10), 0, 0.9)

    result = evaluate(predictions, targets, num_classes=1)
    assert result["recall_small"] == pytest.approx(1.0)
    assert result["recall_large"] == pytest.approx(0.0)
    assert result["recall_medium"] == pytest.approx(0.0)


def test_predictions_for_an_unknown_scene_are_rejected() -> None:
    with pytest.raises(ValueError, match="without targets"):
        evaluate({"nope": Predictions()}, {"scene": Targets()}, num_classes=1)


def test_rotated_nms_suppresses_duplicates_and_keeps_neighbours() -> None:
    polygons = [rect(0, 0, 10, 10), rect(0.5, 0.5, 10.5, 10.5), rect(40, 40, 50, 50)]
    kept = rotated_nms(polygons, [0.9, 0.8, 0.7], 0.5)
    assert kept == [0, 2]


def test_rotated_nms_uses_polygon_iou_not_the_hull() -> None:
    # two long diagonal boxes cross each other; their axis-aligned hulls overlap
    # almost completely, the polygons themselves do not
    first = obb2poly(0, 0, 20, 2, math.radians(45))
    second = obb2poly(0, 0, 20, 2, math.radians(-45))
    kept = rotated_nms([first, second], [0.9, 0.8], 0.5)
    assert kept == [0, 1]


def test_merge_tiles_shifts_to_image_coordinates_and_merges_the_overlap() -> None:
    # the same object seen in two overlapping tiles: tile B starts at x=824
    tile_a = [Detection(rect(800, 100, 900, 200), 0.95, 0)]
    tile_b = [Detection(rect(-24, 100, 76, 200), 0.90, 0)]  # 800-824 = -24
    merged = merge_tiles([(0, 0, tile_a), (824, 0, tile_b)], 0.5)
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.95)
    assert merged[0].polygon[0].tolist() == [800.0, 100.0]


def test_merge_tiles_keeps_different_classes_separate() -> None:
    merged = merge_tiles(
        [(0, 0, [Detection(rect(0, 0, 10, 10), 0.9, 0), Detection(rect(0, 0, 10, 10), 0.8, 1)])],
        0.5,
    )
    assert len(merged) == 2


def test_predicted_boxes_are_clipped_to_the_image() -> None:
    from rs_backbone_eval.detection.nms import clip_polygon

    # a 100x100 box whose right half hangs outside a 64 px image
    outside = rect(14, 14, 114, 114)
    clipped = clip_polygon(outside, 64, 64)
    assert clipped is not None
    assert clipped[:, 0].max() <= 64 and clipped[:, 1].max() <= 64
    assert clipped[:, 0].min() >= 0 and clipped[:, 1].min() >= 0
    # fully inside keeps its corners; fully outside disappears
    assert polygon_iou(clip_polygon(rect(0, 0, 10, 10), 64, 64), rect(0, 0, 10, 10)) == 1.0
    assert clip_polygon(rect(100, 100, 110, 110), 64, 64) is None
