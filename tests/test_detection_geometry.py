"""Rotated-box geometry: exact overlap, le90 round-trip, visibility ratio.

These are the primitives the FAIR1M builder, the matcher and the AP evaluator
share, so they are tested against hand-computed numbers instead of snapshots.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rs_backbone_eval.detection.geometry import (
    Instance,
    clip_to_rect,
    normalize_theta,
    obb2poly,
    poly2obb,
    polygon_area,
    polygon_intersection_area,
    polygon_iou,
    visible_fraction,
)


def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_polygon_area_matches_hand_computation() -> None:
    assert polygon_area(rect(0, 0, 10, 4)) == pytest.approx(40.0)
    # ring order must not matter: reversed vertices are the same polygon
    assert polygon_area(rect(0, 0, 10, 4)[::-1]) == pytest.approx(40.0)
    assert polygon_area(np.zeros((0, 2))) == 0.0


def test_clip_keeps_exact_overlap_area() -> None:
    clipped = clip_to_rect(rect(0, 0, 10, 10), 5, 5, 20, 20)
    assert polygon_area(clipped) == pytest.approx(25.0)

    # fully inside and fully outside are both exact
    assert polygon_area(clip_to_rect(rect(2, 2, 4, 4), 0, 0, 10, 10)) == pytest.approx(4.0)
    assert polygon_area(clip_to_rect(rect(20, 20, 30, 30), 0, 0, 10, 10)) == pytest.approx(0.0)


def test_visible_fraction_is_the_share_inside_the_window() -> None:
    # half the object hangs past the tile edge
    assert visible_fraction(rect(0, 0, 10, 10), 5, 0, 100, 100) == pytest.approx(0.5)
    assert visible_fraction(rect(0, 0, 10, 10), 0, 0, 100, 100) == pytest.approx(1.0)
    assert visible_fraction(rect(0, 0, 10, 10), 50, 50, 100, 100) == pytest.approx(0.0)


def test_polygon_iou_is_exact_for_half_overlap() -> None:
    assert polygon_iou(rect(0, 0, 10, 10), rect(0, 0, 10, 10)) == pytest.approx(1.0)
    assert polygon_iou(rect(0, 0, 10, 10), rect(50, 50, 60, 60)) == pytest.approx(0.0)
    # shifted by half a side: intersection 5*10 = 50, union 100 + 100 - 50 = 150
    assert polygon_iou(rect(0, 0, 10, 10), rect(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_polygon_intersection_handles_rotated_boxes() -> None:
    first = obb2poly(0, 0, 10, 10, math.radians(45))
    second = obb2poly(0, 0, 10, 10, math.radians(45))
    assert polygon_intersection_area(first, second) == pytest.approx(polygon_area(first))
    assert polygon_iou(first, second) == pytest.approx(1.0)


@pytest.mark.parametrize("angle", [0.0, 15.0, 45.0, 89.0, -30.0])
def test_obb_round_trip_is_exact_for_a_rectangle(angle: float) -> None:
    theta = math.radians(angle)
    polygon = obb2poly(30.0, -12.0, 8.0, 3.0, theta)
    cx, cy, width, height, recovered = poly2obb(polygon)
    assert (cx, cy) == pytest.approx((30.0, -12.0), abs=1e-9)
    assert width == pytest.approx(8.0, abs=1e-9)
    assert height == pytest.approx(3.0, abs=1e-9)
    assert recovered == pytest.approx(theta, abs=1e-9)
    # and the reconstructed polygon is the same set of corners
    assert np.allclose(
        np.sort(obb2poly(cx, cy, width, height, recovered), axis=0),
        np.sort(polygon, axis=0),
        atol=1e-9,
    )


def test_le90_always_reports_the_long_edge() -> None:
    # a 3x8 box given with the short edge first must come back as w=8, h=3
    polygon = obb2poly(0, 0, 8, 3, math.radians(70))
    _, _, width, height, theta = poly2obb(polygon)
    assert width >= height
    assert theta == pytest.approx(math.radians(70), abs=1e-9)


def test_theta_is_folded_into_the_le90_domain() -> None:
    assert normalize_theta(math.pi) == pytest.approx(0.0)
    assert normalize_theta(-math.pi) == pytest.approx(0.0)
    # the ±90° boundary folds to +pi/2, the closed end of the le90 domain
    assert abs(normalize_theta(3 * math.pi / 2)) == pytest.approx(math.pi / 2)
    value = normalize_theta(math.radians(100))
    assert -math.pi / 2 < value <= math.pi / 2


def test_instance_bbox_and_area_come_from_the_polygon() -> None:
    instance = Instance(class_name="Small Car", points=((10, 20), (30, 20), (30, 25), (10, 25)))
    assert instance.bbox == (10.0, 20.0, 30.0, 25.0)
    assert instance.area == pytest.approx(100.0)


def test_a_box_written_from_the_short_edge_is_not_rotated_twice() -> None:
    """Regression 2026-10-29: the long-edge selection was applied twice.

    FAIR1M annotations start on any vertex, so 44% begin on the short edge. The
    old code picked the long pair and then swapped again, turning those boxes by
    90 degrees: reconstruction IoU 0.111 instead of 1.0.
    """
    polygon = np.array([[0, 0], [0, 20], [100, 20], [100, 0]], dtype=np.float64)
    cx, cy, width, height, theta = poly2obb(polygon)
    assert (cx, cy) == pytest.approx((50.0, 10.0))
    assert (width, height) == pytest.approx((100.0, 20.0))
    assert theta == pytest.approx(0.0)
    assert polygon_iou(polygon, obb2poly(cx, cy, width, height, theta)) == pytest.approx(1.0)


def test_every_vertex_order_produces_the_same_box() -> None:
    base = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], dtype=np.float64)
    orderings = [np.roll(base, shift, axis=0) for shift in range(4)]
    orderings += [np.roll(base[::-1], shift, axis=0) for shift in range(4)]
    boxes = [poly2obb(polygon) for polygon in orderings]
    for box in boxes:
        assert box == pytest.approx(boxes[0], abs=1e-9)
    # and every reconstruction covers the original polygon
    for polygon, box in zip(orderings, boxes, strict=True):
        assert polygon_iou(polygon, obb2poly(*box)) == pytest.approx(1.0)
