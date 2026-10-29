"""Rotated-box geometry shared by the detection builder and the evaluators.

Pure numpy, no shapely and no torch: the manifest builder runs in the control
plane, and the evaluator runs on CPU. Every function here is exact for convex
quadrilaterals (FAIR1M annotations are quadrilaterals; clipping a convex polygon
by an axis-aligned box or by another convex polygon keeps it convex).

Angle convention: **le90** — ``theta`` in ``(-pi/2, pi/2]``, ``w`` is the longer
edge. This is the convention MMRotate's ``le90`` configs use, so a FAIR1M result
can be compared to published OBB numbers without re-deriving boxes (FAIR1M
review 2026-10-29, §5.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    """One annotated object: class name plus four ring-ordered vertices."""

    class_name: str
    points: tuple[tuple[float, float], ...]

    @property
    def polygon(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float64)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        values = self.polygon
        return (
            float(values[:, 0].min()),
            float(values[:, 1].min()),
            float(values[:, 0].max()),
            float(values[:, 1].max()),
        )

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a simple polygon (abs, so ring order does not matter)."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        return 0.0
    x, y = values[:, 0], values[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _clip_half_plane(points: np.ndarray, axis: int, bound: float, keep_greater: bool) -> np.ndarray:
    """Sutherland–Hodgman against one axis-aligned half-plane."""
    if points.size == 0:
        return points
    values = np.asarray(points, dtype=np.float64)
    inside = values[:, axis] >= bound if keep_greater else values[:, axis] <= bound
    if inside.all():
        return values
    if not inside.any():
        return np.empty((0, 2), dtype=np.float64)
    output: list[tuple[float, float]] = []
    count = len(values)
    for index in range(count):
        current = values[index]
        following = values[(index + 1) % count]
        current_in = bool(inside[index])
        following_in = bool(inside[(index + 1) % count])
        if current_in:
            output.append((float(current[0]), float(current[1])))
        if current_in != following_in:
            delta = following[axis] - current[axis]
            if delta == 0:
                continue
            ratio = (bound - current[axis]) / delta
            crossing = current + ratio * (following - current)
            output.append((float(crossing[0]), float(crossing[1])))
    return np.asarray(output, dtype=np.float64) if output else np.empty((0, 2), dtype=np.float64)


def clip_to_rect(points: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Intersect a convex polygon with an axis-aligned rectangle."""
    clipped = np.asarray(points, dtype=np.float64)
    clipped = _clip_half_plane(clipped, 0, x0, keep_greater=True)
    clipped = _clip_half_plane(clipped, 0, x1, keep_greater=False)
    clipped = _clip_half_plane(clipped, 1, y0, keep_greater=True)
    clipped = _clip_half_plane(clipped, 1, y1, keep_greater=False)
    return clipped


def visible_fraction(points: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    """Share of the object's area that falls inside the window.

    The detection protocol keeps objects with at least a declared share inside a
    tile and treats the rest as ignore, so a truncated object is never trained
    as background (FAIR1M plan §5.2).
    """
    total = polygon_area(points)
    if total <= 0:
        return 0.0
    return polygon_area(clip_to_rect(points, x0, y0, x1, y1)) / total


def _convex_half_planes(points: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Inward half-planes ``(normal, offset)`` of a convex, CCW polygon."""
    values = np.asarray(points, dtype=np.float64)
    signed = np.dot(values[:, 0], np.roll(values[:, 1], -1)) - np.dot(
        values[:, 1], np.roll(values[:, 0], -1)
    )
    if signed < 0:  # make it counter-clockwise
        values = values[::-1]
    planes: list[tuple[np.ndarray, float]] = []
    for index in range(len(values)):
        start = values[index]
        end = values[(index + 1) % len(values)]
        edge = end - start
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        length = float(np.hypot(*normal))
        if length == 0:
            continue
        normal /= length
        planes.append((normal, float(np.dot(normal, start))))
    return planes


def polygon_intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    """Exact overlap area of two convex polygons."""
    clipped = np.asarray(first, dtype=np.float64)
    for normal, offset in _convex_half_planes(np.asarray(second, dtype=np.float64)):
        if clipped.size == 0:
            return 0.0
        signs = clipped @ normal - offset
        inside = signs >= 0
        if inside.all():
            continue
        if not inside.any():
            return 0.0
        output: list[tuple[float, float]] = []
        count = len(clipped)
        for index in range(count):
            current, following = clipped[index], clipped[(index + 1) % count]
            current_in, following_in = bool(inside[index]), bool(inside[(index + 1) % count])
            if current_in:
                output.append((float(current[0]), float(current[1])))
            if current_in != following_in:
                denominator = float(np.dot(normal, following - current))
                if denominator == 0:
                    continue
                ratio = -float(np.dot(normal, current) - offset) / denominator
                crossing = current + ratio * (following - current)
                output.append((float(crossing[0]), float(crossing[1])))
        clipped = (
            np.asarray(output, dtype=np.float64) if output else np.empty((0, 2), dtype=np.float64)
        )
    return polygon_area(clipped)


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """IoU of two convex polygons (used by the OBB matcher)."""
    intersection = polygon_intersection_area(first, second)
    if intersection <= 0:
        return 0.0
    union = polygon_area(first) + polygon_area(second) - intersection
    return float(intersection / union) if union > 0 else 0.0


def normalize_theta(theta: float) -> float:
    """Fold an angle into ``(-pi/2, pi/2]`` (the le90 domain)."""
    value = math.atan2(math.sin(theta), math.cos(theta))
    if value <= -math.pi / 2 + 1e-12:
        value += math.pi
    elif value > math.pi / 2 + 1e-12:
        value -= math.pi
    return value


def poly2obb(points: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(cx, cy, w, h, theta)`` in le90 convention from a quadrilateral."""
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (4, 2):
        raise ValueError(f"expected four vertices, got {values.shape}")
    centre = values.mean(axis=0)
    edges = np.roll(values, -1, axis=0) - values
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    # average the two opposite sides: a real annotation is never a perfect
    # rectangle, and using one edge would make the box depend on vertex order
    width = float((lengths[0] + lengths[2]) / 2.0)
    height = float((lengths[1] + lengths[3]) / 2.0)
    longest = 0 if lengths[0] + lengths[2] >= lengths[1] + lengths[3] else 1
    edge = edges[longest]
    theta = normalize_theta(math.atan2(float(edge[1]), float(edge[0])))
    if width < height:
        width, height = height, width
        theta = normalize_theta(theta + math.pi / 2)
    return float(centre[0]), float(centre[1]), width, height, theta


def obb2poly(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """Four vertices of an le90 rotated box (counter-clockwise)."""
    cos, sin = math.cos(theta), math.sin(theta)
    half_w, half_h = w / 2.0, h / 2.0
    corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])
    rotation = np.array([[cos, -sin], [sin, cos]])
    return corners @ rotation.T + np.array([cx, cy])
