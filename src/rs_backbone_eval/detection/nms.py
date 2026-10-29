"""Rotated NMS and whole-image merging for the detection track.

A detection run evaluates whole scenes, not tiles: every tile produces boxes in
its own coordinates, they are shifted into image coordinates and suppressed
across tile borders. Two tiles overlap by 200 px, so the same object is normally
seen more than once and must be merged exactly once — the segmentation track gets
this from averaging logits, detection needs NMS (FAIR1M plan §7.1).

Only convex quadrilaterals are handled; FAIR1M annotations are quadrilaterals and
the detector regresses four corners, so nothing here needs a general polygon
engine. Pure numpy, CPU: it never runs inside autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import polygon_iou


@dataclass(frozen=True)
class Detection:
    """One predicted box in image coordinates."""

    polygon: np.ndarray  # [4, 2] float64
    score: float
    class_id: int


def rotated_nms(
    polygons: list[np.ndarray], scores: list[float], iou_threshold: float = 0.5
) -> list[int]:
    """Greedy NMS over convex polygons; returns kept indices, score-ordered.

    Suppression uses exact polygon IoU, not the IoU of the axis-aligned hull:
    two long diagonal ships overlap heavily as polygons and barely as hulls, and
    hull IoU would keep both.
    """
    if len(polygons) != len(scores):
        raise ValueError("polygons and scores must have the same length")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    kept: list[int] = []
    suppressed = [False] * len(polygons)
    for position, index in enumerate(order):
        if suppressed[index]:
            continue
        kept.append(index)
        for other in order[position + 1 :]:
            if suppressed[other]:
                continue
            if polygon_iou(polygons[index], polygons[other]) > iou_threshold:
                suppressed[other] = True
    return kept


def shift_polygon(polygon: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move a tile-local polygon into image coordinates."""
    return np.asarray(polygon, dtype=np.float64) + np.array([dx, dy])


def merge_tiles(
    tile_detections: list[tuple[int, int, list[Detection]]],
    iou_threshold: float = 0.5,
    *,
    max_detections: int | None = None,
) -> list[Detection]:
    """Shift every tile's detections to image coordinates, then NMS per class.

    ``tile_detections`` is ``[(x, y, detections), ...]`` with tile origins in
    image pixels. Per-class NMS, because a car next to a van is not a duplicate.
    ``max_detections`` caps the result after merging (applied per class, then
    globally by score) so one pathological tile cannot flood the evaluation.
    """
    merged: dict[int, list[Detection]] = {}
    for origin_x, origin_y, detections in tile_detections:
        for detection in detections:
            merged.setdefault(detection.class_id, []).append(
                Detection(
                    polygon=shift_polygon(detection.polygon, origin_x, origin_y),
                    score=detection.score,
                    class_id=detection.class_id,
                )
            )

    output: list[Detection] = []
    for class_id in sorted(merged):
        group = merged[class_id]
        keep = rotated_nms(
            [item.polygon for item in group],
            [item.score for item in group],
            iou_threshold,
        )
        output.extend(group[index] for index in keep)
    output.sort(key=lambda item: item.score, reverse=True)
    if max_detections is not None:
        output = output[:max_detections]
    return output
