"""Reference-aligned AP for G1 calibration only — never the formal metric.

MMRotate's ``DOTAMetric`` matches with
``mmrotate/evaluation/functional/mean_ap.py:tpfp_default``: for every detection
it takes the **highest-IoU ground truth over all of them** and, if that trip is
already covered by a higher-scoring detection, records a false positive without
falling back to the second-best GT. Our formal evaluator
(:func:`rs_backbone_eval.detection.ap.evaluate`) instead matches each detection
to the best *still-unmatched* target and interpolates over 101 COCO points. The
two protocols are different metrics, not two readings of one number: on the
reviewer's minimal case (2026-11-02) the reference gives TP ``[1, 0]`` and
AP ``0.545``, ours gives AP ``1.0``.

So G1 compares against the published 0.5454 with *this* module on the saved
predictions, while the formal COCO number stands on its own. The DOTA reference
config converts quadrilateral ground truth to ``rbox`` before scoring, so every
polygon is converted the same way here and IoU is computed on the resulting
rotated rectangles.
"""

from __future__ import annotations

import numpy as np

from .ap import Predictions, Targets, _iou_matrix
from .geometry import poly2obb

# MMDetection's '11points' average precision (VOC-07).
RECALL_LEVELS = np.linspace(0.0, 1.0, 11)


def rbox_corners(polygon: np.ndarray) -> np.ndarray:
    """Corners of the rotated box a quad converts to (the reference eval unit)."""
    cx, cy, width, height, theta = poly2obb(polygon)
    cos, sin = np.cos(theta), np.sin(theta)
    half = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    rotation = np.array([[cos, -sin], [sin, cos]])
    return (rotation @ half.T).T + np.array([cx, cy])


def _voc07_ap(tp: np.ndarray, positives: int) -> float | None:
    if positives <= 0:
        return None
    if tp.size == 0:
        return 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(~tp)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    sampled = [
        precision[recall >= level].max() if np.any(recall >= level) else 0.0
        for level in RECALL_LEVELS
    ]
    return float(np.mean(sampled))


def _reference_tp(
    detections: list[tuple[float, np.ndarray]],
    targets: list[tuple[np.ndarray, bool]],
    iou_threshold: float,
) -> np.ndarray:
    """tpfp_default semantics: argmax over *all* GT, no second-best fallback."""
    tp = np.zeros(len(detections), dtype=bool)
    if not targets:
        return tp
    # the shared AABB-prefiltered matrix: same values, no polygon clipping for
    # pairs whose hulls cannot touch
    ious = _iou_matrix(detections, targets)
    best = ious.max(axis=1) if ious.size else np.zeros(len(detections))
    index = ious.argmax(axis=1) if ious.size else np.zeros(len(detections), dtype=np.int64)
    covered = np.zeros(len(targets), dtype=bool)
    order = np.argsort(-np.asarray([score for score, _ in detections]), kind="stable")
    for position in order:
        if best[position] < iou_threshold:
            continue
        matched = int(index[position])
        if targets[matched][1]:
            # difficult/ignored: neither TP nor FP, exactly as the reference
            continue
        if not covered[matched]:
            covered[matched] = True
            tp[position] = True
    return tp


def evaluate_reference(
    predictions: dict[str, Predictions],
    targets: dict[str, Targets],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """VOC-07 11-point AP with MMRotate's matching on rbox-converted boxes."""
    if set(predictions) - set(targets):
        missing = sorted(set(predictions) - set(targets))
        raise ValueError(f"predictions without targets: {missing[:5]}")

    per_class: dict[int, float] = {}
    for class_id in range(num_classes):
        positives = 0
        chunks: list[np.ndarray] = []
        for image_id, target in targets.items():
            ground = [
                (rbox_corners(polygon), ignored)
                for polygon, cid, ignored in zip(
                    target.polygons, target.class_ids, target.ignored, strict=True
                )
                if cid == class_id
            ]
            positives += sum(1 for _, ignored in ground if not ignored)
            scene = predictions.get(image_id)
            if scene is None:
                continue
            detections = sorted(
                (
                    (float(score), rbox_corners(polygon))
                    for polygon, cid, score in zip(
                        scene.polygons, scene.class_ids, scene.scores, strict=True
                    )
                    if cid == class_id
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            chunks.append(_reference_tp(detections, ground, iou_threshold))
        if positives == 0:
            continue
        tp = np.concatenate(chunks) if chunks else np.zeros(0, dtype=bool)
        value = _voc07_ap(tp, positives)
        per_class[class_id] = 0.0 if value is None else value
    return {
        "map": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "per_class_ap": {class_id: float(ap) for class_id, ap in sorted(per_class.items())},
        "classes_with_targets": len(per_class),
        "iou_threshold": iou_threshold,
        "interpolation": "11points",
        "matching": "reference tpfp_default (argmax over all GT, no fallback)",
    }
