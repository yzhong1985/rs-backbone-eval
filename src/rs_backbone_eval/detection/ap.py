"""COCO-style AP for the FAIR1M detection track (101-point interpolation).

The protocol publishes COCO AP (AP, AP50, AP75) so a FAIR1M cell can be compared
with published OBB numbers without re-deriving the metric, plus the two things a
backbone comparison actually needs: per-class AP and size-binned recall
(FAIR1M plan §9, decisions 5: COCO 101, comparable).

Deliberately not a reimplementation of pycocotools: matching is greedy by score
against exact polygon IoU, ignored ground truth neither counts as a hit nor as a
false positive, and a class with no ground truth is excluded from the mean rather
than scored zero. That is the same contract, without a compiled dependency in the
evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import polygon_area, polygon_iou

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
# COCO area bins in pixels squared. At the declared 0.5 m GSD a 32 px box is
# ~16 m across and a 96 px box ~48 m, so these separate cars from ships and
# airfields; the runs record gsd_source=declared next to them.
SMALL_AREA = 32**2
LARGE_AREA = 96**2
RECALL_LEVELS = 101


@dataclass
class Predictions:
    polygons: list[np.ndarray] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.scores)

    def append(self, polygon: np.ndarray, class_id: int, score: float) -> None:
        self.polygons.append(np.asarray(polygon, dtype=np.float64))
        self.class_ids.append(int(class_id))
        self.scores.append(float(score))


@dataclass
class Targets:
    polygons: list[np.ndarray] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    # Objects cut by the tile edge: matched detections are neither TP nor FP.
    ignored: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.class_ids)

    def append(self, polygon: np.ndarray, class_id: int, *, ignored: bool = False) -> None:
        self.polygons.append(np.asarray(polygon, dtype=np.float64))
        self.class_ids.append(int(class_id))
        self.ignored.append(bool(ignored))


def _match_class(
    detections: list[tuple[float, np.ndarray]],
    targets: list[tuple[np.ndarray, bool]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy score-ordered matching for one class and one IoU threshold.

    Returns ``(tp, fp, ignored)`` boolean arrays aligned with ``detections``.
    A detection that overlaps an ignored target above the threshold is dropped
    from the counts instead of being called a false positive — that is what lets
    the tile protocol discard truncated objects without punishing the model.
    """
    taken = [False] * len(targets)
    tp = np.zeros(len(detections), dtype=bool)
    fp = np.zeros(len(detections), dtype=bool)
    ignored = np.zeros(len(detections), dtype=bool)
    order = sorted(range(len(detections)), key=lambda index: detections[index][0], reverse=True)
    for index in order:
        _, polygon = detections[index]
        best_index, best_iou, best_ignored = -1, iou_threshold, False
        for target_index, (target, is_ignored) in enumerate(targets):
            if taken[target_index]:
                continue
            overlap = polygon_iou(polygon, target)
            if overlap >= best_iou:
                best_index, best_iou, best_ignored = target_index, overlap, is_ignored
        if best_index < 0:
            fp[index] = True
            continue
        taken[best_index] = True
        if best_ignored:
            ignored[index] = True
        else:
            tp[index] = True
    return tp, fp, ignored


def average_precision(tp: np.ndarray, fp: np.ndarray, positives: int) -> float | None:
    """101-point interpolated AP; ``None`` when the class has no ground truth."""
    if positives <= 0:
        return None
    if tp.size == 0:
        return 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    levels = np.linspace(0.0, 1.0, RECALL_LEVELS)
    sampled = np.array(
        [envelope[recall >= level].max() if np.any(recall >= level) else 0.0 for level in levels]
    )
    return float(sampled.mean())


def evaluate(
    predictions: dict[str, Predictions],
    targets: dict[str, Targets],
    num_classes: int,
    *,
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    score_threshold: float = 0.0,
    small_area: float = SMALL_AREA,
    large_area: float = LARGE_AREA,
) -> dict[str, object]:
    """AP / AP50 / AP75, per-class AP50 and size-binned recall at IoU 0.5.

    ``predictions`` and ``targets`` are keyed by image id (the scene, not the
    tile) so a merged detection list is scored once per scene.
    """
    if set(predictions) - set(targets):
        missing = sorted(set(predictions) - set(targets))
        raise ValueError(f"predictions without targets: {missing[:5]}")

    per_class_ap: dict[int, list[float]] = {}
    recall_bins = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for class_id in range(num_classes):
        thresholds_ap: list[float] = []
        for threshold in iou_thresholds:
            tp_all: list[np.ndarray] = []
            fp_all: list[np.ndarray] = []
            scores_all: list[float] = []
            positives = 0
            for image_id, target in targets.items():
                detection_items = [
                    (score, polygon)
                    for polygon, cid, score in zip(
                        predictions.get(image_id, Predictions()).polygons,
                        predictions.get(image_id, Predictions()).class_ids,
                        predictions.get(image_id, Predictions()).scores,
                        strict=True,
                    )
                    if cid == class_id and score >= score_threshold
                ]
                target_items = [
                    (polygon, ignored)
                    for polygon, cid, ignored in zip(
                        target.polygons, target.class_ids, target.ignored, strict=True
                    )
                    if cid == class_id
                ]
                positives += sum(1 for _, ignored in target_items if not ignored)
                if not detection_items:
                    continue
                tp, fp, _ = _match_class(detection_items, target_items, threshold)
                tp_all.append(tp)
                fp_all.append(fp)
                scores_all.extend(score for score, _ in detection_items)
            if positives == 0:
                break
            order = np.argsort(-np.asarray(scores_all)) if scores_all else np.array([], dtype=int)
            tp = np.concatenate(tp_all)[order] if tp_all else np.zeros(0, dtype=bool)
            fp = np.concatenate(fp_all)[order] if fp_all else np.zeros(0, dtype=bool)
            value = average_precision(tp, fp, positives)
            thresholds_ap.append(0.0 if value is None else value)

            if abs(threshold - 0.5) < 1e-9:
                _accumulate_recall(
                    targets,
                    predictions,
                    class_id,
                    threshold,
                    score_threshold,
                    small_area,
                    large_area,
                    recall_bins,
                )
        if thresholds_ap:
            per_class_ap[class_id] = thresholds_ap

    values = np.asarray(list(per_class_ap.values())) if per_class_ap else np.zeros((0, 0))
    index_50 = list(iou_thresholds).index(0.5) if 0.5 in iou_thresholds else 0
    index_75 = list(iou_thresholds).index(0.75) if 0.75 in iou_thresholds else 0
    return {
        "map": float(values.mean()) if values.size else 0.0,
        "map_50": float(values[:, index_50].mean()) if values.size else 0.0,
        "map_75": float(values[:, index_75].mean()) if values.size else 0.0,
        "classes_with_targets": len(per_class_ap),
        "per_class_ap50": {
            class_id: float(aps[index_50]) for class_id, aps in sorted(per_class_ap.items())
        },
        "recall_small": _ratio(recall_bins["small"]),
        "recall_medium": _ratio(recall_bins["medium"]),
        "recall_large": _ratio(recall_bins["large"]),
    }


def _ratio(pair: list[int]) -> float:
    return float(pair[0] / pair[1]) if pair[1] else 0.0


def _accumulate_recall(
    targets: dict[str, Targets],
    predictions: dict[str, Predictions],
    class_id: int,
    threshold: float,
    score_threshold: float,
    small_area: float,
    large_area: float,
    bins: dict[str, list[int]],
) -> None:
    """Count matched/total ground truth per COCO area bin at one threshold."""
    for image_id, target in targets.items():
        detection_items = [
            (score, polygon)
            for polygon, cid, score in zip(
                predictions.get(image_id, Predictions()).polygons,
                predictions.get(image_id, Predictions()).class_ids,
                predictions.get(image_id, Predictions()).scores,
                strict=True,
            )
            if cid == class_id and score >= score_threshold
        ]
        target_items: list[tuple[np.ndarray, bool]] = []
        areas: list[float] = []
        for polygon, cid, ignored in zip(
            target.polygons, target.class_ids, target.ignored, strict=True
        ):
            if cid != class_id or ignored:
                continue
            target_items.append((polygon, False))
            areas.append(polygon_area(polygon))
        if not target_items:
            continue
        # recall per bin only needs which targets were hit, so the greedy matcher
        # is run once and its target indices are bucketed by area
        hits = _greedy_hits(detection_items, target_items, threshold)
        for index, area in enumerate(areas):
            name = "small" if area < small_area else ("large" if area > large_area else "medium")
            bins[name][1] += 1
            if index in hits:
                bins[name][0] += 1


def _greedy_hits(
    detections: list[tuple[float, np.ndarray]],
    targets: list[tuple[np.ndarray, bool]],
    iou_threshold: float,
) -> set[int]:
    """Indices of targets matched by the greedy score-ordered matcher."""
    taken: set[int] = set()
    for _, polygon in sorted(detections, key=lambda item: item[0], reverse=True):
        best_index, best_iou = -1, iou_threshold
        for index, (target, _ignored) in enumerate(targets):
            if index in taken:
                continue
            overlap = polygon_iou(polygon, target)
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0:
            taken.add(best_index)
    return taken
