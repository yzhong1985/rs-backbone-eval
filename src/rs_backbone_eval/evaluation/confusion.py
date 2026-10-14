"""Confusion matrix accumulator shared by segmentation and classification."""

from __future__ import annotations

import math

import numpy as np


def _as_flat_integers(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    array = array.ravel()
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got dtype {array.dtype}")
    return array


def _safe_div(numerator: float, denominator: float, *, zero: float) -> float:
    return float(numerator / denominator) if denominator else zero


class ConfusionMatrix:
    """Accumulate integer label/prediction pairs into one global matrix.

    Conventions: precision, recall, and F1 use the zero-division-is-zero rule
    (a never-predicted class scores 0, matching common evaluator defaults so
    macro averages stay defined); IoU is NaN only when a class is absent from
    both ground truth and predictions, and macro averages skip those classes.
    """

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self._counts = np.zeros((num_classes, num_classes), dtype=np.int64)

    @classmethod
    def from_counts(cls, counts: np.ndarray) -> ConfusionMatrix:
        """Rebuild an accumulator from a stored ``[K, K]`` count matrix."""
        array = np.asarray(counts, dtype=np.int64)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"counts must be square [K, K], got {array.shape}")
        if array.min() < 0:
            raise ValueError("counts must be non-negative")
        matrix = cls(int(array.shape[0]))
        matrix._counts = array.copy()
        return matrix

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        ignore_index: int | None = None,
    ) -> None:
        """Add one batch; rows are ground truth, columns are predictions."""
        truth = _as_flat_integers(y_true, "y_true")
        pred = _as_flat_integers(y_pred, "y_pred")
        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch: y_true {truth.shape} != y_pred {pred.shape}")
        valid = np.ones(truth.shape, dtype=bool)
        if valid_mask is not None:
            mask = np.asarray(valid_mask).ravel().astype(bool)
            if mask.shape != truth.shape:
                raise ValueError(f"valid_mask shape {mask.shape} != labels {truth.shape}")
            valid &= mask
        if ignore_index is not None:
            valid &= truth != ignore_index
        truth, pred = truth[valid], pred[valid]
        if truth.size == 0:
            return
        if truth.min() < 0 or truth.max() >= self.num_classes:
            raise ValueError(f"y_true out of range [0, {self.num_classes})")
        if pred.min() < 0 or pred.max() >= self.num_classes:
            raise ValueError(f"y_pred out of range [0, {self.num_classes})")
        self._counts += np.bincount(
            self.num_classes * truth + pred, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    @property
    def counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def total(self) -> int:
        return int(self._counts.sum())

    def support(self, class_index: int) -> int:
        """Number of ground-truth pixels for one class."""
        return int(self._counts[class_index].sum())

    def true_positives(self, class_index: int) -> int:
        return int(self._counts[class_index, class_index])

    def precision(self, class_index: int) -> float:
        column = self._counts[:, class_index].sum()
        return _safe_div(self.true_positives(class_index), float(column), zero=0.0)

    def recall(self, class_index: int) -> float:
        denominator = float(self.support(class_index))
        return _safe_div(self.true_positives(class_index), denominator, zero=0.0)

    def f1(self, class_index: int) -> float:
        precision, recall = self.precision(class_index), self.recall(class_index)
        return _safe_div(2 * precision * recall, precision + recall, zero=0.0)

    def iou(self, class_index: int) -> float:
        true, pred = self._counts[class_index, :], self._counts[:, class_index]
        denominator = true.sum() + pred.sum() - self.true_positives(class_index)
        return _safe_div(self.true_positives(class_index), float(denominator), zero=math.nan)

    def present_classes(self, *, ground_truth_only: bool = True) -> list[int]:
        """Classes seen in ground truth (or in either axis) for macro averages."""
        seen: list[int] = []
        for index in range(self.num_classes):
            gt_seen = self.support(index) > 0
            pred_seen = self._counts[:, index].sum() > 0
            if gt_seen or (not ground_truth_only and pred_seen):
                seen.append(index)
        return seen


class GroupConfusion:
    """One ``ConfusionMatrix`` per group (region/tile/event).

    The report resamples groups and re-accumulates a global matrix to get an IoU
    interval, so evaluation must keep the per-group matrices instead of only the
    grand total. ``to_json`` matches ``RunWriter.add_confusion``.
    """

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self._groups: dict[str, ConfusionMatrix] = {}

    def update(
        self,
        group_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        ignore_index: int | None = None,
    ) -> None:
        matrix = self._groups.get(group_id)
        if matrix is None:
            matrix = ConfusionMatrix(self.num_classes)
            self._groups[group_id] = matrix
        matrix.update(y_true, y_pred, valid_mask=valid_mask, ignore_index=ignore_index)

    @property
    def groups(self) -> list[str]:
        return sorted(self._groups)

    def combined(self) -> ConfusionMatrix:
        total = ConfusionMatrix(self.num_classes)
        for matrix in self._groups.values():
            total._counts += matrix._counts  # noqa: SLF001 - same-package accumulator
        return total

    def to_json(self) -> dict[str, list[list[int]]]:
        return {group: self._groups[group].counts.tolist() for group in self.groups}
