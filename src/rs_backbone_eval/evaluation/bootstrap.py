"""Deterministic bootstrap confidence intervals for evaluation statistics.

``bootstrap_ci`` resamples per-sample values and is meant for statistics that
are themselves a mean of per-sample quantities. Global IoU is *not* such a
statistic: it is computed from one accumulated confusion matrix, so its interval
must be taken by resampling groups (regions, tiles, events), re-accumulating the
confusion matrix for each resample, and recomputing the metric. That is what
``bootstrap_confusion`` and ``bootstrap_confusion_paired`` do, and the paired
form uses the same resampled groups for both models so the interval is on the
model difference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .confusion import ConfusionMatrix


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    sample_count: int


def bootstrap_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 17,
) -> BootstrapInterval:
    """Percentile bootstrap of the mean; identical inputs and seed give identical bounds."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("values must contain at least one finite entry")
    point = float(array.mean())
    if array.size == 1 or n_resamples <= 0:
        return BootstrapInterval(point, point, point, confidence, max(n_resamples, 0), array.size)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return BootstrapInterval(
        point=point,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        sample_count=array.size,
    )


def _as_group_counts(groups: Sequence[np.ndarray]) -> np.ndarray:
    array = np.asarray(groups, dtype=np.int64)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f"groups must be a sequence of [K, K] matrices, got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("at least one group is required")
    return array


def _percentile(values: np.ndarray, point: float, confidence: float, n_resamples: int, groups: int):
    alpha = (1 - confidence) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return BootstrapInterval(
        point=float(point),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        sample_count=groups,
    )


def bootstrap_confusion(
    groups: Sequence[np.ndarray],
    metric: Callable[[ConfusionMatrix], float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 17,
) -> BootstrapInterval:
    """Grouped bootstrap of a metric computed from a global confusion matrix.

    ``groups`` is one ``[K, K]`` matrix per group (region/tile/event); each
    resample draws groups with replacement, re-accumulates the matrix, and
    recomputes ``metric`` (for example ``lambda cm: cm.iou(1)``).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    array = _as_group_counts(groups)

    def statistic(counts: np.ndarray) -> float:
        return float(metric(ConfusionMatrix.from_counts(counts)))

    point = statistic(array.sum(axis=0))
    if array.shape[0] == 1 or n_resamples <= 0:
        return BootstrapInterval(
            point, point, point, confidence, max(n_resamples, 0), array.shape[0]
        )

    rng = np.random.default_rng(seed)
    index = rng.integers(0, array.shape[0], size=(n_resamples, array.shape[0]))
    summed = array[index].sum(axis=1)  # [n_resamples, K, K]
    values = np.array([statistic(counts) for counts in summed], dtype=np.float64)
    return _percentile(values, point, confidence, n_resamples, array.shape[0])


def bootstrap_confusion_paired(
    groups_a: Sequence[np.ndarray],
    groups_b: Sequence[np.ndarray],
    metric: Callable[[ConfusionMatrix], float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 17,
) -> BootstrapInterval:
    """Paired grouped bootstrap of ``metric(A) - metric(B)`` on shared groups.

    Both sequences must be aligned by group (same order, same groups); each
    resample draws group indices once and applies them to both models, so the
    interval is on the paired difference and cancels group difficulty.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    a = _as_group_counts(groups_a)
    b = _as_group_counts(groups_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"group counts differ: {a.shape[0]} vs {b.shape[0]}")

    def statistic(counts: np.ndarray) -> float:
        return float(metric(ConfusionMatrix.from_counts(counts)))

    point = statistic(a.sum(axis=0)) - statistic(b.sum(axis=0))
    if a.shape[0] == 1 or n_resamples <= 0:
        return BootstrapInterval(point, point, point, confidence, max(n_resamples, 0), a.shape[0])

    rng = np.random.default_rng(seed)
    index = rng.integers(0, a.shape[0], size=(n_resamples, a.shape[0]))
    summed_a, summed_b = a[index].sum(axis=1), b[index].sum(axis=1)
    values = np.array(
        [statistic(ca) - statistic(cb) for ca, cb in zip(summed_a, summed_b, strict=True)],
        dtype=np.float64,
    )
    return _percentile(values, point, confidence, n_resamples, a.shape[0])
