import math

import numpy as np
import pytest

from rs_backbone_eval.evaluation.bootstrap import (
    bootstrap_ci,
    bootstrap_confusion,
    bootstrap_confusion_paired,
)
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix


def test_bootstrap_ci_is_deterministic_for_seed() -> None:
    values = np.arange(1, 101, dtype=float)

    first = bootstrap_ci(values, seed=17)
    second = bootstrap_ci(values, seed=17)
    other = bootstrap_ci(values, seed=42)

    assert first == second
    assert first.point == pytest.approx(50.5)
    assert first != other
    assert first.low <= first.point <= first.high


def test_bootstrap_ci_brackets_the_mean_for_spread_values() -> None:
    values = np.array([0.0, 1.0] * 25)

    interval = bootstrap_ci(values, confidence=0.95, n_resamples=2000, seed=17)

    assert interval.sample_count == 50
    assert interval.n_resamples == 2000
    assert interval.point == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.5, abs=0.15)
    assert interval.high == pytest.approx(0.5, abs=0.15)


def test_bootstrap_ci_degenerate_cases() -> None:
    single = bootstrap_ci(np.array([3.0]))
    assert single.point == single.low == single.high == 3.0

    dropped = bootstrap_ci(np.array([1.0, math.nan, 2.0]))
    assert dropped.sample_count == 2


def test_bootstrap_ci_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.array([1.0]), confidence=1.5)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci(np.array([math.nan]))


def test_confusion_from_counts_roundtrip() -> None:
    counts = np.array([[9, 1], [1, 9]], dtype=np.int64)
    matrix = ConfusionMatrix.from_counts(counts)
    assert np.array_equal(matrix.counts, counts)
    assert matrix.iou(1) == pytest.approx(9 / 11)


def test_grouped_bootstrap_reaccumulates_the_confusion_matrix() -> None:
    groups = [
        np.array([[9, 1], [1, 9]], dtype=np.int64),
        np.array([[8, 0], [0, 8]], dtype=np.int64),
    ]
    metric = lambda cm: cm.iou(1)  # noqa: E731
    interval = bootstrap_confusion(groups, metric, n_resamples=500, seed=17)

    # point is the global IoU of the summed matrix, not the mean of per-group IoU
    assert interval.point == pytest.approx(17 / 19)
    assert interval.sample_count == 2
    assert interval.low <= interval.point <= interval.high
    assert bootstrap_confusion(groups, metric, n_resamples=500, seed=17) == interval


def test_paired_bootstrap_is_zero_for_identical_models() -> None:
    groups = [
        np.array([[9, 1], [1, 9]], dtype=np.int64),
        np.array([[8, 0], [0, 8]], dtype=np.int64),
        np.array([[5, 2], [1, 6]], dtype=np.int64),
    ]
    metric = lambda cm: cm.iou(1)  # noqa: E731
    interval = bootstrap_confusion_paired(groups, groups, metric, n_resamples=500, seed=17)
    assert interval.point == 0.0
    assert interval.low == 0.0 and interval.high == 0.0


def test_paired_bootstrap_rejects_mismatched_groups() -> None:
    metric = lambda cm: cm.iou(1)  # noqa: E731
    with pytest.raises(ValueError, match="group counts differ"):
        bootstrap_confusion_paired(
            [np.eye(2, dtype=np.int64)],
            [np.eye(2, dtype=np.int64), np.eye(2, dtype=np.int64)],
            metric,
        )
