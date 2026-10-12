import math

import numpy as np
import pytest

from rs_backbone_eval.contracts import MetricRecord
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.classification import classification_metrics
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.regression import RegressionStats
from rs_backbone_eval.evaluation.segmentation import (
    binary_metrics,
    multiclass_metrics,
    per_sample_foreground_iou,
)


def test_confusion_matrix_accumulates_with_ignore_index() -> None:
    cm = ConfusionMatrix(num_classes=2)
    cm.update([0, 0, 1], [0, 1, 1], ignore_index=-1)
    cm.update([-1, 1], [1, 1], ignore_index=-1)

    assert cm.total == 4
    assert cm.counts.tolist() == [[1, 1], [0, 2]]


def test_confusion_matrix_rejects_out_of_range_and_floats() -> None:
    cm = ConfusionMatrix(num_classes=2)
    with pytest.raises(ValueError, match="out of range"):
        cm.update([0, 2], [0, 0])
    with pytest.raises(TypeError, match="integers"):
        cm.update([0.5], [0])


def test_binary_metrics_match_hand_computation() -> None:
    cm = ConfusionMatrix(num_classes=2)
    cm.update([0, 0, 0, 1, 1], [0, 1, 0, 1, 0])

    metrics = binary_metrics(cm)

    assert metrics["foreground_iou"] == pytest.approx(1 / 3)
    assert metrics["precision"] == pytest.approx(1 / 2)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["foreground_f1"] == pytest.approx(1 / 2)
    assert metrics["mean_iou"] == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_binary_metrics_undefined_without_foreground() -> None:
    cm = ConfusionMatrix(num_classes=2)
    cm.update([0, 0], [0, 0])

    metrics = binary_metrics(cm)

    assert math.isnan(metrics["foreground_iou"])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["mean_iou"] == pytest.approx(1.0)


def test_multiclass_metrics_match_hand_computation() -> None:
    cm = ConfusionMatrix(num_classes=3)
    cm.update([0, 0, 1, 1, 2], [0, 1, 1, 1, 0])

    metrics = multiclass_metrics(cm)

    assert cm.support(0) == 2 and cm.support(1) == 2 and cm.support(2) == 1
    assert metrics["iou/class_0"] == pytest.approx(1 / 3)
    assert metrics["iou/class_1"] == pytest.approx(2 / 3)
    assert metrics["iou/class_2"] == pytest.approx(0 / 3)
    assert metrics["mean_iou"] == pytest.approx(np.mean([1 / 3, 2 / 3, 0 / 3]))
    assert metrics["macro_f1"] == pytest.approx(np.mean([0.5, 0.8, 0.0]))
    expected_weighted = (2 * (1 / 3) + 2 * (2 / 3) + 1 * 0) / 5
    assert metrics["frequency_weighted_iou"] == pytest.approx(expected_weighted)


def test_classification_metrics_match_hand_computation() -> None:
    cm = ConfusionMatrix(num_classes=3)
    cm.update([0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 1, 2, 2, 0])

    metrics = classification_metrics(cm)

    assert metrics["accuracy"] == pytest.approx(5 / 7)
    assert metrics["balanced_accuracy"] == pytest.approx((1 / 2 + 1 + 2 / 3) / 3)
    assert metrics["macro_f1"] == pytest.approx((1 / 2 + 4 / 5 + 4 / 5) / 3)
    assert metrics["recall/class_2"] == pytest.approx(2 / 3)


def test_regression_metrics_match_hand_computation() -> None:
    stats = RegressionStats()
    stats.update([10, 20, 30], [12, 18, 36])

    metrics = stats.metrics()

    assert stats.count == 3
    assert metrics["rmse"] == pytest.approx(math.sqrt((4 + 4 + 36) / 3))
    assert metrics["mae"] == pytest.approx((2 + 2 + 6) / 3)
    assert metrics["bias"] == pytest.approx(2.0)
    ss_res = 44
    ss_tot = (10 - 20) ** 2 + 0**2 + (30 - 20) ** 2
    assert metrics["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_regression_respects_valid_mask_and_bins() -> None:
    stats = RegressionStats(bin_edges=(0, 50, 100, math.inf))
    stats.update([10, 60, 120, 90], [10, 60, 100, 110], valid_mask=[True, True, False, True])

    assert stats.count == 3
    bins = stats.binned_metrics()

    assert [b.label for b in bins] == ["0-50", "50-100", ">=100"]
    assert [b.count for b in bins] == [1, 2, 0]
    assert bins[0].rmse == pytest.approx(0.0)
    assert bins[1].rmse == pytest.approx(math.sqrt((0**2 + 20**2) / 2))
    assert bins[1].mae == pytest.approx(10.0)
    assert bins[1].bias == pytest.approx(10.0)
    assert math.isnan(bins[2].rmse)


def test_regression_rejects_nonfinite_values() -> None:
    stats = RegressionStats()
    with pytest.raises(ValueError, match="finite"):
        stats.update([1, 2], [1, float("nan")])


def test_per_sample_foreground_iou_ignores_labels() -> None:
    y_true = np.array([[[0, 1], [-1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    y_pred = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]])

    scores = per_sample_foreground_iou(y_true, y_pred, ignore_index=-1)

    assert scores[0] == pytest.approx(1 / 2)
    assert scores[1] == pytest.approx(0.0)
    assert math.isnan(scores[2])


def test_to_metric_records_flags_primary_and_classes() -> None:
    records = to_metric_records(
        {"mean_iou": 0.5, "macro_f1": 0.6, "iou/class_3": 0.25},
        primary_name="mean_iou",
        direction="maximize",
        split="official_validation",
        unit="%",
    )

    by_key = {(record.name, record.class_name): record for record in records}
    assert by_key[("mean_iou", None)].is_primary is True
    assert by_key[("macro_f1", None)].is_primary is False
    assert by_key[("iou", "class_3")].class_name == "class_3"
    assert by_key[("iou", "class_3")].split == "official_validation"
    assert all(isinstance(record, MetricRecord) for record in records)
