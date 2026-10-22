"""Tests for the OlmoEarth T2/T3/T4 frozen pipeline and the Base variant."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "olmoearth" / "src"))

from olmoearth_worker.bands import bandset_usable, to_olmoearth_s2l2a  # noqa: E402
from olmoearth_worker.tasks import (  # noqa: E402
    RECIPES,
    _metrics,
    _relative_timestamps,
    build_criterion,
)

T2_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
T3_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]


def _labels_dataset(num_classes: int):
    class _Rows(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.rows = [{"group_id": "g"}]
            self.labels = np.array([[0, 1], [num_classes - 1, 1]], dtype=np.int64)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            return {"label": self.labels}

    return _Rows()


def test_recipes_match_the_task_specs() -> None:
    assert set(RECIPES) == {"T2", "T3", "T4"}
    assert RECIPES["T2"].loss == "ce" and RECIPES["T2"].metric_name == "burn_scar_iou"
    assert RECIPES["T2"].eval_split == "official_validation"
    assert RECIPES["T3"].loss == "weighted_ce" and RECIPES["T3"].num_classes == 5
    assert RECIPES["T3"].eval_split == "test"
    assert RECIPES["T4"].loss == "focal" and RECIPES["T4"].num_classes == 2


def test_broadband_nir_maps_to_b08_and_covers_the_first_bandset() -> None:
    """T2/T3 name the broadband NIR differently; both are B08, not B8A."""
    for bands in (T2_BANDS, T3_BANDS):
        expanded = to_olmoearth_s2l2a(np.ones((6, 2, 2), dtype=np.float32), bands, fill=0.0)
        assert expanded.shape[0] == 12
        # B08 is the 4th channel of the first band-set.
        assert np.allclose(expanded[3], 1.0)
        # B8A (the 8th channel) stays filled: broadband NIR is not narrow NIR.
        assert np.allclose(expanded[7], 0.0)
        assert bandset_usable(bands) == (True, True, False)


def test_relative_timestamps_keep_real_dates_and_number_stage_labels() -> None:
    assert _relative_timestamps(["2019-06-14"], 1) == [[14, 5, 2019]]
    assert _relative_timestamps(["pre", "during", "post"], 3) == [
        [1, 0, 2020],
        [2, 0, 2020],
        [3, 0, 2020],
    ]
    # Missing stamps still yield one ordered entry per date.
    assert _relative_timestamps([], 2) == [[1, 0, 2020], [2, 0, 2020]]


def test_criterion_follows_the_design_doc_per_task() -> None:
    t2 = build_criterion(RECIPES["T2"], _labels_dataset(2), "cpu")
    t3 = build_criterion(RECIPES["T3"], _labels_dataset(5), "cpu")
    t4 = build_criterion(RECIPES["T4"], _labels_dataset(2), "cpu")
    assert type(t2).__name__ == "CrossEntropyLoss"
    assert type(t3).__name__ == "WeightedCrossEntropy"
    assert type(t4).__name__ == "FocalLoss"
    # T3 weights are normalized to mean 1 over the present classes.
    present = t3.weight[t3.weight > 0]
    assert float(present.mean()) == pytest.approx(1.0, rel=1e-5)


def test_binary_metric_is_renamed_to_the_task_metric() -> None:
    from rs_backbone_eval.evaluation.confusion import ConfusionMatrix

    matrix = ConfusionMatrix(num_classes=2)
    matrix.update(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [0, 0]]), ignore_index=-1)
    values = _metrics(RECIPES["T2"], matrix)
    assert "burn_scar_iou" in values
    assert "foreground_iou" not in values


def test_registry_lists_tiny_and_base_as_separate_models() -> None:
    from rs_backbone_eval.registry import Registry

    registry = Registry(ROOT / "configs")
    models = {spec.model_id: spec for spec in registry.models()}
    assert "olmoearth_v1_2_tiny" in models
    assert "olmoearth_v1_2_base" in models
    assert models["olmoearth_v1_2_base"].checkpoint_env == "OLMOEARTH_BASE_CHECKPOINT"
    assert models["olmoearth_v1_2_tiny"].checkpoint_env == "OLMOEARTH_CHECKPOINT"
