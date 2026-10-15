"""Regression tests for the OlmoEarth stage-B guards (matrix types, protocol)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "olmoearth" / "src"))

from olmoearth_worker.t1_frozen import _evaluate_features, frozen_backbone_protocol  # noqa: E402


class _StubDataset:
    """Minimal rows attribute, as far as evaluation reads it."""

    def __init__(self, count: int) -> None:
        self.rows = [{"group_id": f"tile{index}"} for index in range(count)]


def test_evaluation_without_groups_returns_plain_metrics() -> None:
    """collect_groups=False must not touch group-only attributes."""
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 2, kernel_size=1))
    features = torch.randn(3, 4, 14, 14)
    labels = torch.full((3, 512, 512), 1, dtype=torch.int64)  # full-label scoring
    metrics, group_json = _evaluate_features(model, features, labels, _StubDataset(3))
    assert group_json is None
    assert "water_iou" in metrics


def test_evaluation_with_groups_returns_group_confusion() -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 2, kernel_size=1))
    features = torch.randn(3, 4, 14, 14)
    labels = torch.full((3, 512, 512), 1, dtype=torch.int64)
    metrics, group_json = _evaluate_features(
        model, features, labels, _StubDataset(3), collect_groups=True
    )
    assert set(group_json or {}) == {"tile0", "tile1", "tile2"}
    assert "water_iou" in metrics


def test_finetune_protocol_is_rejected_and_frozen_adapter_accepted() -> None:
    protocols = ROOT / "configs" / "protocols"
    allowed, message = frozen_backbone_protocol(protocols / "frozen_adapter.yaml")
    assert allowed and not message
    for name in ("full_finetune.yaml", "reference.yaml", "native_input.yaml"):
        path = protocols / name
        if path.is_file():
            allowed, message = frozen_backbone_protocol(path)
            assert not allowed
            assert "train_backbone" in message
