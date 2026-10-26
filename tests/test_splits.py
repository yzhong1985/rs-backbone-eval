"""Holdout split resolution shared by the workers."""

from __future__ import annotations

import pytest

from rs_backbone_eval.splits import extra_evaluation_splits


def test_extra_evaluation_splits_refuses_non_holdouts(monkeypatch) -> None:
    monkeypatch.delenv("RS_EVAL_EXTRA_EVAL_SPLITS", raising=False)
    assert extra_evaluation_splits("test") == []
    monkeypatch.setenv("RS_EVAL_EXTRA_EVAL_SPLITS", "bolivia_holdout, bolivia_holdout")
    assert extra_evaluation_splits("test") == ["bolivia_holdout"]
    for bad in ("validation", "train", "test"):
        monkeypatch.setenv("RS_EVAL_EXTRA_EVAL_SPLITS", bad)
        with pytest.raises(ValueError):
            extra_evaluation_splits("test")
