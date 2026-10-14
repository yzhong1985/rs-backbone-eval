"""Tests for content-addressed cache keys and per-group confusion accumulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rs_backbone_eval.cache import cache_key, model_identity
from rs_backbone_eval.evaluation.confusion import GroupConfusion


def test_cache_key_depends_on_manifest_content_and_parts(tmp_path: Path) -> None:
    manifest = tmp_path / "T1.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    first = cache_key(manifest, "train", 10)
    assert first == cache_key(manifest, "train", 10)
    assert first != cache_key(manifest, "validation", 10)

    manifest.write_text("{}\n{}\n", encoding="utf-8")
    assert cache_key(manifest, "train", 10) != first


def test_model_identity_separates_init_and_seed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    pretrained = model_identity(checkpoint)
    random_a = model_identity(checkpoint, seed=42, random_init=True)
    random_b = model_identity(checkpoint, seed=17, random_init=True)
    assert pretrained != random_a
    assert random_a != random_b
    assert model_identity(checkpoint) == pretrained


def test_group_confusion_keeps_groups_separate() -> None:
    grouped = GroupConfusion(num_classes=2)
    grouped.update("tileA", np.array([0, 1]), np.array([0, 1]))
    grouped.update("tileB", np.array([0, 0]), np.array([1, 0]))

    assert grouped.groups == ["tileA", "tileB"]
    payload = grouped.to_json()
    assert payload["tileA"] == [[1, 0], [0, 1]]
    assert payload["tileB"] == [[1, 1], [0, 0]]
    assert grouped.combined().counts.tolist() == [[2, 1], [0, 1]]
