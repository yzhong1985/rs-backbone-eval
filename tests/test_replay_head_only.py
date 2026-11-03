"""Regression tests for the head_only replay path (review 2026-11-03).

A pruned run keeps only ``head_only.pt``; the frozen backbone must be rebuilt
into the encoder the model actually uses. The previous code loaded it into a
throwaway ``mit_huge()``, so the strict state-dict load failed and every pruned
run became unreplayable.

These tests need torch, so the root environment skips them. Run with:

    uv run --project workers/lim --with pytest pytest tests/test_replay_head_only.py
"""

import hashlib
import types

import pytest

torch = pytest.importorskip("torch")
replay = pytest.importorskip("lim_worker.replay")


def _context(tmp_path, encoder):
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"frozen-weights")
    digest = hashlib.sha256(b"frozen-weights").hexdigest()
    context = types.SimpleNamespace(
        checkpoint_kind="head_only",
        checkpoint_path=tmp_path / "head_only.pt",
        checkpoint_sha256="0" * 64,
        tags={},
        model=types.SimpleNamespace(encoder=encoder),
    )
    return context, backbone, digest


def test_head_only_loads_the_backbone_into_the_model_encoder(tmp_path, monkeypatch) -> None:
    encoder = torch.nn.Linear(2, 2)
    context, backbone, digest = _context(tmp_path, encoder)
    monkeypatch.setenv("LIM_CHECKPOINT", str(backbone))
    calls = {}

    def fake_load(path, target):
        calls["target"] = target
        with torch.no_grad():
            target.weight.fill_(3.0)
        return {"loaded": 4, "missing": 0, "unexpected": 0}

    monkeypatch.setattr("lim_worker.mit.load_encoder", fake_load)
    payload = {
        "identity": {"pretrained_checkpoint": digest},
        "head_state_dict": {"weight": torch.zeros(2, 2)},
    }
    state = replay._load_state(context, payload)

    assert calls["target"] is encoder, "backbone must be loaded into model.encoder, not a copy"
    assert torch.all(encoder.weight == 3.0)
    assert "weight" in state


def test_head_only_rejects_a_backbone_that_does_not_match_the_run(tmp_path, monkeypatch) -> None:
    encoder = torch.nn.Linear(2, 2)
    context, backbone, _ = _context(tmp_path, encoder)
    monkeypatch.setenv("LIM_CHECKPOINT", str(backbone))
    payload = {
        "identity": {"pretrained_checkpoint": "f" * 64},
        "head_state_dict": {"weight": torch.zeros(2, 2)},
    }
    with pytest.raises(ValueError, match="sha256"):
        replay._load_state(context, payload)


def test_head_only_requires_the_backbone_checkpoint(tmp_path, monkeypatch) -> None:
    encoder = torch.nn.Linear(2, 2)
    context, _, digest = _context(tmp_path, encoder)
    monkeypatch.delenv("LIM_CHECKPOINT", raising=False)
    payload = {
        "identity": {"pretrained_checkpoint": digest},
        "head_state_dict": {"weight": torch.zeros(2, 2)},
    }
    with pytest.raises(ValueError, match="LIM_CHECKPOINT"):
        replay._load_state(context, payload)
