"""CPU-only regression tests for the Clay T11D fusion diagnostic.

These run in the Clay worker environment (the root environment has no torch and
skips them), matching the existing worker-level tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from clay_worker import fusion  # noqa: E402


def test_normalize_branch_has_declared_norm() -> None:
    z = torch.randn(2, fusion.TEACHER_DIMS, 16, 16)
    normalized = fusion._normalize_branch(z)
    assert normalized.shape == z.shape
    norms = normalized.norm(dim=1)
    expected = float(fusion.TEACHER_DIMS) ** 0.5
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-4)


def test_expand_head_keeps_raw_logits_when_new_channels_are_zero() -> None:
    base = fusion.build_head(fusion.TOKEN_DIM, 20, fusion.HEAD_HIDDEN)
    wide = fusion.build_head(fusion.TOKEN_DIM + fusion.TEACHER_DIMS, 20, fusion.HEAD_HIDDEN)
    fusion._expand_head(base, wide)
    features = torch.randn(1, fusion.TOKEN_DIM, 16, 16)
    zeros = torch.zeros(1, fusion.TEACHER_DIMS, 16, 16)
    base.eval()
    wide.eval()
    with torch.no_grad():
        raw = base(features)
        expanded = wide(torch.cat([features, zeros], dim=1))
    assert torch.allclose(raw, expanded, atol=1e-6)


def test_head_input_dimensions_and_raw_passthrough() -> None:
    features = torch.randn(1, fusion.TOKEN_DIM, 16, 16)
    projector = fusion.build_projector(fusion.TOKEN_DIM, fusion.TEACHER_DIMS)
    combined = fusion._head_input(features, projector)
    assert combined.shape == (1, fusion.TOKEN_DIM + fusion.TEACHER_DIMS, 16, 16)
    assert torch.equal(fusion._head_input(features, None), features)


def test_frozen_branches_have_no_gradients() -> None:
    projector = fusion.build_projector(fusion.TOKEN_DIM, fusion.TEACHER_DIMS)
    for parameter in projector.parameters():
        parameter.requires_grad = False
    state_before = {key: value.clone() for key, value in projector.state_dict().items()}
    features = torch.randn(1, fusion.TOKEN_DIM, 16, 16)
    fusion._head_input(features, projector)
    for key, value in projector.state_dict().items():
        assert torch.equal(value, state_before[key])


def _source_record(diagnostic: bool = True) -> dict:
    return {
        "status": "completed",
        "protocol_id": fusion.SOURCE_PROTOCOL,
        "seed": 17,
        "git_commit": "deadbeef",
        "training": {"epochs_completed": 8},
        "tags": {
            "panel": "diagnostic" if diagnostic else "primary",
            "init": "pretrained",
            "label_fraction": "1.0",
        },
        "dataset": {"manifest_sha256": "manifest"},
    }


def test_source_lock_records_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fusion, "cache_root", lambda: tmp_path / "cache")
    request = SimpleNamespace(runs_root=str(tmp_path / "runs"), seed=17)
    run_dir = tmp_path / "runs" / fusion.source_run_id(17)
    run_dir.mkdir(parents=True)
    (run_dir / "model.pt").write_bytes(b"adapted")
    (run_dir / "run.json").write_text(json.dumps(_source_record()), encoding="utf-8")
    checkpoint = tmp_path / "original.ckpt"
    checkpoint.write_bytes(b"original")

    lock = fusion._source_lock(request, 17, checkpoint)

    assert lock["source_run_id"] == fusion.source_run_id(17)
    assert lock["adapted_model_sha256"]
    assert lock["original_checkpoint_sha256"]
    assert lock["required_fixes"] == list(fusion.REQUIRED_SOURCE_FIXES)
    saved = tmp_path / "cache" / "fusion" / "source_lock" / "seed17.json"
    assert saved.is_file()


def test_source_lock_rejects_primary_panel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fusion, "cache_root", lambda: tmp_path / "cache")
    request = SimpleNamespace(runs_root=str(tmp_path / "runs"), seed=17)
    run_dir = tmp_path / "runs" / fusion.source_run_id(17)
    run_dir.mkdir(parents=True)
    (run_dir / "model.pt").write_bytes(b"adapted")
    (run_dir / "run.json").write_text(
        json.dumps(_source_record(diagnostic=False)), encoding="utf-8"
    )
    checkpoint = tmp_path / "original.ckpt"
    checkpoint.write_bytes(b"original")

    with pytest.raises(fusion.RgbnFailure, match="diagnostic"):
        fusion._source_lock(request, 17, checkpoint)


def test_fusion_cache_roundtrip_is_order_independent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fusion, "cache_root", lambda: tmp_path / "cache")
    profile = fusion.BAND_PROFILES["T11D"]
    encoder = SimpleNamespace()
    mean = torch.zeros(4)
    std = torch.ones(4)
    waves = torch.tensor([0.49, 0.56, 0.66, 0.84])
    cache = fusion.FusionCache(
        tmp_path / "cache", "fingerprint", encoder, mean, std, waves, "cpu", profile
    )
    sample_id = "10000"
    features = np.zeros((fusion.TOKEN_DIM, fusion.TOKENS, fusion.TOKENS), np.float32)
    np.save(cache.directory / f"{sample_id}.npy", features)
    (cache.directory / f"{sample_id}.json").write_text(
        json.dumps(
            {
                "version": fusion.CACHE_VERSION,
                "fingerprint": "fingerprint",
                "shape": [fusion.TOKEN_DIM, fusion.TOKENS, fusion.TOKENS],
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    loaded = cache.get({"sample_id": sample_id}, tmp_path)
    assert loaded.shape == (fusion.TOKEN_DIM, fusion.TOKENS, fusion.TOKENS)
    assert cache.hits == 1


def test_zero_train_limit_does_not_relax_the_minimum(monkeypatch) -> None:
    monkeypatch.setenv("CLAY_WORKER_TRAIN_LIMIT", "0")
    assert fusion._train_limit() == 0
    assert fusion._fusion_minimum_rows() == fusion._minimum_train_rows()
    request = SimpleNamespace(run_id="cell-train-17")
    assert fusion._budget_identity(request)["smoke"] is False


def test_budget_identity_marks_a_positive_train_limit_as_smoke(monkeypatch) -> None:
    monkeypatch.setenv("CLAY_WORKER_TRAIN_LIMIT", "2")
    request = SimpleNamespace(run_id="cell-train-17")
    budget = fusion._budget_identity(request)
    assert budget["smoke"] is True
    assert budget["train_limit"] == 2
