"""CPU tests for the T13 fusion diagnostics on a frozen FT-B0 head (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_fusion.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.fusion_branches import FusionBranches  # noqa: E402
from lim_worker.t13_fusion import (  # noqa: E402
    branch_parameters,
    find_full_dependency,
    fused_logits,
)

from rs_backbone_eval.heads import UPerNetHead  # noqa: E402

CHANNELS = [32, 64, 128, 256]  # small stand-in for the LIM pyramid
SIZE = 64  # levels at 16, 8, 4, 2


def _run(root: Path, name: str, **overrides) -> Path:
    record = {
        "task_id": "T13",
        "protocol_id": "whu_ft_b0",
        "seed": 17,
        "model_id": "lim_mit_huge",
        "status": "completed",
        "notes": "",
        "tags": {"epochs_budget": "20", "effective_batch": "8"},
        "dataset": {"manifest_sha256": "m" * 64},
        "model_checkpoint_sha256": "b" * 64,
        "training": {"epochs_completed": 20},
    }
    for key, value in overrides.items():
        record[key] = value
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def _dependency(root: Path, **kwargs):
    args = dict(
        seed=17,
        model_id="lim_mit_huge",
        manifest_sha="m" * 64,
        backbone_sha="b" * 64,
        epochs=20,
        batch=8,
    )
    args.update(kwargs)
    return find_full_dependency(root, **args)


def test_dependency_returns_the_single_completed_full_checkpoint(tmp_path: Path) -> None:
    run = _run(tmp_path, "lim_mit_huge-T13-whu_ft_b0-train-17")
    (run / "best_full_checkpoint.pt").write_bytes(b"x")
    assert _dependency(tmp_path) == run / "best_full_checkpoint.pt"


def test_dependency_refuses_missing_checkpoint(tmp_path: Path) -> None:
    _run(tmp_path, "lim_mit_huge-T13-whu_ft_b0-train-17")
    with pytest.raises(RuntimeError, match="best_full_checkpoint"):
        _dependency(tmp_path)


def test_dependency_ignores_superseded_and_mismatched_runs(tmp_path: Path) -> None:
    bad = _run(
        tmp_path, "old", tags={"epochs_budget": "20", "effective_batch": "8", "superseded": "yes"}
    )
    (bad / "best_full_checkpoint.pt").write_bytes(b"x")
    good = _run(tmp_path, "good")
    (good / "best_full_checkpoint.pt").write_bytes(b"x")
    assert _dependency(tmp_path) == good / "best_full_checkpoint.pt"
    with pytest.raises(RuntimeError, match="found 0"):
        _dependency(tmp_path, manifest_sha="z" * 64)


def test_dependency_refuses_two_matching_runs(tmp_path: Path) -> None:
    for name in ("a", "b"):
        run = _run(tmp_path, name)
        (run / "best_full_checkpoint.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="found 2"):
        _dependency(tmp_path)


def _head_and_levels(seed: int = 0):
    torch.manual_seed(seed)
    head = UPerNetHead(CHANNELS, num_classes=8, fpn_dim=32).eval()
    gen = torch.Generator().manual_seed(seed + 1)
    sizes = [SIZE // s for s in (4, 8, 16, 32)]
    levels = [torch.randn(2, c, n, n, generator=gen) for c, n in zip(CHANNELS, sizes, strict=True)]
    return head, levels


def _modalities(seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    nir = torch.rand(2, 1, SIZE, SIZE, generator=gen)
    sar = torch.rand(2, 1, SIZE, SIZE, generator=gen)
    valid = torch.ones(2, 1, SIZE, SIZE)
    return nir, sar, valid


def _randomise(branches: FusionBranches) -> None:
    gen = torch.Generator().manual_seed(9)
    for module in (branches.nir, branches.sar):
        if module is None:
            continue
        for proj in module.proj.values():
            with torch.no_grad():
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=gen) * 0.05)


def test_zero_branches_reproduce_the_frozen_head_logits() -> None:
    head, levels = _head_and_levels()
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    nir, sar, valid = _modalities()
    baseline = head(levels, output_size=(SIZE, SIZE))
    fused = fused_logits(
        head, levels, branches, (SIZE, SIZE), nir=nir, nir_valid=valid, sar=sar, sar_valid=valid
    )
    assert torch.equal(baseline, fused)


def test_all_invalid_masks_keep_the_baseline_even_with_nonzero_weights() -> None:
    head, levels = _head_and_levels()
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    _randomise(branches)
    nir, sar, _ = _modalities()
    zero = torch.zeros(2, 1, SIZE, SIZE)
    baseline = head(levels, output_size=(SIZE, SIZE))
    fused = fused_logits(
        head, levels, branches, (SIZE, SIZE), nir=nir, nir_valid=zero, sar=sar, sar_valid=zero
    )
    assert torch.allclose(baseline, fused, atol=1e-6)


def test_only_branch_parameters_train_and_one_step_moves_them() -> None:
    head, levels = _head_and_levels()
    for param in head.parameters():
        param.requires_grad_(False)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).train()
    trainable = branch_parameters(branches)
    assert trainable and all(p.requires_grad for p in trainable)
    assert all(not p.requires_grad for p in head.parameters())
    nir, sar, valid = _modalities()
    before = {id(p): p.detach().clone() for p in trainable}
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    logits = fused_logits(
        head, levels, branches, (SIZE, SIZE), nir=nir, nir_valid=valid, sar=sar, sar_valid=valid
    )
    loss = logits.square().mean()
    loss.backward()
    optimizer.step()
    moved = [p for p in trainable if not torch.equal(before[id(p)], p.detach())]
    assert moved, "no branch parameter changed after one optimiser step"
