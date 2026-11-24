"""CPU tests for the T13 fusion diagnostics on a frozen FT-B0 head (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_fusion.py -q
"""

from __future__ import annotations

import hashlib
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
RUN = "lim_mit_huge-T13-whu_ft_b0-train-17"
COMMIT = "b431d11a91168b08dd7377129545a1021d213344"


def _run(root: Path, *, body: bytes = b"weights", **overrides) -> Path:
    """A baseline-shaped run directory with a checkpoint whose hash is returned via pins."""
    record = {
        "run_id": RUN,
        "task_id": "T13",
        "protocol_id": "whu_ft_b0",
        "seed": 17,
        "status": "completed",
        "notes": "",
        "git_commit": COMMIT,
        "tags": {"epochs_budget": "20"},
        "training": {"epochs_completed": 20},
    }
    record.update(overrides)
    run_dir = root / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "best_full_checkpoint.pt").write_bytes(body)
    return run_dir


def _pins(body: bytes = b"weights") -> dict:
    return {
        17: {
            "run_id": RUN,
            "git_commit": COMMIT,
            "checkpoint_sha256": hashlib.sha256(body).hexdigest(),
        }
    }


def test_pinned_baseline_returns_its_full_checkpoint(tmp_path: Path) -> None:
    run = _run(tmp_path)
    got = find_full_dependency(tmp_path, seed=17, pins=_pins())
    assert got == run / "best_full_checkpoint.pt"


def test_unpinned_seed_is_refused(tmp_path: Path) -> None:
    _run(tmp_path)
    with pytest.raises(RuntimeError, match="no pinned FT-B0 baseline for seed 73"):
        find_full_dependency(tmp_path, seed=73, pins=_pins())


def test_checkpoint_replaced_at_the_same_path_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path, body=b"other")
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        find_full_dependency(tmp_path, seed=17, pins=_pins(b"weights"))
    assert (run / "best_full_checkpoint.pt").is_file()


def test_record_without_training_commit_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    record = json.loads((run / "run.json").read_text())
    del record["git_commit"]
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="training commit"):
        find_full_dependency(tmp_path, seed=17, pins=_pins(), verify_sha=False)


def test_superseded_or_smoke_baseline_is_refused(tmp_path: Path) -> None:
    _run(tmp_path, tags={"epochs_budget": "20", "superseded": "yes"})
    with pytest.raises(RuntimeError, match="not superseded"):
        find_full_dependency(tmp_path, seed=17, pins=_pins(), verify_sha=False)


def test_unrelated_broken_records_do_not_block_the_lookup(tmp_path: Path) -> None:
    _run(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert find_full_dependency(tmp_path, seed=17, pins=_pins(), verify_sha=False)


def test_unreadable_pinned_record_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not an object"):
        find_full_dependency(tmp_path, seed=17, pins=_pins(), verify_sha=False)


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


def test_nan_in_invalid_pixels_does_not_reach_the_logits() -> None:
    head, levels = _head_and_levels()
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    _randomise(branches)
    nir, sar, _ = _modalities()
    valid = torch.ones(2, 1, SIZE, SIZE)
    valid[..., :, SIZE // 2 :] = 0  # right half missing
    nir[..., :, SIZE // 2 :] = float("nan")
    sar[..., :, SIZE // 2 :] = float("nan")
    out = fused_logits(
        head, levels, branches, (SIZE, SIZE), nir=nir, nir_valid=valid, sar=sar, sar_valid=valid
    )
    assert torch.isfinite(out).all()


def test_fully_missing_nan_modality_equals_the_baseline() -> None:
    head, levels = _head_and_levels()
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    _randomise(branches)
    nir = torch.full((2, 1, SIZE, SIZE), float("nan"))
    zero = torch.zeros(2, 1, SIZE, SIZE)
    baseline = head(levels, output_size=(SIZE, SIZE))
    out = fused_logits(head, levels, branches, (SIZE, SIZE), nir=nir, nir_valid=zero)
    assert torch.isfinite(out).all()
    assert torch.allclose(baseline, out, atol=1e-6)


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
    logits.square().mean().backward()
    optimizer.step()
    moved = [p for p in trainable if not torch.equal(before[id(p)], p.detach())]
    assert moved, "no branch parameter changed after one optimiser step"
