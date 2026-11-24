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


MANIFEST = "93483483b9a7f8b29ad8896010eebe3b7963a78dd240c4512389a1b7fdfce91e"
BACKBONE = "3c2e1b2e9257d8a74330a507145b82d4e6dd5bea65705aec73c8ae0caa8a6985"


def _no_admit(run: dict, repo: Path) -> None:
    return None


def _run(root: Path, *, body: bytes = b"weights", **overrides) -> Path:
    """A baseline-shaped run directory with a checkpoint file."""
    record = {
        "run_id": RUN,
        "task_id": "T13",
        "protocol_id": "whu_ft_b0",
        "seed": 17,
        "model_id": "lim_mit_huge",
        "status": "completed",
        "notes": "",
        "git_commit": COMMIT,
        "model_checkpoint_sha256": BACKBONE,
        "dataset": {"manifest_sha256": MANIFEST},
        "tags": {"epochs_budget": "20", "effective_batch": "8"},
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
            "manifest_sha256": MANIFEST,
            "backbone_sha256": BACKBONE,
            "model_id": "lim_mit_huge",
            "effective_batch": "8",
        }
    }


def _call(
    root: Path,
    *,
    seed: int = 17,
    manifest: str = MANIFEST,
    batch: int = 8,
    pins: dict | None = None,
    verify_sha: bool = True,
    admit=_no_admit,
) -> Path:
    return find_full_dependency(
        root,
        seed=seed,
        manifest_sha=manifest,
        model_id="lim_mit_huge",
        batch=batch,
        repo=root,
        pins=pins if pins is not None else _pins(),
        verify_sha=verify_sha,
        admit=admit,
    )


def test_pinned_baseline_returns_its_full_checkpoint(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert _call(tmp_path) == run / "best_full_checkpoint.pt"


def test_unpinned_seed_is_refused(tmp_path: Path) -> None:
    _run(tmp_path)
    with pytest.raises(RuntimeError, match="no pinned FT-B0 baseline for seed 73"):
        _call(tmp_path, seed=73)


def test_checkpoint_replaced_at_the_same_path_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path, body=b"other")
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        _call(tmp_path, pins=_pins(b"weights"))
    assert (run / "best_full_checkpoint.pt").is_file()


@pytest.mark.parametrize(
    ("overrides", "failed"),
    [
        ({"git_commit": None}, "training commit"),
        (
            {"tags": {"epochs_budget": "20", "effective_batch": "8", "superseded": "yes"}},
            "not superseded",
        ),
        ({"notes": "REDUCED-BUDGET smoke-like run"}, "not reduced budget"),
        ({"dataset": {"manifest_sha256": "0" * 64}}, "manifest"),
        ({"model_checkpoint_sha256": "1" * 64}, "backbone"),
    ],
)
def test_identity_and_admission_mismatches_are_refused(tmp_path: Path, overrides, failed) -> None:
    _run(tmp_path, **overrides)
    with pytest.raises(RuntimeError, match=failed):
        _call(tmp_path, pins=_pins(), verify_sha=False)


def test_request_with_another_manifest_or_batch_is_refused(tmp_path: Path) -> None:
    _run(tmp_path)
    with pytest.raises(RuntimeError, match="manifest"):
        _call(tmp_path, manifest="2" * 64, verify_sha=False)
    with pytest.raises(RuntimeError, match="effective batch"):
        _call(tmp_path, batch=4, verify_sha=False)


def test_admission_hook_can_refuse_a_formal_result(tmp_path: Path) -> None:
    _run(tmp_path)

    def refuse(run: dict, repo: Path) -> None:
        raise RuntimeError("excluded: stale commit")

    with pytest.raises(RuntimeError, match="excluded: stale commit"):
        _call(tmp_path, verify_sha=False, admit=refuse)


def test_unrelated_broken_records_do_not_block_the_lookup(tmp_path: Path) -> None:
    _run(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert _call(tmp_path, verify_sha=False)


def test_unreadable_pinned_record_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not an object"):
        _call(tmp_path, verify_sha=False)


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
