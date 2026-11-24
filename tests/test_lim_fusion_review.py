"""Regression tests for the review of 5a3e7a8 (LIM worker env, CPU only).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_fusion_review.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.fusion_branches import FusionBranches, resize_mask  # noqa: E402

CHANNELS = [256, 512, 1024, 2048]
IMAGE = 512


def _levels(batch: int = 1, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    sizes = [IMAGE // s for s in (4, 8, 16, 32)]
    return [
        torch.randn(batch, c, n, n, generator=gen) for c, n in zip(CHANNELS, sizes, strict=True)
    ]


def _randomise(branches: FusionBranches, seed: int = 2) -> None:
    gen = torch.Generator().manual_seed(seed)
    for module in (branches.nir, branches.sar):
        if module is None:
            continue
        for proj in module.proj.values():
            with torch.no_grad():
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=gen) * 0.05)
                proj.bias.copy_(torch.randn(proj.bias.shape, generator=gen) * 0.05)


def test_one_valid_pixel_gives_fractional_coverage_not_a_flip() -> None:
    first = torch.zeros(1, 1, IMAGE, IMAGE)
    first[..., 0, 0] = 1.0
    moved = torch.zeros(1, 1, IMAGE, IMAGE)
    moved[..., 0, 1] = 1.0
    a = resize_mask(first, (64, 64))[0, 0, 0, 0]
    b = resize_mask(moved, (64, 64))[0, 0, 0, 0]
    assert torch.isclose(a, torch.tensor(1 / 64))
    assert torch.isclose(b, torch.tensor(1 / 64))


def test_joint_output_is_baseline_plus_each_branch_residual() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    _randomise(branches)
    levels = _levels()
    nir = torch.rand(1, 1, IMAGE, IMAGE)
    sar = torch.rand(1, 1, IMAGE, IMAGE, generator=torch.Generator().manual_seed(4))
    valid = torch.ones(1, 1, IMAGE, IMAGE)
    joint = branches(levels, nir=nir, nir_valid=valid, sar=sar, sar_valid=valid)
    nir_only = branches.nir(levels, nir, valid)
    sar_only = branches.sar(levels, sar, valid)
    for i in range(len(levels)):
        expected = levels[i] + (nir_only[i] - levels[i]) + (sar_only[i] - levels[i])
        assert torch.allclose(joint[i], expected, atol=1e-6)


def test_sar_gate_does_not_see_nir_residual() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    _randomise(branches)
    levels = _levels()
    sar = torch.rand(1, 1, IMAGE, IMAGE, generator=torch.Generator().manual_seed(4))
    valid = torch.ones(1, 1, IMAGE, IMAGE)
    sar_alone = branches.sar(levels, sar, valid)
    nir = torch.rand(1, 1, IMAGE, IMAGE)
    joint = branches(levels, nir=nir, sar=sar)
    nir_delta = branches.nir(levels, nir, torch.ones_like(nir))
    for i in branches.sar.levels:
        sar_delta_joint = joint[i] - levels[i] - (nir_delta[i] - levels[i])
        assert torch.allclose(sar_delta_joint, sar_alone[i] - levels[i], atol=1e-6)
