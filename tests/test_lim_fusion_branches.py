"""CPU tests for the NIR/SAR residual branches (LIM worker env, no data, no GPU).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_fusion_branches.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.fusion_branches import (  # noqa: E402
    FusionBranches,
    NirBranch,
    SarBranch,
    masked_mean,
)

CHANNELS = [256, 512, 1024, 2048]
IMAGE = 512  # stride 4/8/16/32 grid: 128, 64, 32, 16


def _pyramid(batch: int = 2, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    sizes = [IMAGE // s for s in (4, 8, 16, 32)]
    return [
        torch.randn(batch, c, n, n, generator=gen) for c, n in zip(CHANNELS, sizes, strict=True)
    ]


def _image_input(batch: int = 2, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 1, IMAGE, IMAGE, generator=gen)


def _randomise_projections(branches: FusionBranches, seed: int = 2) -> None:
    gen = torch.Generator().manual_seed(seed)
    modules = [branches.nir, branches.sar]
    for module in modules:
        if module is None:
            continue
        for proj in module.proj.values():
            with torch.no_grad():
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=gen) * 0.05)
                proj.bias.copy_(torch.randn(proj.bias.shape, generator=gen) * 0.05)


def test_zero_init_leaves_baseline_pyramid_unchanged() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    levels = _pyramid()
    out = branches(
        levels,
        nir=_image_input(),
        sar=_image_input(seed=3),
    )
    for before, after in zip(levels, out, strict=True):
        assert torch.equal(before, after)


def test_no_modality_passed_returns_pyramid_unchanged() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).eval()
    _randomise_projections(branches)
    levels = _pyramid()
    out = branches(levels)
    for before, after in zip(levels, out, strict=True):
        assert torch.equal(before, after)


def test_disabled_branch_is_skipped_even_when_its_input_is_given() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=False).eval()
    _randomise_projections(branches)
    levels = _pyramid()
    out = branches(levels, sar=_image_input(seed=5))
    for before, after in zip(levels, out, strict=True):
        assert torch.equal(before, after)


def test_invalid_pixels_get_no_residual() -> None:
    torch.manual_seed(0)
    nir = NirBranch(CHANNELS).eval()
    for proj in nir.proj.values():
        with torch.no_grad():
            proj.weight.fill_(0.1)
            proj.bias.fill_(0.1)
    levels = _pyramid()
    valid = torch.zeros(2, 1, IMAGE, IMAGE)
    valid[..., :, :256] = 1.0  # left half valid, right half invalid
    out = nir(levels, _image_input(), valid)
    for i in nir.levels:
        size = levels[i].shape[-1]
        half = size // 2
        right_delta = (out[i] - levels[i])[..., half:]
        assert torch.equal(right_delta, torch.zeros_like(right_delta))
        left_delta = (out[i] - levels[i])[..., :half]
        assert left_delta.abs().sum() > 0


def test_all_invalid_sar_leaves_pyramid_unchanged_even_with_weights() -> None:
    torch.manual_seed(0)
    sar = SarBranch(CHANNELS).eval()
    for proj in sar.proj.values():
        with torch.no_grad():
            proj.weight.fill_(0.1)
            proj.bias.fill_(0.1)
    levels = _pyramid()
    out = sar(levels, _image_input(), torch.zeros(2, 1, IMAGE, IMAGE))
    for before, after in zip(levels, out, strict=True):
        assert torch.equal(before, after)


def test_masked_mean_uses_valid_pixels_only_and_zero_when_empty() -> None:
    feat = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    got = masked_mean(feat, mask)
    expected = torch.tensor([[[[(0 + 2) / 2]], [[(4 + 6) / 2]]]])
    assert torch.allclose(got, expected)
    assert torch.equal(masked_mean(feat, torch.zeros(1, 1, 2, 2)), torch.zeros(1, 2, 1, 1))


def test_gradients_reach_new_parameters_at_init() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_nir=True, use_sar=True).train()
    levels = _pyramid()
    out = branches(levels, nir=_image_input(), sar=_image_input(seed=3))
    loss = sum(level.square().mean() for level in out)
    loss.backward()
    # the zero-initialised projections get gradient from the features
    for module in (branches.nir, branches.sar):
        for proj in module.proj.values():
            assert torch.isfinite(proj.weight.grad).all()
            assert proj.weight.grad.abs().sum() > 0


def test_one_update_changes_projection_and_keeps_baseline_levels_unmodified() -> None:
    torch.manual_seed(0)
    branches = FusionBranches(CHANNELS, use_sar=True).train()
    levels = [level.detach().clone().requires_grad_(False) for level in _pyramid()]
    optimizer = torch.optim.AdamW(branches.parameters(), lr=1e-3, weight_decay=1e-4)
    before = {k: v.detach().clone() for k, v in branches.state_dict().items()}
    out = branches(levels, sar=_image_input(seed=3))
    loss = sum(level.square().mean() for level in out)
    loss.backward()
    optimizer.step()
    changed = [
        k
        for k, v in branches.state_dict().items()
        if v.dtype.is_floating_point and not torch.equal(before[k], v)
    ]
    assert any(k.startswith("sar.proj") for k in changed)
    assert all(level.requires_grad is False for level in levels)


def test_new_parameters_stay_well_below_design_budget() -> None:
    nir = NirBranch(CHANNELS)
    sar = SarBranch(CHANNELS)
    nir_n = sum(p.numel() for p in nir.parameters())
    sar_n = sum(p.numel() for p in sar.parameters())
    assert nir_n + sar_n < 1_000_000, (nir_n, sar_n)
