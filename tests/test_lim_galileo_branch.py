"""CPU tests for the frozen Galileo-Nano residual branch on F2 and F3 (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_galileo_branch.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.fusion_branches import GALILEO_DIM, GALILEO_LEVELS, GalileoBranch  # noqa: E402

F23 = (1, 2)  # F2 + F3

CHANNELS = [256, 512, 1024, 2048]  # LIM widths
# 512 px tile: F1 128x128, F2 64x64, F3 32x32, F4 16x16 (strides 4, 8, 16, 32)
SIZES = [128, 64, 32, 16]
GRID = 64  # native Galileo patch grid for a 512 px tile


def _levels(seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(2, c, s, s, generator=gen) for c, s in zip(CHANNELS, SIZES, strict=True)]


def _tokens(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, GALILEO_DIM, GRID, GRID, generator=gen)


def test_default_levels_stay_f3_only() -> None:
    assert GALILEO_LEVELS == (2,)


def test_zero_projections_return_the_baseline_levels() -> None:
    branch = GalileoBranch(CHANNELS, levels=F23).eval()
    levels = _levels()
    out = branch(levels, _tokens(), torch.ones(2, 1, GRID, GRID))
    for a, b in zip(levels, out, strict=True):
        assert torch.equal(a, b)


def test_only_the_projections_are_trainable_and_they_are_small() -> None:
    branch = GalileoBranch(CHANNELS, levels=F23)
    total = sum(p.numel() for p in branch.parameters())
    expected = sum(GALILEO_DIM * CHANNELS[i] + CHANNELS[i] for i in F23)
    assert total == expected
    assert all(p.requires_grad for p in branch.parameters())


def test_invalid_tokens_never_reach_either_level() -> None:
    branch = GalileoBranch(CHANNELS, levels=F23).eval()
    with torch.no_grad():
        for proj in branch.proj.values():
            proj.weight.normal_(0, 0.05)
    tokens = _tokens()
    tokens[..., GRID // 2 :] = float("nan")
    valid = torch.ones(2, 1, GRID, GRID)
    valid[..., GRID // 2 :] = 0
    levels = _levels()
    out = branch(levels, tokens, valid)
    for i in F23:
        assert torch.isfinite(out[i]).all()
        half = SIZES[i] // 2
        assert torch.equal(out[i][..., half:], levels[i][..., half:])


def test_one_step_moves_both_projections() -> None:
    branch = GalileoBranch(CHANNELS, levels=F23).train()
    levels = _levels()
    out = branch(levels, _tokens(), torch.ones(2, 1, GRID, GRID))
    loss = sum((out[i] - torch.randn_like(out[i])).square().mean() for i in F23)
    before = {k: m.weight.detach().clone() for k, m in branch.proj.items()}
    loss.backward()
    torch.optim.AdamW(branch.parameters(), lr=1e-3).step()
    for k, m in branch.proj.items():
        assert not torch.equal(before[k], m.weight.detach()), k


def test_legacy_single_projection_checkpoint_loads_strictly() -> None:
    # Plugins trained before multi-level support hold the F3 projection at "proj.weight".
    current = GalileoBranch(CHANNELS)
    legacy = {k.replace("proj.2.", "proj.", 1): v.clone() for k, v in current.state_dict().items()}
    assert set(legacy) == {"proj.weight", "proj.bias"}
    fresh = GalileoBranch(CHANNELS)
    fresh.load_state_dict(legacy, strict=True)
    assert torch.equal(fresh.proj["2"].weight, current.proj["2"].weight)
