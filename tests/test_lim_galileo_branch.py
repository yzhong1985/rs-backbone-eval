"""CPU tests for the frozen Galileo-Nano residual branch (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_galileo_branch.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.fusion_branches import GALILEO_DIM, GALILEO_LEVEL, GalileoBranch  # noqa: E402

CHANNELS = [256, 512, 1024, 2048]  # LIM widths; F3 has 1024 channels at 32x32 for 512 px
GRID = 32


def _levels(seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    sizes = [128, 64, 32, 16]
    return [torch.randn(2, c, s, s, generator=gen) for c, s in zip(CHANNELS, sizes, strict=True)]


def _tokens(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, GALILEO_DIM, GRID, GRID, generator=gen)


def test_zero_projection_returns_the_baseline_levels() -> None:
    branch = GalileoBranch(CHANNELS).eval()
    levels = _levels()
    out = branch(levels, _tokens(), torch.ones(2, 1, GRID, GRID))
    for a, b in zip(levels, out, strict=True):
        assert torch.equal(a, b)


def test_only_the_projection_is_trainable_and_it_is_small() -> None:
    branch = GalileoBranch(CHANNELS)
    total = sum(p.numel() for p in branch.parameters())
    assert total == GALILEO_DIM * CHANNELS[GALILEO_LEVEL] + CHANNELS[GALILEO_LEVEL]
    assert all(p.requires_grad for p in branch.parameters())


def test_invalid_tokens_never_reach_the_level() -> None:
    branch = GalileoBranch(CHANNELS).eval()
    with torch.no_grad():
        branch.proj.weight.normal_(0, 0.05)
    tokens = _tokens()
    tokens[..., 16:] = float("nan")
    valid = torch.ones(2, 1, GRID, GRID)
    valid[..., 16:] = 0
    levels = _levels()
    out = branch(levels, tokens, valid)
    assert torch.isfinite(out[GALILEO_LEVEL]).all()
    # Where the tokens are invalid the level must equal the baseline exactly.
    assert torch.equal(out[GALILEO_LEVEL][..., 16:], levels[GALILEO_LEVEL][..., 16:])


def test_one_step_moves_the_projection() -> None:
    branch = GalileoBranch(CHANNELS).train()
    levels = _levels()
    target = torch.randn_like(levels[GALILEO_LEVEL])
    out = branch(levels, _tokens(), torch.ones(2, 1, GRID, GRID))
    before = branch.proj.weight.detach().clone()
    loss = (out[GALILEO_LEVEL] - target).square().mean()
    loss.backward()
    torch.optim.AdamW(branch.parameters(), lr=1e-3).step()
    assert not torch.equal(before, branch.proj.weight.detach())
