"""CPU-only tests for the warmup + cosine schedule used by the budget controls."""

from __future__ import annotations

import pytest

from rs_backbone_eval.schedule import warmup_cosine_factor


def test_warmup_starts_at_zero_reaches_peak_then_decays_to_zero() -> None:
    assert warmup_cosine_factor(0, 100, 10) == pytest.approx(0.0)
    assert warmup_cosine_factor(5, 100, 10) == pytest.approx(0.5)
    assert warmup_cosine_factor(10, 100, 10) == pytest.approx(1.0)
    assert warmup_cosine_factor(100, 100, 10) == pytest.approx(0.0)
    assert 0.0 < warmup_cosine_factor(55, 100, 10) < 1.0


def test_no_warmup_decays_from_the_first_step_and_handles_zero_budget() -> None:
    assert warmup_cosine_factor(0, 100, 0) == pytest.approx(1.0)
    assert warmup_cosine_factor(100, 100, 0) == pytest.approx(0.0)
    assert warmup_cosine_factor(3, 0, 0) == 0.0
