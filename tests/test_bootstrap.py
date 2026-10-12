import math

import numpy as np
import pytest

from rs_backbone_eval.evaluation.bootstrap import bootstrap_ci


def test_bootstrap_ci_is_deterministic_for_seed() -> None:
    values = np.arange(1, 101, dtype=float)

    first = bootstrap_ci(values, seed=17)
    second = bootstrap_ci(values, seed=17)
    other = bootstrap_ci(values, seed=42)

    assert first == second
    assert first.point == pytest.approx(50.5)
    assert first != other
    assert first.low <= first.point <= first.high


def test_bootstrap_ci_brackets_the_mean_for_spread_values() -> None:
    values = np.array([0.0, 1.0] * 25)

    interval = bootstrap_ci(values, confidence=0.95, n_resamples=2000, seed=17)

    assert interval.sample_count == 50
    assert interval.n_resamples == 2000
    assert interval.point == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.5, abs=0.15)
    assert interval.high == pytest.approx(0.5, abs=0.15)


def test_bootstrap_ci_degenerate_cases() -> None:
    single = bootstrap_ci(np.array([3.0]))
    assert single.point == single.low == single.high == 3.0

    dropped = bootstrap_ci(np.array([1.0, math.nan, 2.0]))
    assert dropped.sample_count == 2


def test_bootstrap_ci_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(np.array([1.0]), confidence=1.5)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci(np.array([math.nan]))
