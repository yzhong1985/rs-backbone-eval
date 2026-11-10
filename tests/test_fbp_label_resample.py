"""Unit tests for the FBP 4 m -> 10 m label policy (no GDAL, no data needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fbp_label_resample.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fbp_label_resample", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mode_would_lose_a_40_percent_class_but_the_rule_keeps_it() -> None:
    """The Jiangxi scene's real numbers: 45% class, 27% unlabeled, 73% valid."""
    module = load_module()
    fractions = np.zeros((24, 1, 1), dtype=np.float32)
    fractions[2] = 0.4545  # class 3
    fractions[10] = 0.0594  # class 11
    coverage = np.ones((1, 1), dtype=np.float32)

    labels, unlabeled = module.combine(fractions, coverage)

    assert labels[0, 0] == 3
    assert unlabeled[0, 0] == pytest.approx(1 - 0.4545 - 0.0594, abs=1e-6)


def test_majority_unlabeled_becomes_ignore() -> None:
    module = load_module()
    fractions = np.zeros((24, 1, 2), dtype=np.float32)
    fractions[0, 0, 0] = 0.45  # 45% class 1, 55% unlabeled -> ignore
    fractions[0, 0, 1] = 0.60  # 60% class 1 -> keep
    coverage = np.ones((1, 2), dtype=np.float32)

    labels, unlabeled = module.combine(fractions, coverage)

    assert labels[0, 0] == module.IGNORE
    assert labels[0, 1] == 1
    assert unlabeled[0, 0] == pytest.approx(0.55)
    assert unlabeled[0, 1] == pytest.approx(0.40)


def test_outside_footprint_is_ignore_not_unlabeled() -> None:
    """A rotated scene leaves bbox corners uncovered; they are not 'unlabeled 100%'."""
    module = load_module()
    fractions = np.zeros((24, 1, 1), dtype=np.float32)
    coverage = np.zeros((1, 1), dtype=np.float32)

    labels, unlabeled = module.combine(fractions, coverage)

    assert labels[0, 0] == module.IGNORE
    assert unlabeled[0, 0] == pytest.approx(1.0)


def test_half_unlabeled_is_kept_at_the_threshold() -> None:
    module = load_module()
    fractions = np.zeros((24, 1, 1), dtype=np.float32)
    fractions[5, 0, 0] = 0.5
    coverage = np.ones((1, 1), dtype=np.float32)

    labels, _ = module.combine(fractions, coverage)

    assert labels[0, 0] == 6


def test_mismatched_shapes_are_rejected() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="do not match"):
        module.combine(np.zeros((24, 2, 2)), np.zeros((1, 1)))
