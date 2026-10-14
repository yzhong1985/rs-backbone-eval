"""Tests for the D2 perturbation environment parsing."""

from __future__ import annotations

import pytest

from rs_backbone_eval.perturb import (
    BAND_DROP,
    DEFAULT_PERTURBATIONS,
    RESOLUTION,
    perturbations_from_env,
    resolve,
)


def test_experiments_are_semicolon_separated(monkeypatch) -> None:
    monkeypatch.setenv("RS_EVAL_PERTURB", "band_drop:NIR_NARROW,SWIR_1;resolution:2")
    parsed = perturbations_from_env()
    assert parsed == ["band_drop:NIR_NARROW,SWIR_1", "resolution:2"]
    # the band list stays one experiment, not three
    assert resolve(parsed[0])["bands"] == ["NIR_NARROW", "SWIR_1"]


def test_default_selector(monkeypatch) -> None:
    monkeypatch.setenv("RS_EVAL_PERTURB", "default")
    assert perturbations_from_env() == list(DEFAULT_PERTURBATIONS)


def test_empty_means_no_perturbation(monkeypatch) -> None:
    monkeypatch.setenv("RS_EVAL_PERTURB", "")
    assert perturbations_from_env() == []


def test_resolve_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown perturbation"):
        resolve("bogus")
    with pytest.raises(ValueError, match="at least one band"):
        resolve(f"{BAND_DROP}:")


def test_resolve_parses_known_kinds() -> None:
    assert resolve("none")["kind"] == "none"
    assert resolve(f"{RESOLUTION}:4") == {"kind": RESOLUTION, "factor": 4}
    assert resolve(f"{BAND_DROP}:NIR_BROAD") == {"kind": BAND_DROP, "bands": ["NIR_BROAD"]}
