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


def test_metadata_perturbations_parse_describe_and_apply(monkeypatch) -> None:
    import pytest

    from rs_backbone_eval.perturb import (
        PLACEHOLDER_DATE,
        applies,
        describe,
        perturbations_from_env,
        resolve,
    )

    monkeypatch.setenv("RS_EVAL_PERTURB", "default;metadata;meta_gsd:2")
    names = perturbations_from_env()
    assert names[:4] == [
        "band_drop:NIR_NARROW,SWIR_1,SWIR_2",
        "band_drop:SWIR_1,SWIR_2",
        "resolution:2",
        "resolution:4",
    ]
    assert names[4:] == ["meta_date:const", "meta_gsd:2", "meta_gsd:0.5"]  # de-duplicated

    date = resolve("meta_date:const")
    assert date == {"kind": "meta_date", "date": PLACEHOLDER_DATE}
    assert describe("meta_date:const") == "test_meta_date-const"
    assert describe("meta_gsd:0.5") == "test_meta_gsdx0.5"
    assert describe("meta_gsd:2") == "test_meta_gsdx2"

    # A model only receives the metadata perturbations it can act on.
    assert applies(resolve("meta_gsd:2"), {"gsd"})
    assert not applies(resolve("meta_gsd:2"), {"date"})
    assert not applies(date, set())
    assert applies(resolve("band_drop:SWIR_1"), set())
    with pytest.raises(ValueError):
        resolve("meta_gsd:1")
    with pytest.raises(ValueError):
        resolve("meta_date:shift")


def test_metadata_splits_count_as_perturbations_but_holdouts_do_not() -> None:
    from rs_backbone_eval.matrix import is_holdout_split, is_perturbation_split

    assert is_perturbation_split("test_meta_gsdx2")
    assert is_perturbation_split("test_meta_date-const")
    assert not is_perturbation_split("bolivia_holdout")
    assert is_holdout_split("bolivia_holdout")
