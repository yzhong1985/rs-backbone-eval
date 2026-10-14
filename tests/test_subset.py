"""Tests for group-aware manifest subsetting (few-shot and holdout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rs_backbone_eval.contracts import ManifestSample
from rs_backbone_eval.data.manifest import write_manifest
from rs_backbone_eval.data.subset import (
    few_shot_subset,
    geographic_holdout,
    groups_in_split,
    parse_cutoff,
    read_manifest,
    temporal_holdout,
)
from rs_backbone_eval.splits import evaluation_split, evaluation_splits


def sample(sample_id: str, group_id: str, split: str) -> ManifestSample:
    return ManifestSample(
        task_id="T1",
        sample_id=sample_id,
        group_id=group_id,
        split=split,
        image_paths=[f"raw/{split}/{sample_id}.tif"],
        label_path=f"raw/{split}/{sample_id}.mask.tif",
        sensor="sentinel-2",
        bands=["BLUE", "GREEN"],
        timestamps=["2020-01-01"],
        temporal_mask=[True],
        gsd_m=10.0,
        checksum="a" * 64,
    )


def manifest() -> list[ManifestSample]:
    samples = [sample(f"t{i}", f"g{i}", "train") for i in range(10)]
    samples += [sample(f"v{i}", f"vg{i}", "validation") for i in range(2)]
    samples += [sample(f"e{i}", f"eg{i}", "test") for i in range(2)]
    return samples


def test_few_shot_keeps_val_test_and_reduces_train_groups() -> None:
    derived = few_shot_subset(manifest(), 0.3, seed=17)
    assert len(groups_in_split(derived, "train")) == 3
    assert len(groups_in_split(derived, "validation")) == 2
    assert len(groups_in_split(derived, "test")) == 2
    # a group never straddles splits
    assert all(
        len({s.split for s in derived if s.group_id == group}) == 1
        for group in {s.group_id for s in derived}
    )


def test_few_shot_is_deterministic() -> None:
    first = few_shot_subset(manifest(), 0.5, seed=17)
    second = few_shot_subset(manifest(), 0.5, seed=17)
    assert {s.sample_id for s in first} == {s.sample_id for s in second}


def test_geographic_holdout_moves_groups_to_a_new_split() -> None:
    derived = geographic_holdout(manifest(), 0.4, seed=17, holdout_split="geographic_holdout")
    assert len(groups_in_split(derived, "geographic_holdout")) == 4
    assert len(groups_in_split(derived, "train")) == 6
    assert len(groups_in_split(derived, "test")) == 2


def test_read_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "T1.jsonl"
    write_manifest(path, manifest())
    loaded = read_manifest(path)
    assert {s.sample_id for s in loaded} == {s.sample_id for s in manifest()}


def test_few_shot_sample_unit_keeps_partial_groups() -> None:
    samples = [sample(f"a{i}", "shared", "train") for i in range(4)]
    samples.append(sample("b0", "other", "train"))
    by_sample = few_shot_subset(samples, 0.4, seed=17, unit="sample")
    assert sum(1 for s in by_sample if s.split == "train") == 2
    by_group = few_shot_subset(samples, 0.5, seed=17, unit="group")
    kept = {s.group_id for s in by_group if s.split == "train"}
    assert len(kept) == 1  # a whole group is kept or dropped, never split


def dated(sample_id: str, group_id: str, split: str, stamp: str) -> ManifestSample:
    base = sample(sample_id, group_id, split)
    return base.model_copy(update={"timestamps": [stamp], "temporal_mask": [True]})


def test_geographic_holdout_rejects_a_group_that_straddles_splits() -> None:
    samples = [sample("t0", "g0", "train"), sample("e0", "g0", "test")]
    samples += [sample(f"t{i}", f"g{i}", "train") for i in range(1, 5)]
    with pytest.raises(ValueError, match="isolate=True"):
        geographic_holdout(samples, 0.5, seed=17)


def test_geographic_holdout_isolate_moves_every_row_of_a_group() -> None:
    samples = [sample("t0", "g0", "train"), sample("e0", "g0", "test")]
    samples += [sample(f"t{i}", f"g{i}", "train") for i in range(1, 5)]
    derived = geographic_holdout(samples, 0.5, seed=17, isolate=True)
    held = {s.group_id for s in derived if s.split == "geographic_holdout"}
    assert "g0" in held
    assert not [s for s in derived if s.group_id == "g0" and s.split != "geographic_holdout"]


def test_temporal_holdout_moves_post_cutoff_groups_only() -> None:
    samples = [dated("a0", "g_old", "train", "2017-05-01")]
    samples += [dated("b0", "g_new", "train", "2019-05-01")]
    samples += [dated("v0", "g_val", "validation", "2017-06-01")]
    derived = temporal_holdout(samples, "2018")
    assert [s.sample_id for s in derived if s.split == "temporal_holdout"] == ["b0"]
    assert {s.sample_id for s in derived if s.split == "train"} == {"a0"}
    assert {s.sample_id for s in derived if s.split == "validation"} == {"v0"}


def test_temporal_holdout_moves_a_group_that_spans_the_cutoff() -> None:
    samples = [dated("old", "g_span", "train", "2017-01-01")]
    samples += [dated("new", "g_span", "train", "2019-01-01")]
    derived = temporal_holdout(samples, "2018-01-01")
    moved = {s.sample_id for s in derived if s.split == "temporal_holdout"}
    assert moved == {"old", "new"}  # no post-cutoff row stays in train


def test_parse_cutoff_accepts_year_and_iso_date() -> None:
    assert parse_cutoff("2018") == parse_cutoff("2018-01-01")
    assert parse_cutoff(2018) == parse_cutoff("2018-01-01")
    with pytest.raises(ValueError):
        parse_cutoff(42)


def test_evaluation_split_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RS_EVAL_EVAL_SPLIT", raising=False)
    assert evaluation_split("test") == "test"
    monkeypatch.setenv("RS_EVAL_EVAL_SPLIT", "geographic_holdout")
    assert evaluation_split("test") == "geographic_holdout"
    assert evaluation_splits("test") == ["geographic_holdout"]
