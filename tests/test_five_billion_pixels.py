"""Five-Billion-Pixels (T8) manifest: GSD, test list and footprint groups.

Synthetic scene tree under ``tmp_path`` (tiny files, no real data). The real
release had every scene in one ``GF2`` group and a 0.8 m GSD; the manifest now
groups scenes by RPC footprint overlap so a test scene that revisits a training
footprint is visible as a shared group, and records the 4 m MSS GSD.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets import five_billion_pixels as fbp
from rs_backbone_eval.data.datasets.sen1floods11 import ManifestBuildError
from rs_backbone_eval.registry import TaskSpec

README = """header
* Test images for baseline (Table 4 in our article):
SCENE_C.tif
"""


def rpb(lat: float, lon: float, scale: float = 0.15) -> str:
    return f"latOffset = {lat};\nlongOffset = {lon};\nlatScale = {scale};\nlongScale = {scale};\n"


def make_tree(root: Path, scenes: dict[str, tuple[float, float]]) -> None:
    for name in ("Image_16bit_BGRNir", "Annotation__index", "Coordinate_files"):
        (root / name).mkdir(parents=True)
    (root / "readme (important).txt").write_text(README, encoding="utf-8")
    for name, (lat, lon) in scenes.items():
        (root / "Image_16bit_BGRNir" / f"{name}.tiff").write_bytes(name.encode())
        (root / "Annotation__index" / f"{name}_24label.png").write_bytes(b"x")
        (root / "Coordinate_files" / f"{name}.rpb").write_text(rpb(lat, lon), encoding="utf-8")


def task_spec() -> TaskSpec:
    return TaskSpec(
        task_id="T8",
        application="Land cover",
        dataset_id="Five-Billion-Pixels",
        family="segmentation",
        sensor="gf-2",
        time_steps=1,
        bands=["BLUE", "GREEN", "RED", "NIR_BROAD"],
        num_outputs=25,
        ignore_index=-1,
        primary_metric={"name": "mean_iou", "direction": "maximize"},
        official_config_url="https://example.org",
        allowed_splits=["train", "validation", "test"],
    )


def test_groups_follow_footprint_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A and C share a footprint (C is the test scene revisiting A); B is apart.
    make_tree(
        tmp_path, {"SCENE_A": (30.0, 110.0), "SCENE_B": (40.0, 120.0), "SCENE_C": (30.05, 110.05)}
    )
    monkeypatch.setenv("RS_EVAL_FBP_ROOT", str(tmp_path))
    samples = {
        s.sample_id: s for s in fbp.build_five_billion_pixels_manifest(tmp_path, task_spec())
    }

    assert samples["SCENE_C"].split == "test"
    assert samples["SCENE_A"].group_id == samples["SCENE_C"].group_id
    assert samples["SCENE_B"].group_id != samples["SCENE_A"].group_id
    assert {s.gsd_m for s in samples.values()} == {4.0}


def test_validation_never_shares_a_footprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = {f"SCENE_{i:02d}": (20.0 + i, 100.0) for i in range(20)}
    scenes["SCENE_C"] = (20.05, 100.05)  # the test scene revisits SCENE_00
    scenes["SCENE_01B"] = (21.05, 100.05)  # two dates of one training footprint
    make_tree(tmp_path, scenes)
    monkeypatch.setenv("RS_EVAL_FBP_ROOT", str(tmp_path))
    samples = list(fbp.build_five_billion_pixels_manifest(tmp_path, task_spec()))

    splits_by_group: dict[str, set[str]] = {}
    for sample in samples:
        splits_by_group.setdefault(sample.group_id, set()).add(sample.split)
    validation_groups = [g for g, splits in splits_by_group.items() if "validation" in splits]
    assert validation_groups
    assert all(splits_by_group[g] == {"validation"} for g in validation_groups)


def test_missing_rpc_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_tree(tmp_path, {"SCENE_A": (30.0, 110.0), "SCENE_C": (31.0, 111.0)})
    (tmp_path / "Coordinate_files" / "SCENE_A.rpb").unlink()
    monkeypatch.setenv("RS_EVAL_FBP_ROOT", str(tmp_path))
    with pytest.raises(ManifestBuildError, match="RPC"):
        fbp.build_five_billion_pixels_manifest(tmp_path, task_spec())


def test_overlap_fraction_uses_the_smaller_box() -> None:
    big = (0.0, 2.0, 0.0, 2.0)
    small = (0.0, 1.0, 0.0, 1.0)
    assert fbp.overlap_fraction(big, small) == pytest.approx(1.0)
    assert fbp.overlap_fraction(big, (5.0, 6.0, 5.0, 6.0)) == 0.0
