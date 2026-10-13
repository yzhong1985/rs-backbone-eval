from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets.hls_burn_scars import (
    ManifestBuildError,
    build_hls_burn_scars_manifest,
    parse_stem,
)
from rs_backbone_eval.registry import TaskSpec

BASE = Path("raw/hls_burn_scars")
T2_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]


@pytest.fixture
def t2_task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "T2",
            "application": "Wildfire Scar Detection",
            "dataset_id": "hls_burn_scars",
            "family": "segmentation",
            "sensor": "hls",
            "time_steps": 1,
            "bands": T2_BANDS,
            "num_outputs": 2,
            "ignore_index": -1,
            "primary_metric": {"name": "burn_scar_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/t2",
        }
    )


def write_chip(directory: Path, stem: str, payload: bytes = b"tif") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}_merged.tif").write_bytes(payload + b"-image")
    (directory / f"{stem}.mask.tif").write_bytes(payload + b"-mask")


def test_parse_stem_extracts_tile_and_date() -> None:
    system, tile, date = parse_stem("subsetted_512x512_HLS.S30.T14SNB.2018215.v1.4")

    assert system == "HLS.S30"
    assert tile == "T14SNB"
    assert date == "2018-08-03"


def test_parse_stem_rejects_unparseable_names() -> None:
    with pytest.raises(ManifestBuildError, match="cannot parse"):
        parse_stem("nonsense_name")


def test_builder_splits_official_training_and_locks_official_validation(
    tmp_path: Path, t2_task: TaskSpec
) -> None:
    training = tmp_path / BASE / "training"
    validation = tmp_path / BASE / "validation"
    for index in range(20):
        write_chip(training, f"subsetted_512x512_HLS.S30.T10SEH.{2018200 + index}.v1.4")
    write_chip(validation, "subsetted_512x512_HLS.L30.T11SKV.2019152.v1.4")

    samples = build_hls_burn_scars_manifest(tmp_path, t2_task)

    splits = {}
    for sample in samples:
        splits.setdefault(sample.split, []).append(sample)
    assert len(samples) == 21
    # 10% of the 20 official training chips become the project validation.
    assert len(splits["validation"]) == 2
    assert len(splits["train"]) == 18
    assert len(splits["official_validation"]) == 1

    official = splits["official_validation"][0]
    assert official.group_id == "T11SKV"
    assert official.timestamps == ["2019-06-01"]
    assert official.sensor == "hls"
    assert official.bands == T2_BANDS
    assert official.gsd_m == pytest.approx(30.0)
    assert official.image_paths == [
        "raw/hls_burn_scars/validation/subsetted_512x512_HLS.L30.T11SKV.2019152.v1.4_merged.tif"
    ]
    assert official.label_path.endswith(".v1.4.mask.tif")


def test_builder_is_deterministic(tmp_path: Path, t2_task: TaskSpec) -> None:
    training = tmp_path / BASE / "training"
    for index in range(20):
        write_chip(training, f"subsetted_512x512_HLS.S30.T10SEH.{2018200 + index}.v1.4")
    write_chip(tmp_path / BASE / "validation", "subsetted_512x512_HLS.L30.T11SKV.2019152.v1.4")

    first = build_hls_burn_scars_manifest(tmp_path, t2_task)
    second = build_hls_burn_scars_manifest(tmp_path, t2_task)

    assert {s.sample_id: s.split for s in first} == {s.sample_id: s.split for s in second}


def test_builder_rejects_missing_mask(tmp_path: Path, t2_task: TaskSpec) -> None:
    training = tmp_path / BASE / "training"
    write_chip(training, "subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4")
    (training / "subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4.mask.tif").unlink()
    write_chip(tmp_path / BASE / "validation", "subsetted_512x512_HLS.L30.T11SKV.2019152.v1.4")

    with pytest.raises(ManifestBuildError, match="missing a mask"):
        build_hls_burn_scars_manifest(tmp_path, t2_task)
