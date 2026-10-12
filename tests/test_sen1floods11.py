import json
from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets import builder_for
from rs_backbone_eval.data.datasets.sen1floods11 import (
    ManifestBuildError,
    build_sen1floods11_manifest,
    file_sha256,
    load_events,
)
from rs_backbone_eval.registry import TaskSpec

VERSION = Path("raw/sen1floods11/v1.1")


@pytest.fixture
def t1_task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "T1",
            "application": "Flood Detection",
            "dataset_id": "Sen1Floods11",
            "family": "segmentation",
            "sensor": "sentinel-2",
            "time_steps": 1,
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "num_outputs": 2,
            "ignore_index": -1,
            "primary_metric": {"name": "water_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/t1",
        }
    )


def write_dataset(data_root: Path, *, chips: dict[str, str], drop_s2: str | None = None) -> None:
    events = [
        {"ID": 2, "location": "Ghana", "s2_date": "2018/09/19"},
        {"ID": 4, "location": "Cambodia", "s2_date": "2018/08/04"},
        {"ID": 1, "location": "Bolivia", "s2_date": "2018/02/15"},
    ]
    (data_root / VERSION).mkdir(parents=True)
    (data_root / VERSION / "Sen1Floods11_Metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"properties": e} for e in events]}),
        encoding="utf-8",
    )
    splits = data_root / VERSION / "splits" / "flood_handlabeled"
    splits.mkdir(parents=True)
    split_files = {
        "flood_train_data.csv": ["Ghana_1_S1Hand.tif,Ghana_1_LabelHand.tif"],
        "flood_valid_data.csv": ["Mekong_7_S1Hand.tif,Mekong_7_LabelHand.tif"],
        "flood_test_data.csv": ["Ghana_9_S1Hand.tif,Ghana_9_LabelHand.tif"],
        "flood_bolivia_data.csv": ["Bolivia_3_S1Hand.tif,Bolivia_3_LabelHand.tif"],
    }
    for name, lines in split_files.items():
        (splits / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    hand = data_root / VERSION / "data" / "flood_events" / "HandLabeled"
    for chip in chips.values():
        for folder, suffix in (("S2Hand", "S2Hand"), ("LabelHand", "LabelHand")):
            path = hand / folder / f"{chip}_{suffix}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(chip.encode() + b"-pixels")
    if drop_s2:
        (hand / "S2Hand" / f"{drop_s2}_S2Hand.tif").unlink()


def test_builder_for_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="no manifest builder"):
        builder_for("UnknownDataset")


def test_events_include_alias_locations(tmp_path: Path) -> None:
    write_dataset(tmp_path, chips={})
    events = load_events(tmp_path)

    assert events["Mekong"].location == "Cambodia"
    assert events["Mekong"].s2_date == "2018-08-04"
    assert events["Ghana"].event_id == 2


def test_builds_manifest_from_official_splits(tmp_path: Path, t1_task: TaskSpec) -> None:
    write_dataset(
        tmp_path,
        chips={"train": "Ghana_1", "valid": "Mekong_7", "test": "Ghana_9", "bolivia": "Bolivia_3"},
    )

    samples = build_sen1floods11_manifest(tmp_path, t1_task)

    assert len(samples) == 4
    by_id = {sample.sample_id: sample for sample in samples}
    assert by_id["Ghana_1"].split == "train"
    assert by_id["Mekong_7"].split == "validation"
    assert by_id["Ghana_9"].split == "test"
    assert by_id["Bolivia_3"].split == "bolivia_holdout"

    mekong = by_id["Mekong_7"]
    assert mekong.group_id == "Cambodia"
    assert mekong.timestamps == ["2018-08-04"]
    assert mekong.sensor == t1_task.sensor
    assert mekong.bands == t1_task.bands
    assert mekong.temporal_mask == [True]
    assert mekong.gsd_m == pytest.approx(10.0)
    assert mekong.image_paths == [
        "raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/Mekong_7_S2Hand.tif"
    ]
    assert mekong.label_path == (
        "raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Mekong_7_LabelHand.tif"
    )
    assert mekong.checksum == file_sha256(
        tmp_path
        / VERSION
        / "data"
        / "flood_events"
        / "HandLabeled"
        / "S2Hand"
        / "Mekong_7_S2Hand.tif"
    )


def test_same_event_spans_splits_and_is_audited(tmp_path: Path, t1_task: TaskSpec) -> None:
    write_dataset(
        tmp_path,
        chips={"train": "Ghana_1", "valid": "Mekong_7", "test": "Ghana_9", "bolivia": "Bolivia_3"},
    )

    samples = build_sen1floods11_manifest(tmp_path, t1_task)

    groups = {sample.group_id for sample in samples if sample.sample_id.startswith("Ghana")}
    assert groups == {"Ghana"}
    splits = sorted(sample.split for sample in samples if sample.sample_id.startswith("Ghana"))
    assert splits == ["test", "train"]


def test_missing_chip_file_is_rejected(tmp_path: Path, t1_task: TaskSpec) -> None:
    write_dataset(
        tmp_path,
        chips={"train": "Ghana_1", "valid": "Mekong_7", "test": "Ghana_9", "bolivia": "Bolivia_3"},
        drop_s2="Ghana_9",
    )

    with pytest.raises(ManifestBuildError, match="missing S2Hand/LabelHand files"):
        build_sen1floods11_manifest(tmp_path, t1_task)


def test_unknown_location_is_rejected(tmp_path: Path, t1_task: TaskSpec) -> None:
    write_dataset(
        tmp_path,
        chips={"train": "Ghana_1", "valid": "Mekong_7", "test": "Ghana_9", "bolivia": "Bolivia_3"},
    )
    splits = tmp_path / VERSION / "splits" / "flood_handlabeled"
    (splits / "flood_train_data.csv").write_text(
        "Atlantis_5_S1Hand.tif,Atlantis_5_LabelHand.tif\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError, match="event metadata"):
        build_sen1floods11_manifest(tmp_path, t1_task)
