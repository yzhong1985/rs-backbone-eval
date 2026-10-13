from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets.multicrop import (
    ManifestBuildError,
    build_multicrop_manifest,
    load_chip_metadata,
)
from rs_backbone_eval.registry import TaskSpec

BASE = Path("raw/multi-temporal-crop-classification")
T5_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


@pytest.fixture
def t5_task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "T5",
            "application": "Multi-temporal Crop Segmentation (US)",
            "dataset_id": "multi-temporal-crop-classification",
            "family": "segmentation",
            "sensor": "hls",
            "time_steps": 3,
            "bands": T5_BANDS,
            "num_outputs": 13,
            "ignore_index": -1,
            "primary_metric": {"name": "mean_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/t5",
        }
    )


def write_chip(directory: Path, chip: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{chip}_merged.tif").write_bytes(b"18-band-stack")
    (directory / f"{chip}.mask.tif").write_bytes(b"labels")


def write_metadata(tmp_path: Path, chips: list[str]) -> None:
    main = tmp_path / BASE / "main"
    main.mkdir(parents=True, exist_ok=True)
    rows = ["chip_id,tile,first_img_date,middle_img_date,last_img_date"]
    for index, chip in enumerate(chips):
        rows.append(
            f"{chip},13TDE,2022-04-{10 + index:02d}T18:02:50.356Z,"
            f"2022-06-{10 + index:02d}T18:03:06.327Z,2022-09-{10 + index:02d}T18:03:02.472Z"
        )
    (main / "chips_df.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_metadata_loads_dates(tmp_path: Path) -> None:
    write_metadata(tmp_path, ["chip_000_001"])

    metadata = load_chip_metadata(tmp_path / BASE)

    assert metadata["chip_000_001"]["tile"] == "13TDE"
    assert metadata["chip_000_001"]["dates"] == ["2022-04-10", "2022-06-10", "2022-09-10"]


def test_builder_binds_three_dates_to_one_stacked_raster(tmp_path: Path, t5_task: TaskSpec) -> None:
    chips = [f"chip_000_{index:03d}" for index in range(20)]
    for chip in chips:
        write_chip(tmp_path / BASE / "training_chips", chip)
    write_chip(tmp_path / BASE / "validation_chips", "chip_999_999")
    write_metadata(tmp_path, [*chips, "chip_999_999"])

    samples = build_multicrop_manifest(tmp_path, t5_task)

    splits = {}
    for sample in samples:
        splits.setdefault(sample.split, []).append(sample)
    assert len(samples) == 21
    assert len(splits["validation"]) == 2  # 10% of the 20 official training chips
    assert len(splits["train"]) == 18
    assert len(splits["official_validation"]) == 1

    official = splits["official_validation"][0]
    assert official.sample_id == "chip_999_999"
    assert official.group_id == "13TDE"
    assert len(official.image_paths) == 1  # stacked raster
    assert len(official.timestamps) == 3
    assert official.temporal_mask == [True, True, True]
    assert official.time_count == 3
    assert official.gsd_m == pytest.approx(30.0)
    assert official.bands == T5_BANDS


def test_builder_ignores_macos_resource_forks(tmp_path: Path, t5_task: TaskSpec) -> None:
    write_chip(tmp_path / BASE / "training_chips", "chip_000_001")
    write_chip(tmp_path / BASE / "validation_chips", "chip_999_999")
    (tmp_path / BASE / "training_chips" / "._chip_000_001_merged.tif").write_bytes(b"junk")
    write_metadata(tmp_path, ["chip_000_001", "chip_999_999"])

    samples = build_multicrop_manifest(tmp_path, t5_task)

    assert {sample.sample_id for sample in samples} == {"chip_000_001", "chip_999_999"}


def test_builder_rejects_chips_without_dates(tmp_path: Path, t5_task: TaskSpec) -> None:
    write_chip(tmp_path / BASE / "training_chips", "chip_000_001")
    write_chip(tmp_path / BASE / "validation_chips", "chip_999_999")
    write_metadata(tmp_path, ["chip_000_001"])  # missing the validation chip

    with pytest.raises(ManifestBuildError, match="lack date metadata"):
        build_multicrop_manifest(tmp_path, t5_task)
