"""CPU-only tests for the RGB+NIR (T11D) distillation track.

Covers the rules that both the manifest builder and the Clay worker depend on:
2019-only frame selection, teacher dequantisation/aggregation, majority labels,
cache fingerprinting plus stale/truncated cache detection, and the four-band
manifest row. No torch, no dataset, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rs_backbone_eval.data.datasets.pastis_hd import (
    build_pastis_hd_manifest,
    select_2019_frames,
)
from rs_backbone_eval.registry import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workers" / "clay" / "src" / "clay_worker" / "rgbn.py"


def load_module():
    spec = importlib.util.spec_from_file_location("clay_rgbn", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_2019_frames_keeps_only_2019_and_caps_the_budget() -> None:
    dates = [20180601, 20181231] + [20190100 + day for day in range(1, 36)] + [20200101]

    frames = select_2019_frames(dates, max_frames=20)

    assert len(frames) == 20
    assert frames == sorted(set(frames))
    assert all(str(dates[index]).startswith("2019") for index in frames)
    assert frames[0] == 2 and frames[-1] == 36


def test_select_2019_frames_keeps_every_frame_below_the_cap() -> None:
    dates = [20190101, 20190201, 20180301]

    assert select_2019_frames(dates, max_frames=20) == [0, 1]


def test_select_2019_frames_refuses_a_patch_without_2019() -> None:
    with pytest.raises(ValueError):
        select_2019_frames([20180901, 20181001])


def test_teacher_decode_matches_the_declared_quantisation() -> None:
    module = load_module()
    raw = np.array([[[127, -128], [0, 64]]], dtype=np.int8)
    valid = np.array([[True, False], [True, True]])

    decoded = module.decode_teacher(raw, valid)

    assert decoded[0, 0, 0] == pytest.approx((127 / 127.5) ** 2)
    assert decoded[0, 0, 1] == 0.0  # NODATA is not a value
    assert decoded[0, 1, 0] == 0.0
    assert decoded[0, 1, 1] == pytest.approx((64 / 127.5) ** 2)


def test_token_aggregation_uses_only_observed_pixels() -> None:
    module = load_module()
    features = np.ones((1, 128, 128), dtype=np.float32)
    valid = np.zeros((128, 128), dtype=bool)
    valid[0, 0] = True  # exactly one observed pixel in the top-left block

    grid, support = module.aggregate_tokens(features, valid)

    assert grid.shape == (1, 16, 16)
    assert support[0, 0] == 1
    assert grid[0, 0, 0] == pytest.approx(1.0)
    assert support[0, 1] == 0
    assert grid[0, 0, 1] == 0.0


def test_majority_labels_aggregates_each_eight_by_eight_block() -> None:
    module = load_module()
    labels = np.zeros((128, 128), dtype=np.int64)
    labels[0:8, 0:8] = 3
    labels[0:2, 0:2] = 19  # void never wins
    labels[0:8, 8:16] = 19  # all-void block keeps the ignore value

    tokens, purity = module.majority_labels(labels)

    assert tokens[0, 0] == 3
    assert tokens[0, 1] == 19
    assert 0.0 < purity < 1.0


def test_cache_fingerprint_changes_with_checkpoint_manifest_and_code() -> None:
    module = load_module()
    base = dict(
        checkpoint_sha256="a" * 64,
        manifest_sha256="b" * 64,
        preprocessing="p",
        code_version="v1",
    )

    canonical = module.cache_fingerprint(**base)
    assert canonical == module.cache_fingerprint(**base)
    assert canonical != module.cache_fingerprint(**{**base, "checkpoint_sha256": "c" * 64})
    assert canonical != module.cache_fingerprint(**{**base, "manifest_sha256": "d" * 64})
    assert canonical != module.cache_fingerprint(**{**base, "code_version": "v2"})


def test_cache_entry_rejects_a_moved_window_or_stale_version(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "10000"
    array = np.zeros((20, 1024, 16, 16), dtype=np.float16)
    module.save_npy_atomic(path.with_suffix(".npy"), array)
    meta = {
        "version": module.CACHE_VERSION,
        "fingerprint": "abc",
        "frames": list(range(20)),
        "dates": list(range(20)),
    }
    module.save_atomic(path.with_suffix(".json"), json.dumps(meta).encode("utf-8"))

    assert module.cache_entry_current(
        path.with_suffix(".json"), {"version": module.CACHE_VERSION, "fingerprint": "abc"}
    )
    assert not module.cache_entry_current(
        path.with_suffix(".json"), {"version": module.CACHE_VERSION, "fingerprint": "xyz"}
    )
    assert not module.cache_entry_current(
        path.with_suffix(".json"), {"version": "old", "fingerprint": "abc"}
    )


def test_cache_entry_rejects_a_truncated_array(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "10000"
    path.with_suffix(".npy").write_bytes(b"\x93NUMPYtruncated")
    meta = {
        "version": module.CACHE_VERSION,
        "fingerprint": "abc",
        "frames": list(range(20)),
        "dates": list(range(20)),
    }
    module.save_atomic(path.with_suffix(".json"), json.dumps(meta).encode("utf-8"))

    assert not module.cache_entry_current(
        path.with_suffix(".json"), {"version": module.CACHE_VERSION, "fingerprint": "abc"}
    )


def test_registry_declares_the_same_inputs_as_the_worker() -> None:
    """The ledger declaration and the worker constants must not drift."""
    module = load_module()
    from rs_backbone_eval.registry import Registry

    registry = Registry(ROOT / "configs")
    task = next(spec for spec in registry.tasks() if spec.task_id == "T11D")
    assert task.input_kind == "raster"
    assert task.bands == module.BAND_NAMES
    assert task.time_steps == module.MAX_FRAMES
    assert task.num_outputs == 20 and task.ignore_index == module.VOID_CLASS

    protocols = {
        spec.protocol_id: spec
        for spec in registry.protocols()
        if spec.protocol_id.startswith("clay_rgbn_")
    }
    transfer = {pid: spec for pid, spec in protocols.items() if pid.startswith("clay_rgbn_t8_")}
    pastis = {pid: spec for pid, spec in protocols.items() if pid not in transfer}
    formal = {pid for pid in pastis if pid.endswith("_v1")}
    assert formal, "the formal variants must exist for the three-seed queue"
    for pid in formal:
        base = pastis[pid[:-3]]
        assert pastis[pid].panel == "primary"
        assert pastis[pid].tasks == base.tasks
        assert pastis[pid].train_backbone is base.train_backbone
        assert pastis[pid].epochs == base.epochs
    declared = {
        pid
        for pid in (
            set(module.DISTILL_PROTOCOLS)
            | set(module.PROBE_PROTOCOLS)
            | set(module.B_STAGE_PROTOCOLS)
        )
        if pid.startswith("clay_rgbn_")
    }
    assert set(pastis) == declared | formal
    for spec in pastis.values():
        assert spec.tasks == ["T11D"]
        assert spec.workers == ["clay"]
        frozen = spec.protocol_id in set(module.DISTILL_PROTOCOLS) | set(module.PROBE_PROTOCOLS)
        assert spec.train_backbone is (not frozen)
    assert set(transfer) == {"clay_rgbn_t8_raw", "clay_rgbn_t8_a64", "clay_rgbn_t8_oe128"}
    for spec in transfer.values():
        assert spec.tasks == ["T8"]
        assert spec.workers == ["clay"]
        assert spec.train_backbone is False
    for protocol_id in module.DISTILL_PROTOCOLS:
        if protocol_id.startswith("clay_rgbn_"):
            assert pastis[protocol_id].panel == "diagnostic"


def test_t11p_registry_matches_the_six_band_profile() -> None:
    from rs_backbone_eval.registry import Registry

    module = load_module()
    registry = Registry(ROOT / "configs")
    task = next(spec for spec in registry.tasks() if spec.task_id == "T11P")
    profile = module.BAND_PROFILES["T11P"]
    assert task.input_kind == "raster"
    assert task.bands == list(profile.bands)
    assert task.time_steps == profile.max_frames == 4
    assert task.num_outputs == 20 and task.ignore_index == module.VOID_CLASS

    six = {
        spec.protocol_id: spec
        for spec in registry.protocols()
        if spec.protocol_id.startswith("clay_s2six_")
    }
    declared = {
        pid
        for pid in (
            set(module.DISTILL_PROTOCOLS)
            | set(module.PROBE_PROTOCOLS)
            | set(module.B_STAGE_PROTOCOLS)
        )
        if pid.startswith("clay_s2six_")
    }
    assert set(six) == declared
    for protocol_id, spec in six.items():
        assert spec.tasks == ["T11P"]
        assert spec.workers == ["clay"]
        frozen = protocol_id in (set(module.DISTILL_PROTOCOLS) | set(module.PROBE_PROTOCOLS))
        assert spec.train_backbone is (not frozen)
    assert six["clay_s2six_distill_o128"].panel == "diagnostic"
    assert all(
        spec.panel == "primary" for pid, spec in six.items() if pid != "clay_s2six_distill_o128"
    )


def _task_spec() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "T11D",
            "application": "test",
            "dataset_id": "PASTIS-HD",
            "family": "segmentation",
            "sensor": "sentinel-2-l2a",
            "input_kind": "raster",
            "time_steps": 20,
            "bands": ["B02", "B03", "B04", "B08"],
            "num_outputs": 20,
            "ignore_index": 19,
            "background_class": 0,
            "primary_metric": {"name": "mean_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/t11d",
        }
    )


def _write_patch(root: Path, patch: int, fold: int, dates: list[int]) -> None:
    payload = root / "derived" / "aef-2019" / "PASTIS-HD"
    (payload / "DATA_S2").mkdir(parents=True, exist_ok=True)
    (payload / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)
    sidecars = root / "derived" / "aef-2019" / "features" / "2019"
    sidecars.mkdir(parents=True, exist_ok=True)
    np.save(payload / "DATA_S2" / f"S2_{patch}.npy", np.zeros((len(dates), 10, 128, 128), np.int16))
    np.save(payload / "ANNOTATIONS" / f"TARGET_{patch}.npy", np.zeros((3, 128, 128), np.uint8))
    np.save(payload / "ANNOTATIONS" / f"ParcelIDs_{patch}.npy", np.zeros((128, 128), np.int32))
    (sidecars / f"{patch}.json").write_text(
        json.dumps(
            {"patch": patch, "fold": fold, "tile": "t30uxv", "dates_s2": dates, "valid_share": 1.0}
        ),
        encoding="utf-8",
    )
    metadata = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ID_PATCH": patch, "Fold": fold, "TILE": "t30uxv"},
                "geometry": None,
            }
        ],
    }
    (payload / "metadata.geojson").write_text(json.dumps(metadata), encoding="utf-8")


def test_rgbn_builder_declares_2019_dates_and_four_bands(tmp_path: Path) -> None:
    dates = [20180924, 20190117, 20190216, 20190313, 20190412]
    _write_patch(tmp_path, 10000, 1, dates)

    rows = build_pastis_hd_manifest(tmp_path, _task_spec())

    assert len(rows) == 1
    row = rows[0]
    assert row.bands == ["B02", "B03", "B04", "B08"]
    assert row.sensor == "sentinel-2-l2a"
    assert row.split == "train"
    assert row.timestamps == ["20190117", "20190216", "20190313", "20190412"]
    assert len(row.temporal_mask) == len(row.timestamps)
    assert row.aux_paths["frame_rule"] == "2019-only-linspace"
    assert row.image_paths[0].endswith("S2_10000.npy")
    assert len(row.checksum) == 64


def _task_spec_six() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "T11P",
            "application": "test",
            "dataset_id": "PASTIS-HD",
            "family": "segmentation",
            "sensor": "sentinel-2-l2a",
            "input_kind": "raster",
            "time_steps": 4,
            "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
            "num_outputs": 20,
            "ignore_index": 19,
            "background_class": 0,
            "primary_metric": {"name": "mean_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/t11p",
        }
    )


def test_t11p_builder_declares_four_frames_and_six_bands(tmp_path: Path) -> None:
    dates = [20180924] + [20190100 + index for index in range(1, 13)]
    _write_patch(tmp_path, 10000, 1, dates)

    rows = build_pastis_hd_manifest(tmp_path, _task_spec_six())

    assert len(rows) == 1
    row = rows[0]
    assert row.bands == ["B02", "B03", "B04", "B8A", "B11", "B12"]
    assert len(row.timestamps) == 4
    assert row.aux_paths["max_frames"] == "4"
    assert row.aux_paths["s2_columns"] == "0,1,2,7,8,9"
    assert row.timestamps == ["20190101", "20190105", "20190108", "20190112"]


def test_t11p_builder_rejects_a_mismatched_time_budget(tmp_path: Path) -> None:
    dates = [20180924] + [20190100 + index for index in range(1, 13)]
    _write_patch(tmp_path, 10000, 1, dates)
    spec = _task_spec_six().model_copy(update={"time_steps": 20})

    with pytest.raises(Exception, match="time_steps"):
        build_pastis_hd_manifest(tmp_path, spec)


def test_six_band_profile_reads_four_frames_and_six_columns(tmp_path: Path) -> None:
    module = load_module()
    dates = [20180924] + [20190100 + index for index in range(1, 13)]
    stack = np.arange(len(dates) * 10 * 4 * 4, dtype=np.int32).reshape(len(dates), 10, 4, 4)
    stack = stack.astype(np.int16)
    np.save(tmp_path / "S2_10000.npy", stack)
    (tmp_path / "10000.json").write_text(json.dumps({"dates_s2": dates}), encoding="utf-8")
    row = {
        "sample_id": "10000",
        "aux_paths": {"s2_path": "S2_10000.npy", "dates_source": "10000.json"},
    }
    profile = module.BAND_PROFILES["T11P"]

    pixels, used_dates, frames = module.read_frames(row, tmp_path, profile)

    assert pixels.shape == (4, 6, 4, 4)
    assert frames == module.select_frames(dates, 4)
    assert used_dates == [int(dates[index]) for index in frames]
    expected = np.asarray(stack[frames][:, [0, 1, 2, 7, 8, 9]], dtype=np.float32)
    assert np.array_equal(pixels, expected)


def test_profile_for_rejects_an_unregistered_task() -> None:
    module = load_module()

    with pytest.raises(module.RgbnFailure, match="T11D"):
        module.profile_for("T99")
