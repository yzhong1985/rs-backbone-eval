"""Runner-level tests for the AEF worker (torch skipped where absent, like the
LIM/visualisation tests). The fixtures build a synthetic contract manifest:
2,433 real patches are far beyond a unit test, but the decode gate, the audit
and the run lifecycle are exorcised end to end on 16x16 grids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")

# importorskip returns the MODULE; the runner's ``main`` sits one level deeper
aef_worker_mod = pytest.importorskip("aef_worker.runner")
runner_main = aef_worker_mod.main  # noqa: E402

from rs_backbone_eval.contracts import ManifestSample  # noqa: E402
from rs_backbone_eval.data.manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"AEF_{index:02d}" for index in range(64)]
GRID = 16


def make_sample(
    sample_id: str, group_id: str, split: str, tmp_path: Path, tile: str
) -> ManifestSample:
    features = np.full((64, GRID, GRID), 40, dtype=np.float16)
    valid = np.ones((GRID, GRID), dtype=bool)
    voxel = tmp_path / "features" / f"{sample_id}.npz"
    voxel.parent.mkdir(parents=True, exist_ok=True)
    np.savez(voxel, features=features, valid=valid)
    labels = np.zeros((1, GRID, GRID), dtype=np.uint8)
    labels[0, : GRID // 2] = 2
    label = tmp_path / "labels" / f"{sample_id}.npy"
    label.parent.mkdir(parents=True, exist_ok=True)
    np.save(label, labels)
    return ManifestSample(
        task_id="T11",
        sample_id=sample_id,
        group_id=group_id,
        split=split,
        image_paths=[voxel.relative_to(tmp_path).as_posix()],
        label_path=label.relative_to(tmp_path).as_posix(),
        aux_paths={"tile": tile, "encoding": "raw_aef_int8_as_float16", "label_year": "2019"},
        sensor="aef",
        bands=list(BANDS),
        timestamps=["2019"],
        temporal_mask=[True],
        gsd_m=10.0,
        checksum=hashlib.sha256(voxel.read_bytes()).hexdigest(),
        window=[0, 0, GRID, GRID],
    )


def write_specs(tmp_path: Path) -> None:
    (tmp_path / "specs").mkdir()
    for group, name in (
        ("tasks", "T11_pastis_hd"),
        ("models", "aef_annual"),
        ("protocols", "aef_probe_mlp"),
    ):
        payload = yaml.safe_load((ROOT / "configs" / group / f"{name}.yaml").read_text())
        (tmp_path / "specs" / f"{name}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def make_request(
    tmp_path: Path, manifest: Path, runs_root: Path, action: str
) -> dict[str, str | None]:
    return {
        "schema_version": "1.0",
        "action": action,
        "run_id": "aef-smoke",
        "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
        "model_config_path": (tmp_path / "specs" / "aef_annual.yaml").as_posix(),
        "protocol_config_path": (tmp_path / "specs" / "aef_probe_mlp.yaml").as_posix(),
        "manifest_path": manifest.as_posix(),
        "data_root": tmp_path.as_posix(),
        "weights_root": (tmp_path / "weights").as_posix(),
        "cache_root": (tmp_path / "cache").as_posix(),
        "runs_root": runs_root.as_posix(),
        "seed": 17,
        "resume_from": None,
    }


def test_probe_builds_a_completed_probe_run(tmp_path) -> None:
    write_specs(tmp_path)
    rows = [
        make_sample("1", "1", "train", tmp_path, "tile-a"),
        make_sample("2", "2", "validation", tmp_path, "tile-b"),
        make_sample("3", "3", "test", tmp_path, "tile-c"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(make_request(tmp_path, manifest, runs_root, "probe")), encoding="utf-8"
    )

    assert runner_main([request.as_posix()]) == 0

    record = json.loads((runs_root / "aef-smoke" / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["task_id"] == "T11"
    assert record["protocol_id"] == "aef_probe_mlp"
    assert "probe ok" in record["notes"]


def test_shrunk_fold_fails_closed(tmp_path) -> None:
    write_specs(tmp_path)
    rows = [
        make_sample("1", "1", "train", tmp_path, "tile-a"),
        make_sample("2", "2", "validation", tmp_path, "tile-b"),
        make_sample("3", "3", "test", tmp_path, "tile-c"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(make_request(tmp_path, manifest, runs_root, "train")), encoding="utf-8"
    )

    assert runner_main([request.as_posix()]) == 1

    record = json.loads((runs_root / "aef-smoke" / "run.json").read_text())
    assert record["status"] == "failed"
    assert "rows" in (record["notes"] or "")


def test_wrong_encoding_fails_closed(tmp_path) -> None:
    write_specs(tmp_path)
    rows = [
        make_sample("1", "1", "train", tmp_path, "tile-a"),
        make_sample("2", "2", "validation", tmp_path, "tile-b"),
        make_sample("3", "3", "test", tmp_path, "tile-c"),
    ]
    # The probe decodes the manifest's first row; the canonical sort puts the
    # smallest sample_id there, so label that row.
    rows[0].aux_paths["encoding"] = "float32_plain"
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(make_request(tmp_path, manifest, runs_root, "probe")), encoding="utf-8"
    )

    assert runner_main([request.as_posix()]) == 1

    record = json.loads((runs_root / "aef-smoke" / "run.json").read_text())
    assert record["status"] == "failed"
    assert "encoding" in (record["notes"] or "")


def test_train_runs_two_tiles_one_epoch(tmp_path, monkeypatch) -> None:
    """Per round-6 review: train() had no test coverage; 2 tiles x 1 epoch."""
    import torch  # noqa: F401

    write_specs(tmp_path)
    rows = [
        make_sample("1", "1", "train", tmp_path, "tile-a"),
        make_sample("2", "2", "train", tmp_path, "tile-b"),
        make_sample("3", "3", "validation", tmp_path, "tile-a"),
        make_sample("4", "4", "test", tmp_path, "tile-b"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(make_request(tmp_path, manifest, runs_root, "train")), encoding="utf-8"
    )
    monkeypatch.setattr("aef_worker.runner._train_min", lambda: 2)
    monkeypatch.setattr("aef_worker.runner._train_budget", lambda: 1)
    # the tiny fold has one test row; pin the official-count anchor to it
    monkeypatch.setattr("aef_worker.runner.OFFICIAL_TEST_SAMPLES", {"T11": 1})

    assert runner_main([request_path.as_posix()]) == 0

    record = json.loads((runs_root / "aef-smoke" / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["epochs_completed"] == 1
    assert record["training"]["optimizer_steps"] == 1
    assert record["tags"]["test_samples_evaluated"] == "1"
    assert record["tags"]["test_expected_samples"] == "1"
    assert record["tags"]["test_manifest_samples"] == "1"
    primary = next(metric for metric in record["metrics"] if metric["is_primary"])
    assert primary["name"] == "mean_iou" and primary["split"] == "test"
    assert primary["value"] >= 0.0


def test_train_min_gate_blocks_default() -> None:
    """Without the env knobs the full-fold requirement stays intact."""
    import torch  # noqa: F401
    from aef_worker.runner import _train_min

    assert _train_min() == 1400


def test_train_refuses_a_shrunken_test_manifest(tmp_path, monkeypatch) -> None:
    """Round-5 reviewer 2: AEF was the remaining admission hole -- a full
    training budget with a one-row test fold must fail, not complete."""
    write_specs(tmp_path)
    rows = [
        make_sample("1", "1", "train", tmp_path, "tile-a"),
        make_sample("2", "2", "train", tmp_path, "tile-b"),
        make_sample("3", "3", "validation", tmp_path, "tile-a"),
        make_sample("4", "4", "test", tmp_path, "tile-b"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(make_request(tmp_path, manifest, runs_root, "train")), encoding="utf-8"
    )
    monkeypatch.setattr("aef_worker.runner._train_min", lambda: 2)
    monkeypatch.setattr("aef_worker.runner._train_budget", lambda: 1)
    # the official T11 fold has 496 rows; this manifest has 1 -> must fail
    monkeypatch.setattr("aef_worker.runner.OFFICIAL_TEST_SAMPLES", {"T11": 496})

    assert runner_main([request_path.as_posix()]) == 1
    record = json.loads((runs_root / "aef-smoke" / "run.json").read_text())
    assert record["status"] == "failed"
    assert "official" in (record["notes"] or "")
