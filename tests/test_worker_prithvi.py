"""Prithvi worker runner tests: run-record lifecycle and probe failure contract.

The runner is executed with the current interpreter as a plain script; the
root environment has no torch, so a probe must fail exactly at the torch
stage with a structured ``failed`` run record and machine-readable events.
"""

import json
import subprocess
import sys
from pathlib import Path

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.data.manifest import manifest_sha256, write_manifest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "workers" / "prithvi" / "src" / "prithvi_worker" / "runner.py"


def build_request(tmp_path: Path, make_row) -> Path:
    manifest = tmp_path / "manifests" / "T3" / "abc123.jsonl"
    write_manifest(manifest, [make_row("one", "fire-a", "train")])
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "probe",
                "run_id": "prithvi-probe-test",
                "task_config_path": str(ROOT / "configs" / "tasks" / "T3_burn_intensity.yaml"),
                "model_config_path": str(ROOT / "configs" / "models" / "prithvi.yaml"),
                "protocol_config_path": str(
                    ROOT / "configs" / "protocols" / "prithvi_repo_config.yaml"
                ),
                "manifest_path": str(manifest),
                "data_root": str(tmp_path / "data"),
                "weights_root": str(tmp_path / "weights"),
                "cache_root": str(tmp_path / "cache"),
                "runs_root": str(tmp_path / "runs"),
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    return request


def test_runner_probe_fails_without_torch(tmp_path: Path, make_row, monkeypatch) -> None:
    request = build_request(tmp_path, make_row)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("PRITHVI_CHECKPOINT", str(checkpoint))

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(request)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )

    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert any(e["event"] == "probe.manifest" for e in events)
    assert any(e["event"] == "probe.checkpoint" for e in events)
    assert any(e["level"] == "error" and e["event"] == "probe.torch" for e in events)

    record = RunRecord.model_validate(
        json.loads(
            (tmp_path / "runs" / "prithvi-probe-test" / "run.json").read_text(encoding="utf-8")
        )
    )
    assert record.status == "failed"
    assert "torch" in (record.notes or "")
    assert record.model_checkpoint_sha256
    assert record.task_id == "T3"
    # the manifest content hash, not the file name (R10)
    manifest = tmp_path / "manifests" / "T3" / "abc123.jsonl"
    assert record.dataset.manifest_sha256 == manifest_sha256(manifest)
    assert record.dataset.manifest_sha256 != "abc123"
    assert record.training.regime == "reference"


def test_runner_probe_fails_on_missing_checkpoint(tmp_path: Path, make_row, monkeypatch) -> None:
    request = build_request(tmp_path, make_row)
    monkeypatch.setenv("PRITHVI_CHECKPOINT", str(tmp_path / "missing.pt"))

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(request)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )

    assert result.returncode == 1
    record = RunRecord.model_validate(
        json.loads(
            (tmp_path / "runs" / "prithvi-probe-test" / "run.json").read_text(encoding="utf-8")
        )
    )
    assert record.status == "failed"
    assert "checkpoint not found" in (record.notes or "")


def test_runner_rejects_unimplemented_action(tmp_path: Path, make_row) -> None:
    request = build_request(tmp_path, make_row)
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["action"] = "evaluate"
    request.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(request)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )

    assert result.returncode == 1
    record = RunRecord.model_validate(
        json.loads(
            (tmp_path / "runs" / "prithvi-probe-test" / "run.json").read_text(encoding="utf-8")
        )
    )
    assert record.status == "failed"
    assert "not implemented" in (record.notes or "")
