"""Runner-level tests for the AnySat T11 protocols (single / median / 20-bin).

Runs in the anysat worker environment (torch + the worker package); the
backbone is stubbed so the shared LiteSpatialHead and the S2 decode path are
exercised end-to-end on CPU in seconds.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import anysat_worker.t11 as t11  # noqa: E402
import torch  # noqa: E402
from anysat_worker.runner import main as runner_main  # noqa: E402

from rs_backbone_eval.contracts import ManifestSample  # noqa: E402
from rs_backbone_eval.data.manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"AEF_{index:02d}" for index in range(64)]


class TinyDenseBackbone(torch.nn.Module):
    """Dense-output stub: (B,T,10,H,W) -> (B,H,W,768)."""

    def __init__(self, dim: int = 1536):
        super().__init__()
        self.dim = dim
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def forward(self, payload, patch_size=None, output=None, output_modality=None):
        s2 = payload["s2"]
        height, width = s2.shape[-2:]
        return torch.randn(
            s2.shape[0],
            height,
            width,
            self.dim,
            device=s2.device,
            generator=None if s2.device.type == "cuda" else torch.Generator().manual_seed(7),
        )


def make_sample(sample_id: str, split: str, tmp_path: Path, tile: str, dates: list[int]):
    frames = len(dates)
    rng = np.random.default_rng(int(sample_id) + 1)
    stack = (rng.normal(2000, 500, size=(frames, 10, 8, 8))).astype(np.int16)
    array_dir = tmp_path / "derived" / "aef-2019"
    s2 = array_dir / "PASTIS-HD" / "DATA_S2" / f"S2_{sample_id}.npy"
    s2.parent.mkdir(parents=True, exist_ok=True)
    np.save(s2, stack)
    voxel = array_dir / "features" / "2019" / f"{sample_id}.npz"
    voxel.parent.mkdir(parents=True, exist_ok=True)
    np.savez(voxel, features=np.zeros((64, 8, 8), np.float16), valid=np.ones((8, 8), bool))
    sidecar = array_dir / "features" / "2019" / f"{sample_id}.json"
    sidecar.write_text(json.dumps({"patch": int(sample_id), "dates_s2": dates}), encoding="utf-8")
    labels = np.zeros((1, 8, 8), dtype=np.uint8)
    labels[0, :4] = 2
    label = tmp_path / "labels" / f"{sample_id}.npy"
    label.parent.mkdir(parents=True, exist_ok=True)
    np.save(label, labels)
    import hashlib

    return ManifestSample(
        task_id="T11",
        sample_id=sample_id,
        group_id=sample_id,
        split=split,
        image_paths=[voxel.relative_to(tmp_path).as_posix()],
        label_path=label.relative_to(tmp_path).as_posix(),
        aux_paths={"s2_path": s2.relative_to(tmp_path).as_posix(), "tile": tile},
        sensor="aef",
        bands=list(BANDS),
        timestamps=["2019"],
        temporal_mask=[True],
        gsd_m=10.0,
        checksum=hashlib.sha256(voxel.read_bytes()).hexdigest(),
        window=[0, 0, 8, 8],
    )


def write_specs(tmp_path: Path, protocol: str) -> None:
    (tmp_path / "specs").mkdir()
    for group, name in (
        ("tasks", "T11_pastis_hd"),
        ("models", "anysat"),
        ("protocols", protocol),
    ):
        payload_spec = __import__("yaml").safe_load(
            (ROOT / "configs" / group / f"{name}.yaml").read_text()
        )
        (tmp_path / "specs" / f"{name}.yaml").write_text(
            __import__("yaml").safe_dump(payload_spec), encoding="utf-8"
        )


@pytest.fixture()
def anysat_stub(monkeypatch, tmp_path):
    monkeypatch.setattr(t11, "build_backbone", lambda checkpoint: TinyDenseBackbone())
    monkeypatch.setattr(
        t11,
        "training_device",
        lambda: ("cpu", {"device": "cpu", "host": "cpu", "torch": "stub"}),
    )
    checkpoint = Path(tmp_path) / "stub-anysat.pth"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv("ANYSAT_CHECKPOINT", checkpoint.as_posix())
    # the runner tags the upstream commit before dispatching; stub that out
    import anysat_worker.runner as runner_mod

    source = Path(tmp_path) / "anysat-src"
    source.mkdir()
    (source / "hubconf.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "add_upstream_to_path", lambda: source)
    monkeypatch.setattr(runner_mod, "upstream_commit", lambda root: "stub-commit")


@pytest.mark.parametrize(
    ("protocol", "variant"),
    (
        ("anysat_s2_single", "single"),
        ("anysat_s2_20bin", "20bin"),
    ),
)
def test_train_two_tiles_one_epoch(anysat_stub, tmp_path, monkeypatch, protocol, variant) -> None:
    dates = [20190610, 20190714, 20190812, 20190920, 20191030]
    write_specs(tmp_path, protocol)
    rows = [
        make_sample("1", "train", tmp_path, "tile-a", dates),
        make_sample("2", "train", tmp_path, "tile-b", dates),
        make_sample("3", "validation", tmp_path, "tile-a", dates),
        make_sample("4", "test", tmp_path, "tile-b", dates),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setattr(t11, "EPOCHS", 1)
    monkeypatch.setattr(t11, "TRAIN_MIN", 2)
    monkeypatch.setattr(t11, "STATS_PATCHES", 1)
    monkeypatch.setattr(t11, "BATCH", 1)
    monkeypatch.setattr(t11, "ACCUM", 1)
    monkeypatch.setattr(
        t11,
        "OFFICIAL_TEST_SAMPLES",
        {"T11": 1},
    )
    monkeypatch.setattr(
        t11,
        "OFFICIAL_VALIDATION_SAMPLES",
        {"T11": 1},
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "train",
                "run_id": "anysat-t11-smoke",
                "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
                "model_config_path": (tmp_path / "specs" / "anysat.yaml").as_posix(),
                "protocol_config_path": (tmp_path / "specs" / f"{protocol}.yaml").as_posix(),
                "manifest_path": manifest.as_posix(),
                "data_root": tmp_path.as_posix(),
                "weights_root": (tmp_path / "weights").as_posix(),
                "cache_root": (tmp_path / "cache").as_posix(),
                "runs_root": runs_root.as_posix(),
                "seed": 17,
                "resume_from": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["runner.py", request_path.as_posix()])
    assert runner_main() == 0

    record = json.loads((runs_root / "anysat-t11-smoke" / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["head_id"] == f"anysat_s2_{variant}_litespatial"
    assert record["input"]["preprocessing_id"] == f"anysat_s2_{variant}_tilestats"
    tags = record["tags"]
    assert tags["device"] == "cpu"
    assert tags["test_samples_evaluated"] == "1"
    assert tags["validation_samples_evaluated"] == "1"
    primary = next(metric for metric in record["metrics"] if metric["is_primary"])
    assert primary["name"] == "mean_iou" and primary["value"] >= 0.0


def test_decode_variants(anysat_stub, tmp_path) -> None:
    dates = [20190610, 20190714, 20190812, 20190920, 20191030]
    sample = make_sample("7", "train", tmp_path, "tile-a", dates)
    single, labels, doys = t11._decode(sample, tmp_path, 19, "single")
    assert single.shape == (1, 10, 8, 8) and doys == [t11._doy(20190714)]
    wide_dates = [
        int((date(2019, 1, 1) + timedelta(days=offset)).strftime("%Y%m%d")) for offset in range(43)
    ]
    wide = make_sample("8", "test", tmp_path, "tile-b", wide_dates)
    binned, _, binned_doys = t11._decode(wide, tmp_path, 19, "20bin")
    assert binned.shape[0] == 20 and len(binned_doys) == 20
