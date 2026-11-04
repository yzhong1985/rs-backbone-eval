"""Runner-level tests for the OlmoEarth T11 single protocol.

Runs in the olmoearth worker environment; the upstream encoder and the
12-band adapter's forward are stubbed (features come back as a fake token
grid) so the head/loss/coverage path is exercised on CPU in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import olmoearth_worker.t1_frozen as t1  # noqa: E402
import olmoearth_worker.t11 as t11  # noqa: E402
import torch  # noqa: E402
from olmoearth_worker.runner import main as runner_main  # noqa: E402

from rs_backbone_eval.contracts import ManifestSample  # noqa: E402
from rs_backbone_eval.data.manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"AEF_{index:02d}" for index in range(64)]


class TinyEncoder(torch.nn.Module):
    def __init__(self, dim: int = 16):  # LTAE GroupNorm needs dim % n_head == 0
        super().__init__()
        self.dim = dim
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, sample, patch_size=None, input_res=None):
        return {"tokens_and_masks": SimpleNamespace()}


def fake_encode(encoder, expanded, mask, stamps):
    grid = 14
    return torch.randn(
        expanded.shape[0],
        expanded.shape[3],  # date axis
        encoder.dim,
        grid,
        grid,
        device=expanded.device,
        generator=None if expanded.device.type == "cuda" else torch.Generator().manual_seed(5),
    )


def make_sample(sample_id: str, split: str, tmp_path: Path, tile: str, dates: list[int]):
    rng = np.random.default_rng(int(sample_id) + 1)
    stack = (rng.normal(2000, 500, size=(len(dates), 10, 64, 64))).astype(np.int16)
    array_dir = tmp_path / "derived" / "aef-2019"
    s2 = array_dir / "PASTIS-HD" / "DATA_S2" / f"S2_{sample_id}.npy"
    s2.parent.mkdir(parents=True, exist_ok=True)
    np.save(s2, stack)
    voxel = array_dir / "features" / "2019" / f"{sample_id}.npz"
    voxel.parent.mkdir(parents=True, exist_ok=True)
    np.savez(voxel, features=np.zeros((64, 64, 64), np.float16), valid=np.ones((64, 64), bool))
    sidecar = array_dir / "features" / "2019" / f"{sample_id}.json"
    sidecar.write_text(json.dumps({"patch": int(sample_id), "dates_s2": dates}), encoding="utf-8")
    labels = np.zeros((1, 64, 64), dtype=np.uint8)
    labels[0, :32] = 2
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
        window=[0, 0, 64, 64],
    )


def write_specs(tmp_path: Path, protocol: str = "olmoearth_s2_single") -> None:
    (tmp_path / "specs").mkdir()
    for group, name in (
        ("tasks", "T11_pastis_hd"),
        ("models", "olmoearth"),
        ("protocols", protocol),
    ):
        payload = __import__("yaml").safe_load(
            (ROOT / "configs" / group / f"{name}.yaml").read_text()
        )
        (tmp_path / "specs" / f"{name}.yaml").write_text(
            __import__("yaml").safe_dump(payload), encoding="utf-8"
        )


@pytest.fixture()
def olmo_stub(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "weights.pth").write_bytes(b"stub")
    monkeypatch.setattr(t1, "build_encoder", lambda path, load_weights=True: TinyEncoder())
    monkeypatch.setattr(t11, "build_encoder", lambda path, load_weights=True: TinyEncoder())
    monkeypatch.setattr(t11, "_encode", fake_encode)
    monkeypatch.setattr(
        t1, "resolve_checkpoint_dir", lambda request: (weights, "OLMOEARTH_CHECKPOINT")
    )
    monkeypatch.setattr(t1, "record_checkpoint_hash", lambda writer, path: None)
    monkeypatch.setattr(
        t11,
        "training_device",
        lambda: ("cpu", {"device": "cpu", "host": "cpu", "torch": "stub"}),
    )
    monkeypatch.setenv("OLMOEARTH_CHECKPOINT", weights.as_posix())


@pytest.mark.parametrize(
    ("protocol", "head_id"),
    [
        ("olmoearth_s2_single", "olmoearth_s2_single_upernet_fpn256"),
        ("olmoearth_s2_20bin", "olmoearth_s2_20bin_ltae2d_upernet_fpn256"),
    ],
)
def test_train_two_tiles_one_epoch(
    olmo_stub, tmp_path, monkeypatch, protocol: str, head_id: str
) -> None:
    dates = [20190610, 20190714, 20190812]
    write_specs(tmp_path, protocol)
    run_id = f"olmoearth-t11-{protocol}"
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
    monkeypatch.setattr(t11, "BATCH", 1)
    monkeypatch.setattr(t11, "ACCUM", 1)
    monkeypatch.setattr(t11, "OFFICIAL_TEST_SAMPLES", {"T11": 1})
    monkeypatch.setattr(t11, "OFFICIAL_VALIDATION_SAMPLES", {"T11": 1})
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "train",
                "run_id": run_id,
                "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
                "model_config_path": (tmp_path / "specs" / "olmoearth.yaml").as_posix(),
                "protocol_config_path": (tmp_path / "specs" / f"{protocol}.yaml").as_posix(),
                "manifest_path": manifest.as_posix(),
                "data_root": tmp_path.as_posix(),
                "weights_root": (tmp_path / "w").as_posix(),
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

    record = json.loads((runs_root / run_id / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["head_id"] == head_id
    variant = protocol.rsplit("_", 1)[-1]
    assert record["input"]["preprocessing_id"] == f"olmoearth_s2_{variant}_tilestats"
    tags = record["tags"]
    assert tags["device"] == "cpu"
    assert tags["adapter_bandsets"] == "set0 online, set1 online, set2 missing"
    # 128 px upsampled to 224 px: the encoder must be told the effective 5.7 m
    assert tags["encoder_input_px"] == "224"
    assert float(tags["encoder_input_res_m"]) == pytest.approx(10 * 128 / 224, abs=1e-3)
    if protocol == "olmoearth_s2_20bin":
        assert tags["temporal_positions"] == "days since 2018-09-01"
        assert tags["ltae_T"] == "1000"
    assert tags["test_samples_evaluated"] == "1"
    assert tags["validation_samples_evaluated"] == "1"
    primary = next(metric for metric in record["metrics"] if metric["is_primary"])
    assert primary["name"] == "mean_iou" and primary["value"] >= 0.0


def test_decode_single_picks_year_mid_frame(olmo_stub, tmp_path) -> None:
    dates = [20190610, 20190714, 20190812]
    sample = make_sample("7", "train", tmp_path, "tile-a", dates)
    frames, labels, stamps, positions = t11._decode_frames(sample, tmp_path, 19, "single")
    assert frames.shape == (1, 10, 64, 64)
    assert labels.shape == (64, 64)
    assert stamps == [[14, 6, 2019]]  # day, month-1, year
    assert positions == [t11.days_since_reference(20190714)]


def test_decode_20bin_positions_are_monotone(olmo_stub, tmp_path) -> None:
    dates = [20180924, 20190117, 20190610, 20190924, 20191019]
    sample = make_sample("8", "train", tmp_path, "tile-a", dates)
    frames, labels, stamps, positions = t11._decode_frames(sample, tmp_path, 19, "20bin")
    assert frames.shape == (5, 10, 64, 64)
    assert labels.shape == (64, 64)
    assert positions == sorted(set(positions))
    assert positions[0] == 23 and positions[-1] > 366
    assert len(stamps) == 5
