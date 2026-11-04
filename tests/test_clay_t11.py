"""Runner-level tests for the Clay T11 single protocol.

Runs in the clay worker environment; the Clay encoder is stubbed (tokens grid
at stride 8, dim 1024) so the shared UPerNet head and the S2 decode path are
exercised on CPU in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import clay_worker.t11 as t11  # noqa: E402
import torch  # noqa: E402
from clay_worker.runner import main as runner_main  # noqa: E402

from rs_backbone_eval.contracts import ManifestSample  # noqa: E402
from rs_backbone_eval.data.manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"AEF_{index:02d}" for index in range(64)]
BAND_NAMES = [
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
]


class TinyClayEncoder(torch.nn.Module):
    """Stub encoder: (B,10,H,W) -> tokens (B,(H/8)^2,1024)."""

    def __init__(self, dim: int = 1024, patch: int = 8):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, datacube):
        pixels = datacube["pixels"]
        grid = pixels.shape[-1] // self.patch
        tokens = torch.randn(
            pixels.shape[0],
            grid * grid,
            self.dim,
            device=pixels.device,
            generator=None if pixels.device.type == "cuda" else torch.Generator().manual_seed(11),
        )
        return tokens


def fake_metadata() -> dict:
    return {
        "band_order": BAND_NAMES,
        "gsd": 10,
        "bands": {
            "mean": {band: 1000.0 for band in BAND_NAMES},
            "std": {band: 500.0 for band in BAND_NAMES},
            "wavelength": {band: 0.5 for band in BAND_NAMES},
        },
    }


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


def write_specs(tmp_path: Path, protocol: str = "clay_s2_single") -> None:
    (tmp_path / "specs").mkdir()
    for group, name in (
        ("tasks", "T11_pastis_hd"),
        ("models", "clay"),
        ("protocols", protocol),
    ):
        payload = __import__("yaml").safe_load(
            (ROOT / "configs" / group / f"{name}.yaml").read_text()
        )
        (tmp_path / "specs" / f"{name}.yaml").write_text(
            __import__("yaml").safe_dump(payload), encoding="utf-8"
        )


@pytest.fixture()
def clay_stub(monkeypatch, tmp_path):
    monkeypatch.setattr(t11, "_build_encoder", lambda checkpoint: TinyClayEncoder())
    monkeypatch.setattr(t11, "load_sensor_metadata", lambda root, sensor: fake_metadata())
    monkeypatch.setattr(t11, "add_upstream_to_path", lambda: Path(tmp_path))
    monkeypatch.setattr(
        t11,
        "training_device",
        lambda: ("cpu", {"device": "cpu", "host": "cpu", "torch": "stub"}),
    )
    checkpoint = Path(tmp_path) / "stub-clay.pth"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv("CLAY_CHECKPOINT", checkpoint.as_posix())
    import clay_worker.runner as runner_mod

    monkeypatch.setattr(runner_mod, "add_upstream_to_path", lambda: Path(tmp_path))
    monkeypatch.setattr(runner_mod, "upstream_commit", lambda root: "stub-commit")


@pytest.mark.parametrize(
    ("protocol", "head_id"),
    [
        ("clay_s2_single", "clay_s2_single_upernet_fpn256"),
        ("clay_s2_20bin", "clay_s2_20bin_ltae2d_upernet_fpn256"),
    ],
)
def test_train_two_tiles_one_epoch(
    clay_stub, tmp_path, monkeypatch, protocol: str, head_id: str
) -> None:
    dates = [20190610, 20190714, 20190812]
    write_specs(tmp_path, protocol)
    run_id = f"clay-t11-{protocol}"
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
                "model_config_path": (tmp_path / "specs" / "clay.yaml").as_posix(),
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

    record = json.loads((runs_root / run_id / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["head_id"] == head_id
    assert record["input"]["preprocessing_id"] == f"clay_s2_{protocol.split('_')[-1]}_claystats"
    tags = record["tags"]
    assert tags["device"] == "cpu"
    assert tags["test_samples_evaluated"] == "1"
    assert tags["validation_samples_evaluated"] == "1"
    assert tags["norm_source"] == "clay_sensor_metadata"
    if protocol == "clay_s2_20bin":
        assert tags["temporal_positions"] == "days since 2018-09-01"
        assert tags["ltae_T"] == "1000"
    primary = next(metric for metric in record["metrics"] if metric["is_primary"])
    assert primary["name"] == "mean_iou" and primary["value"] >= 0.0


def test_decode_single_picks_year_mid_frame(clay_stub, tmp_path) -> None:
    dates = [20190610, 20190714, 20190812]
    sample = make_sample("7", "train", tmp_path, "tile-a", dates)
    pixels, labels, positions = t11._decode(sample, tmp_path, 19, "single")
    assert pixels.shape == (10, 64, 64)
    assert labels.shape == (64, 64)
    assert positions == [t11._date_position(20190714)]


def test_decode_20bin_positions_are_monotone(clay_stub, tmp_path) -> None:
    dates = [20180924, 20190117, 20190610, 20190924, 20191019]
    sample = make_sample("8", "train", tmp_path, "tile-a", dates)
    pixels, labels, positions = t11._decode(sample, tmp_path, 19, "20bin")
    assert pixels.shape == (5, 10, 64, 64)
    assert labels.shape == (64, 64)
    assert positions == sorted(set(positions))
    assert positions[0] == 23 and positions[-1] > 366
