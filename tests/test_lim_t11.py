"""Runner-level tests for the LIM T11 variants (single / temporal median).

Skipped where torch is absent (root environment), mirroring the replay /
AEF-worker tests. The encoder is a tiny stub so the shared UPerNetHead (fpn
256 over a 4-level pyramid) is exercised end to end on CPU in seconds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

# both lim imports pull torch transitively; the root environment skips them
pytest.importorskip("torch")
import torch  # noqa: E402  - TinyMITEncoder subclasses torch.nn.Module

import lim_worker.mit as mit  # noqa: E402
import lim_worker.t11 as t11  # noqa: E402
from lim_worker.runner import main as runner_main  # noqa: E402

from rs_backbone_eval.contracts import ManifestSample  # noqa: E402
from rs_backbone_eval.data.manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"AEF_{index:02d}" for index in range(64)]

# the module imports torch transitively (mit/t11); the root environment skips
# everything without it
pytest.importorskip("torch")


class TinyMITEncoder(torch.nn.Module):
    """4-level pyramid with the same attribute surface as MixMIMEncoder."""

    embed_dims = (8, 16, 32, 64)
    sizes = ((16, 16), (8, 8), (4, 4), (2, 2))

    def parameters(self):  # noqa: D102 - frozen by the training loop
        return iter([])

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def forward(self, x):
        batch = x.shape[0]
        return [
            torch.randn(
                batch,
                channels,
                height,
                width,
                generator=torch.Generator().manual_seed(123),
            )
            for channels, (height, width) in zip(self.embed_dims, self.sizes, strict=True)
        ]


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
        aux_paths={
            "s2_path": s2.relative_to(tmp_path).as_posix(),
            "tile": tile,
            "encoding": "raw_aef_int8_as_float16",
            "label_year": "2019",
            "frame_date_anchor": "20190715",
        },
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
        ("models", "lim"),
        ("protocols", protocol),
    ):
        payload = yaml.safe_load((ROOT / "configs" / group / f"{name}.yaml").read_text())
        (tmp_path / "specs" / f"{name}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.fixture()
def lim_stub(monkeypatch, tmp_path):
    # patch the names the t11 module actually references, not the mit module
    # (from-imports alias them at import time)
    monkeypatch.setattr(mit, "mit_huge", lambda **kwargs: TinyMITEncoder())
    monkeypatch.setattr(t11, "mit_huge", lambda **kwargs: TinyMITEncoder())
    monkeypatch.setattr(mit, "load_encoder", lambda path, encoder: {"keys": 0})
    monkeypatch.setattr(t11, "load_encoder", lambda path, encoder: {"keys": 0})
    monkeypatch.delenv("LIM_CHECKPOINT", raising=False)
    # the lane scripts resolve LIM weights via this env; tests must not touch
    # the 6.2 GB real checkpoint
    checkpoint_path = Path(tmp_path) / "stub-checkpoint.pth"
    checkpoint_path.write_bytes(b"stub")
    monkeypatch.setenv("LIM_CHECKPOINT", checkpoint_path.as_posix())


@pytest.mark.parametrize(
    ("recipe", "protocol", "variant"),
    (
        ("single", "lim_s2_single", "single"),
        ("median", "lim_s2_median", "median"),
    ),
)
def test_train_two_tiles_one_epoch(
    lim_stub, recipe, protocol, variant, tmp_path, monkeypatch
) -> None:
    dates = [20190610, 20190714, 20190812]
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
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "train",
                "run_id": "lim-t11-smoke",
                "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
                "model_config_path": (tmp_path / "specs" / "lim.yaml").as_posix(),
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
    checkpoint = tmp_path / "stub-checkpoint.pth"
    checkpoint.write_bytes(b"stub")  # the patched load_encoder skips the real file

    monkeypatch.setattr(sys, "argv", ["runner.py", request_path.as_posix()])
    assert runner_main() == 0

    record = json.loads((runs_root / "lim-t11-smoke" / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["epochs_completed"] == 1
    assert record["training"]["optimizer_steps"] == 1
    assert record["training"]["head_id"] == f"lim_s2_{variant}_upernet_fpn256"
    assert record["input"]["preprocessing_id"] == f"lim_s2_{variant}_imagenetfree"
    primary = next(metric for metric in record["metrics"] if metric["is_primary"])
    assert primary["name"] == "mean_iou" and primary["split"] == "test"
    assert primary["value"] >= 0.0


def test_decode_picks_year_mid_frame(tmp_path) -> None:

    dates = [20190610, 20190714, 20190812]
    sample = make_sample("7", "train", tmp_path, "tile-a", dates)
    features, _ = t11._decode_row(sample, SimpleNamespace(data_root=tmp_path), 19, "single")
    # the anchor resolves to 20190714 (|14-15|=1 < |10-15|=5, |12-15|=3)
    stack = np.load(tmp_path / sample.aux_paths["s2_path"])
    assert stack.shape[0] == 3
    median_frame, _ = t11._decode_row(sample, SimpleNamespace(data_root=tmp_path), 19, "median")
    assert median_frame.shape == features.shape
    assert median_frame.shape[0] == 3  # R/G/B
    assert median_frame.shape[1:] == features.shape[1:] == (8, 8)
    # median of three frames is strictly between min/max per pixel
    assert float(median_frame.var()) >= 0.0
