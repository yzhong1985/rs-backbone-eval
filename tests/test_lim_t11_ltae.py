"""Runner-level tests for the LIM T11 20-bin LTAE2d protocol.

Runs in the lim worker environment only (torch is transitive); the MiT stub
gives a tiny 4-level pyramid, and the vendored official LTAE2d is
instantiated at full size (n_head 16 / d_model 256) because its GroupNorm
needs those divisors either way - the CPU cost stays in the seconds range.
Only the *binning* side of the protocol needs small mocks: one epoch, two
tiles, batch 2.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

# lim_worker imports pull torch transitively; the root environment skips them
pytest.importorskip("torch")
import lim_worker.mit as mit  # noqa: E402
import lim_worker.t11_ltae as ltae  # noqa: E402
import torch  # noqa: E402
from lim_worker.runner import main as runner_main  # noqa: E402

from rs_backbone_eval.contracts import ManifestSample  # noqa: E402
from rs_backbone_eval.data.manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"AEF_{index:02d}" for index in range(64)]


class TinyMITEncoder(torch.nn.Module):
    """4-level pyramid with the same attribute surface as MixMIMEncoder.

    It carries one real parameter so the device-wiring regression cannot pass
    on an empty parameter list (review round 5).
    """

    embed_dims = (16, 32, 64, 128)  # all divisible by n_head=16 (vendored GroupNorm)
    sizes = ((16, 16), (8, 8), (4, 4), (2, 2))

    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def forward(self, x):
        batch = x.shape[0]
        # cuda tensors need a matching-device generator (torch 2.9); the cpu
        # path keeps the fixed generator so the metric tests stay stable
        generator = None if x.device.type == "cuda" else torch.Generator().manual_seed(123)
        return [
            torch.randn(
                batch,
                channels,
                height,
                width,
                device=x.device,
                generator=generator,
            )
            + self.probe
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
        },
        sensor="aef",
        bands=list(BANDS),
        timestamps=["2019"],
        temporal_mask=[True],
        gsd_m=10.0,
        checksum=hashlib.sha256(voxel.read_bytes()).hexdigest(),
        window=[0, 0, 8, 8],
    )


def write_specs(tmp_path: Path, protocol: str = "lim_s2_20bin") -> None:
    (tmp_path / "specs").mkdir()
    for group, name in (
        ("tasks", "T11_pastis_hd"),
        ("models", "lim"),
        ("protocols", protocol),
    ):
        payload = yaml.safe_load((ROOT / "configs" / group / f"{name}.yaml").read_text())
        (tmp_path / "specs" / f"{name}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_bin_indices_deterministic_and_full_coverage_edges() -> None:
    indices = ltae._bin_indices(43)
    assert indices == sorted(set(indices))
    assert len(indices) == 20
    assert indices[0] == 0 and indices[-1] == 42
    # fewer frames than bins -> every frame is used, unchanged
    assert ltae._bin_indices(5) == [0, 1, 2, 3, 4]
    assert ltae._bin_indices(43) == ltae._bin_indices(43)


def test_positions_are_days_since_the_pastis_reference() -> None:
    # 2018-09-01 is day 0; the series is monotone across New Year and has no
    # duplicate positions a year apart (review 2026-11-04).
    assert ltae._date_position(20180901) == 0
    assert ltae._date_position(20180924) == 23
    assert ltae._date_position(20190610) == 282
    assert ltae._date_position(20190924) == 388
    positions = [ltae._date_position(day) for day in (20180924, 20190117, 20190924)]
    assert positions == sorted(set(positions))
    assert positions[-1] > 366


@pytest.fixture()
def lim_stub(monkeypatch, tmp_path):
    # patched at the t11_ltae namespace (from-imports alias at import time)
    monkeypatch.setattr(mit, "mit_huge", lambda **kwargs: TinyMITEncoder())
    monkeypatch.setattr(mit, "load_encoder", lambda path, encoder: {"keys": 0})
    monkeypatch.setattr(ltae, "mit_huge", lambda **kwargs: TinyMITEncoder())
    monkeypatch.setattr(
        ltae, "training_device", lambda: ("cpu", {"device": "cpu", "host": "cpu", "torch": "stub"})
    )
    monkeypatch.setattr(ltae, "load_encoder", lambda path, encoder: {"keys": 0})
    checkpoint_path = Path(tmp_path) / "stub-checkpoint.pth"
    checkpoint_path.write_bytes(b"stub")
    monkeypatch.setenv("LIM_CHECKPOINT", checkpoint_path.as_posix())


def test_train_two_tiles_one_epoch(lim_stub, tmp_path, monkeypatch) -> None:
    dates = [20190610, 20190714, 20190812, 20190920, 20191030]
    write_specs(tmp_path)
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
    monkeypatch.setattr(ltae, "EPOCHS", 1)
    monkeypatch.setattr(ltae, "TRAIN_MIN", 2)
    import lim_worker.t11_common as t11_common

    monkeypatch.setattr(t11_common, "EXPECTED_TEST_SAMPLES", 1)
    monkeypatch.setattr(t11_common, "EXPECTED_VALIDATION_SAMPLES", 1)
    monkeypatch.setattr(ltae, "STATS_PATCHES", 1)
    monkeypatch.setattr(ltae, "BATCH", 2)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "train",
                "run_id": "lim-t11-20bin-smoke",
                "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
                "model_config_path": (tmp_path / "specs" / "lim.yaml").as_posix(),
                "protocol_config_path": (tmp_path / "specs" / "lim_s2_20bin.yaml").as_posix(),
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

    record = json.loads((runs_root / "lim-t11-20bin-smoke" / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["epochs_completed"] == 1
    assert record["training"]["optimizer_steps"] == 1
    assert record["training"]["head_id"] == "lim_s2_20bin_ltae2d_upernet_fpn256"
    assert record["input"]["preprocessing_id"] == "lim_s2_20bin_tilestats"
    tags = record["tags"]
    assert tags["device"] == "cpu" and tags["host"] == "cpu"
    assert tags["norm_source"] == "train_fold_tile_stats"
    assert tags["test_manifest_samples"] == "1"
    assert tags["test_expected_samples"] == "1"
    assert tags["test_samples_evaluated"] == "1"
    assert tags["validation_samples_evaluated"] == "1"
    assert tags["validation_expected_samples"] == "1"
    assert f"formal host {record['tags']['host']} (whole cell)" in (record["notes"] or "")
    primary = next(metric for metric in record["metrics"] if metric["is_primary"])
    assert primary["name"] == "mean_iou" and primary["split"] == "test"
    assert primary["value"] >= 0.0


def test_decode_stack_binned_frames_and_labels(lim_stub, tmp_path) -> None:
    dates = [20190610, 20190714, 20190812, 20190920, 20191030]
    sample = make_sample("7", "train", tmp_path, "tile-a", dates)
    features, labels, positions = ltae._decode_stack(sample, tmp_path, 19)
    # T=5 < BINS=20 -> all frames kept; positions are days since 2018-09-01
    assert features.shape == (5, 3, 8, 8)
    assert labels.shape == (8, 8)
    assert positions.tolist() == [ltae._date_position(day) for day in dates]
    # 43-frame stack keeps exactly 20 spread bins
    # 43 ascending *valid* dates spanning 2019; dtypes (T,10,8,8) like the real stacks
    wide_dates = [
        int((date(2019, 1, 1) + timedelta(days=offset)).strftime("%Y%m%d")) for offset in range(43)
    ]
    wide = make_sample("8", "test", tmp_path, "tile-b", wide_dates)
    wide_features, _, wide_positions = ltae._decode_stack(wide, tmp_path, 19)
    assert wide_features.shape[0] == 20
    assert len(wide_positions) == 20
    assert wide_positions.tolist() == sorted(set(wide_positions.tolist()))
    assert wide_positions[-1] > wide_positions[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a visible GPU")
def test_review_round2_all_module_wiring_ends_on_the_training_device(
    lim_stub, tmp_path, monkeypatch
) -> None:
    """Reviewer round-2 #1: the encoder must share the head's training device.

    The earlier fix moved only the head/model, so the first CUDA batch crashed
    with CPU encoder weights. This regression asserts every parameter of the
    (stub) encoder sits on the device the run tagged - the stub here is left
    un-pinned so training_device() reports the real hardware.
    """
    # repin the fixture's cpu override: this test must negotiate for real
    from lim_worker.t11_common import training_device as real_device

    monkeypatch.setattr(ltae, "training_device", real_device)

    dates = [20190610, 20190714, 20190812]
    write_specs(tmp_path)
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
    import lim_worker.t11_common as t11_common

    monkeypatch.setattr(ltae, "EPOCHS", 0)  # straight to eval, seconds
    monkeypatch.setattr(ltae, "TRAIN_MIN", 2)
    monkeypatch.setattr(t11_common, "EXPECTED_TEST_SAMPLES", 1)
    monkeypatch.setattr(t11_common, "EXPECTED_VALIDATION_SAMPLES", 1)

    encoders: list = []
    real_mit = ltae.mit_huge

    def capture_mit(*args, **kwargs):
        enc = real_mit(*args, **kwargs)
        encoders.append(enc)
        return enc

    monkeypatch.setattr(ltae, "mit_huge", capture_mit)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "train",
                "run_id": "lim-t11-20bin-device",
                "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
                "model_config_path": (tmp_path / "specs" / "lim.yaml").as_posix(),
                "protocol_config_path": (tmp_path / "specs" / "lim_s2_20bin.yaml").as_posix(),
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
    record = json.loads((runs_root / "lim-t11-20bin-device" / "run.json").read_text())
    claimed = record["tags"]["device"]
    assert claimed == "cuda"
    # the tiny stub's parameters must live on the claimed training device;
    # the list must be non-empty or this assertion proves nothing (round 5)
    parameters = list(encoders[0].parameters())
    assert parameters, "encoder stub exposes no parameters - regression is vacuous"
    assert all(parameter.device.type == "cuda" for parameter in parameters)


def test_require_device_flag_refuses_cpu_fallback(monkeypatch) -> None:
    """Review round 4: a GPU-declared cell must fail loudly, not train on CPU."""
    import lim_worker.t11_common as t11_common

    monkeypatch.setenv("RS_EVAL_REQUIRE_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="refusing to train on CPU"):
        t11_common.training_device()


def test_full_finetune_unfreezes_encoder_and_checkpoints_frames(
    lim_stub, tmp_path, monkeypatch
) -> None:
    """B1 variant: encoder gradients flow, one frame per checkpoint, same budget."""
    dates = [20190610, 20190714, 20190812, 20190920, 20191030]
    write_specs(tmp_path, "lim_s2_20bin_finetune")
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

    captured = {}

    def factory(**kwargs):
        encoder = TinyMITEncoder()
        captured["encoder"] = encoder
        encoder.probe.register_hook(lambda grad: captured.setdefault("grad", grad.detach()))
        return encoder

    monkeypatch.setattr(ltae, "mit_huge", factory)
    monkeypatch.setattr(ltae, "EPOCHS", 1)
    monkeypatch.setattr(ltae, "TRAIN_MIN", 2)
    monkeypatch.setattr(ltae, "STATS_PATCHES", 1)
    monkeypatch.setattr(ltae, "BATCH", 1)  # two tiles -> two micro-batches
    monkeypatch.setenv("LIM_T11_BATCH", "1")
    monkeypatch.setenv("LIM_T11_ACCUM", "1")
    monkeypatch.setenv("LIM_T11_CHECKPOINT_FRAMES", "4")
    import lim_worker.t11_common as t11_common

    monkeypatch.setattr(t11_common, "EXPECTED_TEST_SAMPLES", 1)
    monkeypatch.setattr(t11_common, "EXPECTED_VALIDATION_SAMPLES", 1)

    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "action": "train",
                "run_id": "lim-t11-20bin-finetune-smoke",
                "task_config_path": (tmp_path / "specs" / "T11_pastis_hd.yaml").as_posix(),
                "model_config_path": (tmp_path / "specs" / "lim.yaml").as_posix(),
                "protocol_config_path": (
                    tmp_path / "specs" / "lim_s2_20bin_finetune.yaml"
                ).as_posix(),
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

    record = json.loads((runs_root / "lim-t11-20bin-finetune-smoke" / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["training"]["head_id"] == ("lim_s2_20bin_finetune_ltae2d_upernet_fpn256")
    tags = record["tags"]
    assert tags["backbone_frozen"] == "False"
    assert tags["grad_checkpointing"] == "frame"
    assert tags["checkpoint_frames"] == "4"
    assert tags["head_lr"] == "0.001"
    assert tags["backbone_lr"] == "0.0001"
    assert captured.get("grad") is not None
    assert float(captured["grad"].abs().sum()) > 0
    assert "full fine-tune" in (record["notes"] or "")
