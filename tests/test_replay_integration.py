"""End-to-end replay regressions for review 1b3ceae (2026-11-03), CPU only.

A tiny encoder stands in for MiT-Huge; the detector, ``load_for_eval``, the
renderer and ``scripts/visualize_run.py`` are the real code. Needs torch and
rasterio, so the root environment skips the module. Run with the LIM worker
environment, e.g.:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:workers/lim/src \\
        workers/lim/.venv/bin/python -m pytest tests/test_replay_integration.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
replay = pytest.importorskip("lim_worker.replay")

import torch.nn.functional as F  # noqa: E402
from lim_worker.t10 import LimDetector  # noqa: E402
from PIL import Image  # noqa: E402
from test_review_fixes import payload  # noqa: E402

from rs_backbone_eval.contracts import RunRecord  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GREEN = (61, 220, 151)


class TinyEncoder(torch.nn.Module):
    embed_dims = [4, 4, 4, 4]

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, x):
        y = self.conv(x)
        return [F.avg_pool2d(y, stride) for stride in (4, 8, 16, 32)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generator():
    spec = importlib.util.spec_from_file_location(
        "visualize_run_under_test", REPO / "scripts" / "visualize_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(runs: Path, *, name: str = "synthetic", checkpoint: str = "latest") -> Path:
    """A completed T10 run with a 64 px scene, a tiny model and a full identity."""
    root = runs / name
    root.mkdir(parents=True)
    with rasterio.open(
        root / "image.png", "w", driver="PNG", width=64, height=64, count=3, dtype="uint8"
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    (root / "ann.json").write_text(json.dumps({"width": 64, "height": 64, "instances": []}))
    row = {
        "window": [0, 0, 64, 64],
        "annotation_path": "ann.json",
        "image_paths": ["image.png"],
        "valid_fraction": 1.0,
    }
    data_manifest = root / "data.jsonl"
    data_manifest.write_text(
        "\n".join(json.dumps(row | {"split": s, "group_id": s}) for s in ("train", "test")) + "\n"
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    raw = payload(run_id=name, model_id="lim_mit_huge", seed=17)
    raw.update(
        task_id="T10",
        protocol_id="rotated_frozen",
        git_commit=commit,
        model_checkpoint_sha256="a" * 64,
    )
    raw["training"].update(regime="frozen_probe", head_id="rotated_frcnn_gn", optimizer_steps=1)
    raw["dataset"].update(dataset_id="FAIR1M", manifest_sha256=_sha(data_manifest))
    raw["tags"].update(
        anchor_sizes="32,64,128,256",
        anchor_ratios="0.5,1,2",
        fpn_dim="4",
        normalization="train-tile mean [123.675,116.28,103.53] std [58.395,57.12,57.375]",
    )
    record = RunRecord.model_validate(raw).model_dump(mode="json")
    (root / "run.json").write_text(json.dumps(record))
    configs = {
        "task": REPO / "configs/tasks/T10_fair1m.yaml",
        "model": REPO / "configs/models/lim.yaml",
        "protocol": REPO / "configs/protocols/rotated_frozen.yaml",
    }
    request = {
        "manifest_path": str(data_manifest),
        "data_root": str(root),
        "task_config_path": str(configs["task"]),
        "model_config_path": str(configs["model"]),
        "protocol_config_path": str(configs["protocol"]),
    }
    (root / "request.json").write_text(json.dumps(request))

    identity = {
        "task": "T10",
        "model": "lim_mit_huge",
        "protocol": "rotated_frozen",
        "seed": 17,
        "manifest": record["dataset"]["manifest_sha256"],
        "pretrained_checkpoint": record["model_checkpoint_sha256"],
        "git_commit": commit,
        "torch_version": str(torch.__version__),
        "device_type": "cpu",
        "configs": {key: _sha(path) for key, path in configs.items()},
    }
    model = LimDetector(TinyEncoder(), 37, fpn_dim=4)
    with torch.no_grad():
        model.detector.cls_score.weight.zero_()
        model.detector.cls_score.bias.zero_()
    if checkpoint == "latest":
        torch.save(
            {"format_version": 1, "model": model.state_dict(), "identity": identity},
            root / "latest.pt",
        )
    return root


def _rewrite_identity(root: Path, **changes) -> None:
    path = root / "latest.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved["identity"].update(changes)
    torch.save(saved, path)


@pytest.fixture
def generator(monkeypatch):
    """The real generator, with the git-dependent inputs pinned.

    Tests run in a dirty worktree whose files differ from the run commit; both
    are inputs to the original-protocol rule and are covered by
    ``test_visualization``. Pinning them isolates the identity checks under test.
    """
    module = _generator()
    monkeypatch.setattr(module, "_worktree_clean", lambda repo: True)
    monkeypatch.setattr(
        module,
        "_file_hashes",
        lambda repo, names, commit: ({n: "h" for n in names}, {n: "h" for n in names}),
    )
    monkeypatch.setattr(replay, "mit_huge", lambda **kwargs: TinyEncoder())
    torch.set_num_threads(1)
    return module


def _run(module, runs: Path, *extra: str) -> int:
    return module.main(
        [
            "--runs-root",
            str(runs),
            "--run-id",
            "synthetic",
            "--pool",
            "1",
            "--per-bucket",
            "1",
            "--device",
            "cpu",
            *extra,
        ]
    )


# ----------------------------------------------------------------- head_only


def test_head_only_replays_through_load_for_eval(tmp_path, monkeypatch) -> None:
    """The full entry point, not just ``_load_state`` (review 1b3ceae §2)."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "task.yaml").write_text("num_outputs: 1\n")
    manifest = root / "data.jsonl"
    manifest.write_text(
        json.dumps({"split": "train", "group_id": "train", "window": [0, 0, 16, 16]}) + "\n"
    )
    backbone = root / "backbone.pth"
    backbone.write_bytes(b"tiny-pretrained")
    record = {
        "run_id": "synthetic",
        "task_id": "T10",
        "protocol_id": "rotated_frozen",
        "model_id": "lim_mit_huge",
        "status": "completed",
        "seed": 17,
        "git_commit": "0" * 40,
        "tags": {
            "anchor_sizes": "32,64,128,256",
            "anchor_ratios": "0.5,1,2",
            "fpn_dim": "4",
            "normalization": "train-tile mean [0,0,0] std [1,1,1]",
        },
    }
    (root / "run.json").write_text(json.dumps(record))
    (root / "request.json").write_text(
        json.dumps(
            {
                "task_config_path": str(root / "task.yaml"),
                "data_root": str(root),
                "manifest_path": str(manifest),
            }
        )
    )
    trained = LimDetector(TinyEncoder(), 1, fpn_dim=4)
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.fill_(0.5)
    head = {
        key: value
        for key, value in trained.state_dict().items()
        if not key.startswith(replay.BACKBONE_PREFIX)
    }
    head_path = root / "head_only.pt"

    def save(state):
        torch.save(
            {"identity": {"pretrained_checkpoint": _sha(backbone)}, "head_state_dict": state},
            head_path,
        )

    def fake_encoder_load(path, encoder):
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.fill_(3.0)
        return {"loaded": 2, "missing": 0, "unexpected": 0}

    monkeypatch.setenv("LIM_CHECKPOINT", str(backbone))
    monkeypatch.setattr(replay, "mit_huge", lambda **kwargs: TinyEncoder())
    monkeypatch.setattr("lim_worker.mit.load_encoder", fake_encoder_load)

    save(head)
    context = replay.load_for_eval(root, device="cpu")
    assert context.checkpoint_kind == "head_only"
    assert torch.all(context.model.encoder.conv.weight == 3.0), "encoder comes from LIM_CHECKPOINT"
    head_key = next(iter(head))
    assert torch.all(context.model.state_dict()[head_key] == 0.5), "head comes from head_only.pt"

    # Only encoder keys may be missing: a head key gone is still an error.
    save({key: value for key, value in head.items() if key != head_key})
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        replay.load_for_eval(root, device="cpu")
    # ... and so is an unexpected key.
    save(head | {"stray.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        replay.load_for_eval(root, device="cpu")


def test_full_checkpoints_still_require_every_key() -> None:
    with pytest.raises(ValueError, match="detector.backbone.conv.weight"):
        replay.check_loaded_keys(["detector.backbone.conv.weight"], [], "latest")
    replay.check_loaded_keys(["detector.backbone.conv.weight"], [], "head_only")


# ------------------------------------------------------------------- overlay


def test_boxes_line_up_with_the_image_under_the_caption(tmp_path, monkeypatch) -> None:
    """Review 1b3ceae §3: the caption strip used to shift every box up 30 px."""
    origin = (512, 256)  # a non-zero window origin
    local = np.array([[10.0, 40.0], [50.0, 40.0], [50.0, 60.0], [10.0, 60.0]])
    polygon = local + np.array(origin, dtype=float)
    detection = replay.Detection(polygon=polygon, score=0.9, class_id=0)
    match = replay.SceneMatch(
        detections=[detection],
        statuses=["tp"],
        targets=[(polygon, 0)],
        matched_targets={0},
        counts={"tp": 1, "fp": 0, "fn": 0, "gt": 1, "pred": 1},
    )
    monkeypatch.setattr(
        replay, "read_rgb", lambda *args, **kwargs: np.full((64, 64, 3), 80, dtype=np.uint8)
    )
    output = tmp_path / "overlay.png"
    replay.render_window(
        type("Context", (), {"data_root": tmp_path})(),
        {"window": [*origin, 64, 64], "image_paths": ["unused.png"]},
        match,
        output,
    )
    image = Image.open(output).convert("RGB")
    header = replay.PANEL_HEADER_PX
    # top edge of the GT box, away from the class label at its corner
    assert image.getpixel((40, 40 + header)) == GREEN
    assert image.getpixel((40, 40)) == (80, 80, 80)
    # the prediction panel sits below the GT panel and uses the same transform
    pred_top = 64 + header
    assert image.getpixel((40, pred_top + 40 + header)) == GREEN
    assert image.getpixel((40, pred_top + 40)) == (80, 80, 80)


# ------------------------------------------------------------------ identity


def test_matching_identity_is_an_original_protocol_replay(tmp_path, generator) -> None:
    root = _make_run(tmp_path)
    assert _run(generator, tmp_path) == 0
    gallery = json.loads((root / "visualization" / "manifest.json").read_text())
    replay_identity = gallery["identity"]["replay"]
    assert replay_identity["mode"] == "original_protocol"
    assert replay_identity["drift"] == []
    assert replay_identity["env_at_run"]["torch_version"] == str(torch.__version__)


def test_data_drift_refuses_original_and_is_published(tmp_path, generator) -> None:
    root = _make_run(tmp_path)
    record = json.loads((root / "run.json").read_text())
    record["dataset"]["manifest_sha256"] = "0" * 64
    (root / "run.json").write_text(json.dumps(record))

    assert _run(generator, tmp_path) == 3
    assert not (root / "visualization").exists()

    assert _run(generator, tmp_path, "--mode", "reevaluate") == 0
    gallery = json.loads((root / "visualization" / "manifest.json").read_text())
    data = gallery["identity"]["data"]
    assert data["manifest_sha256"] == _sha(root / "data.jsonl")
    assert data["manifest_sha256_recorded"] == "0" * 64
    assert gallery["identity"]["replay"]["mode"] == "reevaluated"
    assert "data manifest" in gallery["identity"]["replay"]["reason"]


def test_training_environment_comes_from_the_checkpoint(tmp_path, generator) -> None:
    root = _make_run(tmp_path)
    _rewrite_identity(root, torch_version="2.4.0")
    assert _run(generator, tmp_path) == 3
    assert _run(generator, tmp_path, "--mode", "reevaluate") == 0
    gallery = json.loads((root / "visualization" / "manifest.json").read_text())
    assert gallery["identity"]["replay"]["env_at_run"]["torch_version"] == "2.4.0"
    assert any("torch_version" in item for item in gallery["identity"]["replay"]["drift"])


def test_missing_training_identity_fails_closed(tmp_path, generator) -> None:
    root = _make_run(tmp_path)
    saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
    del saved["identity"]
    torch.save(saved, root / "latest.pt")
    assert _run(generator, tmp_path) == 3


# ---------------------------------------------------------- GPU and the lock


def test_a_second_generator_is_refused(tmp_path, generator) -> None:
    _make_run(tmp_path)
    held = generator._acquire_lock(tmp_path)
    try:
        assert _run(generator, tmp_path) == 4
    finally:
        held.close()
    assert _run(generator, tmp_path) == 0


def test_busy_gpu_is_refused_before_the_model_is_built(tmp_path, generator, monkeypatch) -> None:
    _make_run(tmp_path)
    monkeypatch.setattr(generator, "_gpu_used_mib", lambda: 9000)

    def must_not_load(*args, **kwargs):
        raise AssertionError("the model was built on a busy GPU")

    monkeypatch.setattr(replay, "load_for_eval", must_not_load)
    _rewrite_identity(tmp_path / "synthetic", device_type="cuda")
    assert (
        generator.main(
            [
                "--runs-root",
                str(tmp_path),
                "--run-id",
                "synthetic",
                "--device",
                "cuda",
            ]
        )
        == 5
    )
