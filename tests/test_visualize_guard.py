"""Regression guards for the visualisation generator (review 2026-11-03)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    target = ROOT / "scripts/visualize_run.py"
    spec = importlib.util.spec_from_file_location("visualize_run", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_indexed_devices_need_the_idle_check() -> None:
    module = _module()
    assert module._is_cuda_device("cuda")
    assert module._is_cuda_device("cuda:0")
    assert module._is_cuda_device("cuda:1")
    assert not module._is_cuda_device("cpu")


def test_anchor_decode_weights_are_in_the_replay_fingerprint() -> None:
    module = _module()
    assert "src/rs_backbone_eval/detection/anchors.py" in module.INFERENCE_FILES
