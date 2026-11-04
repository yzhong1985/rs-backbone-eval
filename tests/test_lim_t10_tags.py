"""T10 run tags must identify the GPU class (review 2026-11-05).

T10 v8 now runs on two hosts (4090 and the 5090 night node). Without a host /
GPU tag the ledger cannot tell which machine produced a record, and the
"one cell == one GPU class" audit would be unverifiable. Needs torch, so the
root environment skips this module; run it in the LIM worker environment:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:workers/lim/src \\
        workers/lim/.venv/bin/python -m pytest tests/test_lim_t10_tags.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
t10 = pytest.importorskip("lim_worker.t10")


def test_cpu_fallback_is_tagged(monkeypatch) -> None:
    monkeypatch.setattr(t10.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("RS_EVAL_REQUIRE_DEVICE", raising=False)
    device, tags = t10._training_device()
    assert device == "cpu"
    assert tags["device"] == "cpu"
    assert tags["host"] == "cpu"
    assert tags["torch"]


def test_gpu_name_is_the_host_tag(monkeypatch) -> None:
    monkeypatch.setattr(t10.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        t10.torch.cuda, "get_device_name", lambda index=0: "NVIDIA GeForce RTX 5090"
    )
    monkeypatch.delenv("RS_EVAL_REQUIRE_DEVICE", raising=False)
    device, tags = t10._training_device()
    assert device == "cuda"
    assert tags["device"] == "cuda"
    assert tags["host"] == "nvidia-geforce-rtx-5090"


def test_declared_gpu_cell_refuses_cpu(monkeypatch) -> None:
    monkeypatch.setattr(t10.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("RS_EVAL_REQUIRE_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="refusing to train on CPU"):
        t10._training_device()
