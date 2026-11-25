"""CPU checks for the DOFA F3 injector (LIM worker env).

The LIM is the fusion test's pyramid, the DOFA encoder is a stride-16 stub, and the
checkpoint is a stand-in. The step-0 and update guards run as in production.

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_dofa.py -q
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker import t13_dofa_eval, t13_dofa_train  # noqa: E402
from lim_worker.t13_fusion import FT_B0_BASELINES  # noqa: E402
from test_lim_t13_fusion_train import (  # noqa: E402
    BASELINE_RUN,
    _dataset,
    _Pyramid,
    _source_record,
    _Writer,
)
from torch import nn  # noqa: E402


class _DofaStub(nn.Module):
    """Stride-16 stand-in for DOFA: [B, 1, H, W] -> [B, 768, H/16, W/16]."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 768, kernel_size=16, stride=16)

    def forward_map(self, image: torch.Tensor, wave_um: list[float]) -> torch.Tensor:
        return self.conv(image)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _dataset(tmp_path)
    return tmp_path


def _checkpoint(root: Path) -> Path:
    path = root / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stand-in")
    return path


def _train(root: Path, monkeypatch, name: str) -> _Writer:
    checkpoint = _checkpoint(root)
    monkeypatch.setenv("LIM_WORKER_MICRO_BATCH", "2")
    monkeypatch.setenv("LIM_WORKER_LIMIT_BATCHES", "0")
    monkeypatch.setenv("LIM_WORKER_EVAL_IMAGES", "0")
    monkeypatch.setenv("LIM_WORKER_NUM_WORKERS", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(t13_dofa_train, "mit_huge", lambda in_chans=3, **kw: _Pyramid())
    monkeypatch.setattr(t13_dofa_train, "find_full_dependency", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(t13_dofa_train, "load_full_checkpoint", lambda *a, **kw: {})
    monkeypatch.setattr(t13_dofa_train, "load_dofa", lambda: _DofaStub())
    request = SimpleNamespace(
        run_id=name,
        seed=17,
        data_root=str(root),
        runs_root=str(root / "runs"),
        manifest_path=str(root / "manifest.jsonl"),
    )
    protocol = SimpleNamespace(
        protocol_id=t13_dofa_train.DOFA_PROTOCOL,
        epochs=2,
        learning_rate=1e-3,
        backbone_learning_rate=1e-4,
        lr_schedule=None,
    )
    (root / "runs" / name).mkdir(parents=True, exist_ok=True)
    writer = _Writer(root / "runs" / name)
    code = t13_dofa_train.train_dofa(
        request,
        writer,
        lambda *a, **kw: None,
        protocol=protocol,
        task_spec=None,
        model_id="lim_mit_huge",
    )
    assert code == 0
    return writer


def test_dofa_training_exports_the_projection_and_the_head(dataset, monkeypatch) -> None:
    writer = _train(dataset, monkeypatch, "dofa-trained")
    plugin = torch.load(writer.run_dir / "dofa_plugin.pt", weights_only=False)
    assert plugin["format"] == t13_dofa_train.DOFA_FORMAT
    assert plugin["config"]["dofa"] is True and plugin["config"]["head"] is True
    assert set(plugin["state_dict"]) == {"dofa.proj.weight", "dofa.proj.bias"}
    assert bool(plugin["state_dict"]["dofa.proj.weight"].abs().max() > 0)
    assert plugin["head_state"], "the trained UPerNet head must be exported"
    assert writer.tags["test_evaluated"] == "no"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "x"}, "plugin format"),
        ({"protocol": "whu_ft_nir"}, "was trained for"),
        ({"config": {"dofa": True, "head": False, "level": 2}}, "not the DOFA F3"),
        ({"head_state": None}, "no head weights"),
    ],
)
def test_foreign_dofa_plugin_is_refused(overrides, message) -> None:
    pin = FT_B0_BASELINES[17]
    plugin = {
        "format": t13_dofa_train.DOFA_FORMAT,
        "protocol": t13_dofa_train.DOFA_PROTOCOL,
        "baseline_run": pin["run_id"],
        "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
        "manifest_sha256": "abc",
        "config": {"dofa": True, "head": True, "level": 2},
        "head_state": {"x": torch.zeros(1)},
    }
    plugin.update(overrides)
    with pytest.raises(RuntimeError, match=message):
        t13_dofa_eval.check_dofa_plugin(plugin, pin=pin, manifest_sha="abc")


def test_locked_dofa_evaluate_scores_the_test_split_once(dataset, monkeypatch) -> None:
    _train(dataset, monkeypatch, "dofa-trained")
    _source_record(dataset / "runs" / "dofa-trained", artifact="dofa_plugin.pt")
    checkpoint = dataset / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    monkeypatch.setattr(t13_dofa_eval, "mit_huge", lambda in_chans=3, **kw: _Pyramid())
    monkeypatch.setattr(t13_dofa_eval, "find_full_dependency", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(t13_dofa_eval, "load_full_checkpoint", lambda *a, **kw: {})
    monkeypatch.setattr(t13_dofa_eval, "load_dofa", lambda: _DofaStub())
    monkeypatch.setenv("LIM_WORKER_PLUGIN_RUN", "dofa-trained")
    monkeypatch.setenv("LIM_WORKER_EVAL_IMAGES", "0")
    monkeypatch.setenv("LIM_WORKER_NUM_WORKERS", "0")
    request = SimpleNamespace(
        run_id="dofa-evaluated",
        seed=17,
        data_root=str(dataset),
        runs_root=str(dataset / "runs"),
        manifest_path=str(dataset / "manifest.jsonl"),
    )
    (dataset / "runs" / "dofa-evaluated").mkdir(parents=True, exist_ok=True)
    writer = _Writer(dataset / "runs" / "dofa-evaluated")
    code = t13_dofa_eval.evaluate_dofa(
        request,
        writer,
        lambda *a, **kw: None,
        protocol=SimpleNamespace(protocol_id=t13_dofa_train.DOFA_PROTOCOL, epochs=2),
        task_spec=None,
        model_id="lim_mit_huge",
    )
    assert code == 0
    assert writer.tags["test_evaluated"] == "yes"
    assert writer.tags["plugin_run"] == "dofa-trained"
    names = {record.name for record in writer.metrics}
    assert {"mean_iou", "boundary_iou"} <= names
