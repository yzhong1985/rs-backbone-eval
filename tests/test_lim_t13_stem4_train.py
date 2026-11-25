"""CPU end-to-end checks for the frozen four-band first layer trainer.

A small pyramid with a real OverlapPatchEmbed stem stands in for MiT-Huge. The FT-B0
checkpoint is replaced by a fixed dependency path. Tiles, group-normalised CE,
validation, export guards and ledger calls run as in production.

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_stem4_train.py -q
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker import t13, t13_stem4_train  # noqa: E402
from lim_worker.mit import OverlapPatchEmbed  # noqa: E402
from test_lim_t13_fusion_train import BASELINE_RUN, _dataset, _Writer  # noqa: E402
from torch import nn  # noqa: E402


class _Stem4Pyramid(nn.Module):
    """Four levels at strides 4, 8, 16, 32 with a real four-band-capable stem."""

    def __init__(self) -> None:
        super().__init__()
        chans = t13.CHANNELS
        self.patch_embed1 = OverlapPatchEmbed(3, 4, 3, chans[0])  # 16x16 at 64 px
        self.stages = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], 2, stride=2) for i in range(3)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        tokens, height, width = self.patch_embed1(x)
        batch, _, channels = tokens.shape
        feat = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        outs = [feat]
        for stage in self.stages:
            feat = stage(feat)
            outs.append(feat)
        return outs


def _run(root: Path, monkeypatch, name: str, *, lr: float = 1e-3, epochs: int = 2):
    checkpoint = root / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"stand-in")
    monkeypatch.setenv("LIM_WORKER_MICRO_BATCH", "2")
    monkeypatch.setenv("LIM_WORKER_LIMIT_BATCHES", "0")
    monkeypatch.setenv("LIM_WORKER_EVAL_IMAGES", "0")
    monkeypatch.setenv("LIM_WORKER_NUM_WORKERS", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(t13_stem4_train, "mit_huge", lambda in_chans=3, **kw: _Stem4Pyramid())
    monkeypatch.setattr(t13_stem4_train, "find_full_dependency", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(t13_stem4_train, "load_full_checkpoint", lambda *a, **kw: {})
    request = SimpleNamespace(
        run_id=name,
        seed=17,
        data_root=str(root),
        runs_root=str(root / "runs"),
        manifest_path=str(root / "manifest.jsonl"),
    )
    protocol = SimpleNamespace(
        protocol_id="whu_ft_nir4", epochs=epochs, learning_rate=lr, lr_schedule=None
    )
    (root / "runs" / name).mkdir(parents=True, exist_ok=True)
    writer = _Writer(root / "runs" / name)
    code = t13_stem4_train.train_stem4(
        request,
        writer,
        lambda *a, **kw: None,
        protocol=protocol,
        task_spec=None,
        model_id="lim_mit_huge",
    )
    assert code == 0
    return writer


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _dataset(tmp_path)
    return tmp_path


def test_frozen_four_band_run_trains_only_the_nir_weights_and_exports(dataset, monkeypatch) -> None:
    writer = _run(dataset, monkeypatch, "stem4")
    assert writer.attached == dataset / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    plugin = torch.load(writer.run_dir / "stem4_plugin.pt")
    assert plugin["format"] == "t13-stem4-plugin-v1"
    assert set(plugin["state_dict"]) == {"nir.weight"}
    assert bool(plugin["state_dict"]["nir.weight"].abs().max() > 0)
    assert "test not evaluated" in writer.completed


def test_zero_learning_rate_is_refused_and_exports_nothing(dataset, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="no trained update"):
        _run(dataset, monkeypatch, "tiny-lr", lr=1e-30)
    assert not (dataset / "runs" / "tiny-lr" / "stem4_plugin.pt").exists()
