"""CPU checks for the locked evaluate of the four-band first layer (whu_ft_nir4).

The plugin refusals run on plain dicts. The end-to-end check trains a stand-in plugin
with the stem4 test helpers, then scores its test split once through the same stub.

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_stem4_eval.py -q
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker import t13_stem4_eval  # noqa: E402
from lim_worker.t13_fusion import FT_B0_BASELINES  # noqa: E402
from lim_worker.t13_stem4_train import STEM4_FORMAT  # noqa: E402
from test_lim_t13_fusion_train import BASELINE_RUN, _dataset, _Writer  # noqa: E402
from test_lim_t13_stem4_train import _run, _Stem4Pyramid  # noqa: E402

PIN = FT_B0_BASELINES[17]


def _plugin(**overrides) -> dict:
    plugin = {
        "format": STEM4_FORMAT,
        "protocol": "whu_ft_nir4",
        "baseline_run": PIN["run_id"],
        "baseline_checkpoint_sha256": PIN["checkpoint_sha256"],
        "manifest_sha256": "abc",
        "config": {"nir": True, "sar": False},
        "state_dict": {"nir.weight": torch.zeros(1)},
    }
    plugin.update(overrides)
    return plugin


def test_matching_plugin_passes_the_check() -> None:
    t13_stem4_eval.check_stem4_plugin(_plugin(), pin=PIN, manifest_sha="abc")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"format": "t13-fusion-plugin-v1"}, "plugin format"),
        ({"protocol": "whu_ft_nir"}, "was trained for"),
        ({"baseline_run": "some-other-run"}, "plugin baseline is"),
        ({"baseline_checkpoint_sha256": "0" * 64}, "checkpoint SHA-256"),
        ({"config": {"nir": True, "sar": True}}, "NIR-only"),
        ({"config": {"nir": False, "sar": False}}, "NIR-only"),
        ({"state_dict": {}}, "no nir.weight"),
    ],
)
def test_foreign_or_wrong_plugin_is_refused(overrides, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        t13_stem4_eval.check_stem4_plugin(_plugin(**overrides), pin=PIN, manifest_sha="abc")


def test_plugin_from_another_manifest_is_refused() -> None:
    with pytest.raises(RuntimeError, match="another manifest"):
        t13_stem4_eval.check_stem4_plugin(_plugin(), pin=PIN, manifest_sha="xyz")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _dataset(tmp_path)
    return tmp_path


def test_locked_stem4_evaluate_scores_the_test_split_once(dataset, monkeypatch) -> None:
    _run(dataset, monkeypatch, "stem4")
    checkpoint = dataset / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(t13_stem4_eval, "mit_huge", lambda in_chans=3, **kw: _Stem4Pyramid())
    monkeypatch.setattr(t13_stem4_eval, "find_full_dependency", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(t13_stem4_eval, "load_full_checkpoint", lambda *a, **kw: {})
    monkeypatch.setenv("LIM_WORKER_PLUGIN_RUN", "stem4")
    monkeypatch.setenv("LIM_WORKER_EVAL_IMAGES", "0")
    monkeypatch.setenv("LIM_WORKER_NUM_WORKERS", "0")
    monkeypatch.setenv("LIM_WORKER_MICRO_BATCH", "2")
    request = SimpleNamespace(
        run_id="evaluated",
        seed=17,
        data_root=str(dataset),
        runs_root=str(dataset / "runs"),
        manifest_path=str(dataset / "manifest.jsonl"),
    )
    protocol = SimpleNamespace(protocol_id="whu_ft_nir4")
    (dataset / "runs" / "evaluated").mkdir(parents=True, exist_ok=True)
    writer = _Writer(dataset / "runs" / "evaluated")
    code = t13_stem4_eval.evaluate_stem4(
        request,
        writer,
        lambda *a, **kw: None,
        protocol=protocol,
        task_spec=None,
        model_id="lim_mit_huge",
    )
    assert code == 0
    assert writer.tags["test_evaluated"] == "yes"
    assert writer.tags["plugin_run"] == "stem4"
    assert writer.tags["test_sheets"] == "1"
    names = {record.name for record in writer.metrics}
    assert {"mean_iou", "boundary_iou"} <= names
