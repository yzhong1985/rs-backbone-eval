"""Tests for scripts/rerun_baselines.sh.

The script must create its log directory, and it must exit non-zero when any
dispatch fails instead of reporting success.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rerun_baselines.sh"


def stub_bin(tmp_path: Path, uv_exit: int) -> tuple[Path, Path]:
    """A PATH with fake nvidia-smi (GPU free) and fake uv; HOME without a real uv."""
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, body in {
        "nvidia-smi": "#!/bin/sh\necho 0\n",
        "uv": f"#!/bin/sh\necho 'dispatch rejected: stub' >&2\nexit {uv_exit}\n",
    }.items():
        path = binary / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"
    home.mkdir()
    return binary, home


def run_script(
    tmp_path: Path, uv_exit: int, logdir: Path, task: str = "T1A"
) -> subprocess.CompletedProcess:
    binary, home = stub_bin(tmp_path, uv_exit)
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "HOME": str(home),
        "RS_EVAL_RERUN_LOGDIR": str(logdir),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), task],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_rerun_script_exits_nonzero_when_every_dispatch_fails(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"  # deliberately missing: the script must create it
    result = run_script(tmp_path, uv_exit=1, logdir=logdir)

    assert result.returncode != 0, result.stdout
    assert logdir.is_dir()
    log = (logdir / "rerun-T1A-baselines.out").read_text(encoding="utf-8")
    assert log.count(" FAIL ") == 3
    assert "3 failed" in log


def test_rerun_script_reports_success_when_dispatch_succeeds(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir)

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T1A-baselines.out").read_text(encoding="utf-8")
    assert log.count(" OK ") == 3
    assert "3 ok, 0 failed" in log


def test_olmoearth_case_dispatches_the_frozen_adapter_seeds(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="O")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-O-baselines.out").read_text(encoding="utf-8")
    assert "3 ok, 0 failed" in log
    # the worker rejects everything but frozen_adapter, so the script must not
    # queue frozen_probe for OlmoEarth
    assert "olmoearth_v1_2_tiny-T1-frozen_adapter-42-v3" in log
    assert "frozen_probe" not in log


def test_unknown_task_is_rejected_with_usage(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T9")

    assert result.returncode == 2, result.stdout


def test_b1_case_dispatches_full_finetune_for_three_models(tmp_path: Path) -> None:
    """Cheap GPUs run prithvi/clay; the AnySat cell has its own B1A case."""
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="B1")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-B1-baselines.out").read_text(encoding="utf-8")
    assert "6 ok, 0 failed" in log
    assert log.count("OK prithvi_eo_v2_300_tl-T1-full_finetune") == 3
    assert log.count("OK clay_v1_5-T1-full_finetune") == 3
    assert "OK anysat" not in log


def test_b1a_case_dispatches_anysat_full_finetune_natively(tmp_path: Path) -> None:
    """B1A queues the AnySat T1 cell with the native batch of 2 (no accumulation)."""
    binary, home = stub_bin(tmp_path, 0)
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "HOME": str(home),
        "RS_EVAL_RERUN_LOGDIR": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "B1A"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    log = ((tmp_path / "logs") / "rerun-B1A-baselines.out").read_text(encoding="utf-8")
    assert "3 ok, 0 failed" in log
    assert log.count("OK anysat_base-T1-full_finetune") == 3


def test_t5b_case_queues_the_remaining_single_seed(tmp_path: Path) -> None:
    """T5B hands one seed to the 4090 when another seed is offloaded to Colab."""
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T5B")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T5B-baselines.out").read_text(encoding="utf-8")
    assert "1 ok, 0 failed" in log
    assert log.count("START anysat_base-T5-frozen_probe-73-") == 1
    assert "-42-" not in log and "-17-" not in log


def test_t2_case_queues_the_frozen_baselines(tmp_path: Path) -> None:
    """Nine runs: prithvi/clay under frozen_probe, anysat under frozen_adapter."""
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T2")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T2-baselines.out").read_text(encoding="utf-8")
    assert "9 ok, 0 failed" in log
    assert log.count("OK prithvi_eo_v2_300_tl-T2-frozen_probe") == 3
    assert log.count("OK clay_v1_5-T2-frozen_probe") == 3
    assert log.count("OK anysat_base-T2-frozen_adapter") == 3


def test_t5a_queue_runs_full_micro_batch_for_anysat(tmp_path: Path) -> None:
    """The batch-2 probe OOMed on T5, so the cell keeps the 1x2 accumulation."""
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T5A")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T5A-baselines.out").read_text(encoding="utf-8")
    assert "3 ok, 0 failed" in log
    assert log.count("OK anysat_base-T5-frozen_probe") == 3


def test_b0_case_dispatches_the_reference_sweep(tmp_path: Path) -> None:
    """11 explicit-identity runs: 3 seeds x (official/1e-4/2e-4) plus repo x2."""
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="B0")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-B0-baselines.out").read_text(encoding="utf-8")
    assert "11 ok, 0 failed" in log
    assert log.count("OK prithvi_eo_v2_300_tl-T1-prithvi_paper_protocol-train-42") == 3
    assert log.count("OK prithvi_eo_v2_300_tl-T1-prithvi_paper_protocol") == 9
    assert log.count("OK prithvi_eo_v2_300_tl-T1-prithvi_repo_config") == 2
    assert "START" in log and "OK prithvi_eo_v2_300_tl-T1-prithvi_repo_config-probe" in log


def test_rerun_suffix_is_honored(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    binary, _ = stub_bin(tmp_path, 0)
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "RS_EVAL_RERUN_LOGDIR": str(logdir),
        "RS_EVAL_RERUN_SUFFIX": "v4",
    }
    (tmp_path / "home").mkdir(exist_ok=True)
    result = subprocess.run(
        ["bash", str(SCRIPT), "O"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-O-baselines.out").read_text(encoding="utf-8")
    assert "olmoearth_v1_2_tiny-T1-frozen_adapter-42-v4" in log
