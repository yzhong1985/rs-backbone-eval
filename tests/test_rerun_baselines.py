"""Tests for scripts/rerun_baselines.sh.

The script must create its log directory, and it must exit non-zero when any
dispatch fails instead of reporting success.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rerun_baselines.sh"


def stub_bin(tmp_path: Path, uv_exit: int, uv_sleep: float = 0) -> tuple[Path, Path]:
    """A PATH with fake nvidia-smi (GPU free) and fake uv; HOME without a real uv."""
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, body in {
        "nvidia-smi": "#!/bin/sh\necho 0\n",
        "uv": (
            f"#!/bin/sh\n[ {uv_sleep} != 0 ] && sleep {uv_sleep}\n"
            f"echo 'dispatch rejected: stub' >&2\nexit {uv_exit}\n"
        ),
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
        # keep the completed-run guard off the real ledger
        "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
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


def test_t3c_case_queues_anysat_t3_single_seed(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T3C")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T3C-baselines.out").read_text(encoding="utf-8")
    assert "1 ok, 0 failed" in log
    assert log.count("START anysat_base-T3-full_finetune-42-") == 1


def test_t4c_case_queues_prithvi_and_anysat_landslide_cells(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T4C")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T4C-baselines.out").read_text(encoding="utf-8")
    assert "2 ok, 0 failed" in log
    assert log.count("START prithvi_eo_v2_300_tl-T4-full_finetune-42-") == 1
    assert log.count("START anysat_base-T4-full_finetune-42-") == 1


def test_t5c_case_hands_the_protocol_seeds_to_a_local_queue(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    binary, home = stub_bin(tmp_path, 0)
    calls = tmp_path / "dispatch.jsonl"
    (binary / "uv").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RS_EVAL_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:], 'env': {k: os.environ.get(k) "
        "for k in ['ANYSAT_T5_STAGE', 'ANYSAT_FROZEN_CACHE', "
        "'ANYSAT_WORKER_BATCH_SIZE', 'ANYSAT_WORKER_GRAD_ACCUM']}}) + '\\n')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "T5C"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "HOME": str(home),
            "RS_EVAL_RERUN_LOGDIR": str(logdir),
            "RS_EVAL_TEST_CALLS": str(calls),
            "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T5C-baselines.out").read_text(encoding="utf-8")
    assert "3 ok, 0 failed" in log
    assert log.count("START anysat_base-T5-frozen_adapter_two_stage-17-") == 1
    assert log.count("START anysat_base-T5-frozen_adapter_two_stage-42-") == 1
    assert log.count("START anysat_base-T5-frozen_adapter_two_stage-73-") == 1
    dispatched = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(dispatched) == 3
    for call, seed in zip(dispatched, ["17", "42", "73"], strict=True):
        args = call["args"]
        assert args[args.index("--task") + 1] == "T5"
        assert args[args.index("--protocol") + 1] == "frozen_adapter_two_stage"
        assert args[args.index("--seed") + 1] == seed
        assert call["env"] == {
            "ANYSAT_T5_STAGE": "two",
            "ANYSAT_FROZEN_CACHE": "1",
            "ANYSAT_WORKER_BATCH_SIZE": "1",
            "ANYSAT_WORKER_GRAD_ACCUM": "2",
        }


def test_t5c_seed_override_runs_only_the_missing_seed(tmp_path: Path) -> None:
    """A top-up must dispatch just the seed asked for, not re-list finished ones."""
    logdir = tmp_path / "logs"
    binary, home = stub_bin(tmp_path, 0)
    calls = tmp_path / "dispatch.jsonl"
    (binary / "uv").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RS_EVAL_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:]}) + '\\n')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "T5C"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "HOME": str(home),
            "RS_EVAL_RERUN_LOGDIR": str(logdir),
            "RS_EVAL_TEST_CALLS": str(calls),
            "RS_EVAL_T5C_SEEDS": "42",
            "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T5C-baselines.out").read_text(encoding="utf-8")
    assert "1 ok, 0 failed" in log
    assert "START anysat_base-T5-frozen_adapter_two_stage-42-" in log
    assert "START anysat_base-T5-frozen_adapter_two_stage-17-" not in log
    assert "START anysat_base-T5-frozen_adapter_two_stage-73-" not in log


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


def test_detached_mode_survives_outside_the_launching_session(tmp_path: Path) -> None:
    """--detached re-execs under setsid; the launcher returns at once and the
    chain runs on in its own session (PID file written, then cleaned on exit)."""
    import time

    logdir = tmp_path / "logs"
    binary, home = stub_bin(tmp_path, uv_exit=0)
    # Hold the stub dispatch until PID/SID/PGID are checked; otherwise a fast
    # successful chain could remove its PID file before the assertions run.
    release = tmp_path / "release"
    (binary / "uv").write_text(
        '#!/bin/sh\nwhile [ ! -f "$RS_EVAL_TEST_RELEASE" ]; do sleep 0.05; done\n',
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "HOME": str(home),
        "RS_EVAL_RERUN_LOGDIR": str(logdir),
        "RS_EVAL_TEST_RELEASE": str(release),
        "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "--detached", "T1A"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    pidfile = logdir / "rerun-T1A-baselines.pid"
    try:
        assert result.returncode == 0, result.stdout
        assert "detached; PID" in result.stdout
        assert pidfile.is_file(), "PID file not written"
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        assert os.getsid(pid) == pid, "PID must identify the detached session leader"
        assert os.getpgid(pid) == pid, "PID must identify the group used by kill -- -PID"
        assert f"PID {pid} " in result.stdout
    finally:
        release.touch()

    # the detached child runs in its own session; poll for it to finish
    log = logdir / "rerun-T1A-baselines.out"
    for _ in range(120):
        text = log.read_text(encoding="utf-8") if log.is_file() else ""
        if "3 ok, 0 failed" in text:
            break
        time.sleep(0.5)
    else:
        raise AssertionError(f"detached chain never reported completion: {text!r}")

    # the trap removes the PID file on a clean exit
    for _ in range(40):
        if not pidfile.is_file():
            break
        time.sleep(0.5)
    assert not pidfile.is_file(), "PID file not cleaned by the exit trap"


def test_completed_run_is_not_clobbered(tmp_path: Path) -> None:
    """A shared suffix must not overwrite a finished run (the v3 accident)."""
    logdir = tmp_path / "logs"
    runs = tmp_path / "runs"
    rid = "anysat_base-T2-frozen_adapter-17-v3"
    (runs / rid).mkdir(parents=True)
    (runs / rid / "run.json").write_text('{"status": "completed"}', encoding="utf-8")

    binary, home = stub_bin(tmp_path, uv_exit=0)
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "HOME": str(home),
        "RS_EVAL_RERUN_LOGDIR": str(logdir),
        "RS_EVAL_RUNS_ROOT": str(runs),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT), "T2A"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = (logdir / "rerun-T2A-baselines.out").read_text(encoding="utf-8")
    assert result.returncode != 0, result.stdout
    assert (runs / rid / "run.json").read_text() == '{"status": "completed"}'
    assert "SKIP anysat_base-T2-frozen_adapter-17-v3" in log
    # the 42/73 seeds still dispatch; only the completed 17-v3 is skipped
    assert "START anysat_base-T2-frozen_adapter-42-v3" in log


def _lr_stub(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Stub uv that records each dispatch's id and the per-worker lr env."""
    binary, home = stub_bin(tmp_path, 0)
    calls = tmp_path / "dispatch.jsonl"
    (binary / "uv").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['RS_EVAL_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'run_id': args[args.index('--run-id') + 1], 'model': "
        "args[args.index('--model') + 1], 'task': args[args.index('--task') + 1], "
        "'lr': {k: os.environ.get(k) for k in ['ANYSAT_WORKER_LR', 'CLAY_WORKER_LR', "
        "'PRITHVI_WORKER_LR']}, 'epochs': {k: os.environ.get(k) for k in "
        "['ANYSAT_WORKER_EPOCHS', 'CLAY_WORKER_EPOCHS', 'PRITHVI_WORKER_EPOCHS']}}) "
        "+ '\\n')\n",
        encoding="utf-8",
    )
    return binary, home, calls


def test_t3b1_uses_each_backbones_declared_lr(tmp_path: Path) -> None:
    """B1 lr is per backbone (T1 search) and must not leak between models."""
    binary, home, calls = _lr_stub(tmp_path)
    logdir = tmp_path / "logs"
    result = subprocess.run(
        ["bash", str(SCRIPT), "T3B1"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "HOME": str(home),
            "RS_EVAL_RERUN_LOGDIR": str(logdir),
            "RS_EVAL_TEST_CALLS": str(calls),
            "RS_EVAL_RERUN_SUFFIX": "v5",
            "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    dispatched = [json.loads(line) for line in calls.read_text().splitlines()]
    # three Clay T1 lr points (the audit's candidate check) then the T3 cell
    assert len(dispatched) == 12
    grid = dispatched[:3]
    assert {call["task"] for call in grid} == {"T1"}
    assert [call["lr"]["CLAY_WORKER_LR"] for call in grid] == ["1e-4", "3e-4", "3e-5"]
    assert all("-lr" in call["run_id"] for call in grid)
    t3 = dispatched[3:]
    assert {call["task"] for call in t3} == {"T3"}
    # T3 keeps the repo budget (50 epochs), so no epoch override is exported
    for call in t3:
        assert not any(call["epochs"].values())
    log = (logdir / "rerun-T3B1-baselines.out").read_text(encoding="utf-8")
    assert "12 ok, 0 failed" in log
    expected = {"anysat_base": 1e-4, "clay_v1_5": 3e-4, "prithvi_eo_v2_300_tl": 3e-5}
    for call in t3:
        assert call["run_id"].endswith("-v5")
        own = {k: v for k, v in call["lr"].items() if v}
        assert len(own) == 1, own  # exactly one worker lr is exported
        assert float(next(iter(own.values()))) == expected[call["model"]]


def test_t5b1_runs_only_the_prithvi_cell(tmp_path: Path) -> None:
    """AnySat/Clay cannot do a 125-epoch three-date full fine-tune on the 4090."""
    binary, home, calls = _lr_stub(tmp_path)
    logdir = tmp_path / "logs"
    result = subprocess.run(
        ["bash", str(SCRIPT), "T5B1"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "HOME": str(home),
            "RS_EVAL_RERUN_LOGDIR": str(logdir),
            "RS_EVAL_TEST_CALLS": str(calls),
            "RS_EVAL_RERUN_SUFFIX": "v5",
            "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout
    dispatched = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(dispatched) == 3
    assert {call["model"] for call in dispatched} == {"prithvi_eo_v2_300_tl"}
    log = (logdir / "rerun-T5B1-baselines.out").read_text(encoding="utf-8")
    assert "3 ok, 0 failed" in log


def test_t4dem_lane_runs_three_band_sets_and_records_the_env(tmp_path: Path) -> None:
    """T4's DEM ablation must be reproducible from the log alone (2026-10-28)."""
    logdir = tmp_path / "logs"
    binary, home = stub_bin(tmp_path, 0)
    calls = tmp_path / "dispatch.jsonl"
    (binary / "uv").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RS_EVAL_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:], "
        "'bands': os.environ.get('CLAY_WORKER_T4_BANDS')}) + '\\n')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "T4DEM"],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "HOME": str(home),
            "RS_EVAL_RERUN_LOGDIR": str(logdir),
            "RS_EVAL_TEST_CALLS": str(calls),
            "RS_EVAL_RUNS_ROOT": str(tmp_path / "runs"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T4DEM-baselines.out").read_text(encoding="utf-8")
    assert "9 ok, 0 failed" in log
    for condition in ("opt", "dem", "ds"):
        assert log.count(f"START clay_v1_5-T4-frozen_probe-17-t4dem-{condition}") == 1
    sets = [json.loads(line)["bands"] for line in calls.read_text().splitlines()]
    assert any(entry.endswith(",DEM") and "SLOPE" not in entry for entry in sets)
    assert any(entry.endswith(",DEM,SLOPE") for entry in sets)


def test_t4b1c_fills_only_the_clay_half_of_the_cell(tmp_path: Path) -> None:
    logdir = tmp_path / "logs"
    result = run_script(tmp_path, uv_exit=0, logdir=logdir, task="T4B1C")

    assert result.returncode == 0, result.stdout
    log = (logdir / "rerun-T4B1C-baselines.out").read_text(encoding="utf-8")
    assert "2 ok, 0 failed" in log
    assert "clay_v1_5-T4-full_finetune-42-v1" in log
    assert "clay_v1_5-T4-full_finetune-73-v1" in log
    # Prithvi's cell belongs to the other host; this lane must not touch it.
    assert "prithvi_eo_v2_300_tl-T4-full_finetune" not in log
