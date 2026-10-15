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


def run_script(tmp_path: Path, uv_exit: int, logdir: Path) -> subprocess.CompletedProcess:
    binary, home = stub_bin(tmp_path, uv_exit)
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "HOME": str(home),
        "RS_EVAL_RERUN_LOGDIR": str(logdir),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "T1A"],
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
