"""Local dry-run for the Colab dispatch helpers (no GPU, no network).

Review 2026-11-04 round 3 required the notebook path to be reviewable and
dry-runnable: a one-patch fake HF layout exercises verification, the copy-back
target and the run-id shape that the real A100 cell will use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

LIB = Path(__file__).resolve().parents[1] / "colab" / "dispatch_lib.py"

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("colab_dispatch_lib", LIB)
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)


def fake_pastis_hd(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """One-patch fake HF layout: PASTIS-HD/DATA_S2/S2_1.npy + sha list."""
    pastis_hd = tmp_path / "derived" / "aef-2019" / "PASTIS-HD"
    data_dir = pastis_hd / "DATA_S2"
    data_dir.mkdir(parents=True)
    stack = np.zeros((2, 10, 8, 8), dtype=np.int16)
    np.save(data_dir / "S2_1.npy", stack)
    digest = hashlib.sha256((data_dir / "S2_1.npy").read_bytes()).hexdigest()
    return pastis_hd, {"S2_1.npy": digest}


def test_verify_s2_matches_the_hf_layout(tmp_path) -> None:
    """The download keeps DATA_S2/ inside local_dir; verification must look
    there (the old script checked the download dir itself and failed 100%)."""
    pastis_hd, hashes = fake_pastis_hd(tmp_path)
    lib.verify_s2(pastis_hd, hashes)  # passes


def test_verify_s2_rejects_missing_and_corrupt(tmp_path) -> None:
    pastis_hd, hashes = fake_pastis_hd(tmp_path)
    (pastis_hd / "DATA_S2" / "S2_1.npy").unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        lib.verify_s2(pastis_hd, hashes)
    pastis_hd, hashes = fake_pastis_hd(tmp_path / "second")
    (pastis_hd / "DATA_S2" / "S2_1.npy").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="corrupt"):
        lib.verify_s2(pastis_hd, hashes)
    with pytest.raises(SystemExit, match="empty"):
        lib.verify_s2(pastis_hd, {})


def test_run_ids_match_the_ledger_shape() -> None:
    assert lib.run_id_for("lim_s2_single", 17) == "lim_mit_huge-T11-lim_s2_single-train-17-a100"
    assert lib.run_id_for("lim_s2_20bin", 73) == "lim_mit_huge-T11-lim_s2_20bin-train-73-a100"


def test_dispatch_env_pins_every_root(tmp_path) -> None:
    env = lib.dispatch_env(
        {"PATH": "/usr/bin", "HOME": "/root"},
        repo=tmp_path / "repo",
        pack=tmp_path / "pack",
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        manifest=tmp_path / "m.jsonl",
        checkpoint=tmp_path / "w.pth",
    )
    for name in lib.REQUIRED_ENV:
        assert env[name]
    assert env["RS_EVAL_CONFIG_ROOT"] == (tmp_path / "repo" / "configs").as_posix()
    assert "RS_EVAL_SKIP_ENV" not in env  # no-op for dispatch; never relied on


def test_collect_run_uses_a_per_run_target(tmp_path) -> None:
    """Two consecutive runs must not collide on the shared parent."""
    runs_root = tmp_path / "runs"
    notebook_runs = tmp_path / "drive" / "runs"
    for index in (1, 2):
        run_id = f"lim_mit_huge-T11-lim_s2_single-train-{index}-a100"
        source = runs_root / run_id
        source.mkdir(parents=True)
        (source / "run.json").write_text(json.dumps({"run_id": run_id}))
        lib.collect_run(source, notebook_runs, run_id)
    assert (notebook_runs / "lim_mit_huge-T11-lim_s2_single-train-1-a100" / "run.json").is_file()
    assert (notebook_runs / "lim_mit_huge-T11-lim_s2_single-train-2-a100" / "run.json").is_file()
    with pytest.raises(FileExistsError):
        lib.collect_run(
            runs_root / "lim_mit_huge-T11-lim_s2_single-train-1-a100",
            notebook_runs,
            "lim_mit_huge-T11-lim_s2_single-train-1-a100",
        )


def test_a100_assertion_accepts_only_the_40gb_class() -> None:
    lib.assert_a100_40gb("NVIDIA A100-SXM4-40GB")
    with pytest.raises(SystemExit):
        lib.assert_a100_40gb("NVIDIA GeForce RTX 4090")
