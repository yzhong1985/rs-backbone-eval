"""Budget/version guards on the rerun lane script (no worker is started).

The lanes drive long GPU jobs: a missing *_TOTAL_STEPS export silently runs the
worker's default epoch loop (up to 6x the protocol budget on T9), and a missing
run-id version suffix collides with the existing run directory at dispatch
(review 2026-11-03, round 4/5).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def lane_block(lane: str) -> str:
    text = (REPO / "scripts" / "rerun_baselines.sh").read_text(encoding="utf-8")
    start = text.index(f"    {lane})")
    end = text.index("    ;;", start)
    return text[start:end]


def test_t9b2c_sets_the_step_budget_and_versioned_ids() -> None:
    block = lane_block("T9B2C")
    assert "export CLAY_WORKER_TOTAL_STEPS=25000" in block
    assert "unset CLAY_WORKER_TOTAL_STEPS" in block
    assert "-v4" in block
    assert "for seed in 17 42 73; do" in block


def test_t9b2p_sets_the_step_budget_and_hashes_all_seeds() -> None:
    block = lane_block("T9B2P")
    assert "export PRITHVI_WORKER_TOTAL_STEPS=25000" in block
    assert "unset PRITHVI_WORKER_TOTAL_STEPS" in block
    assert "for seed in 17 42 73; do" in block
    assert "-v4" in block


def test_t11ba_pins_suffixed_run_ids_and_version_note() -> None:
    block = lane_block("T11BA")
    assert "RS_EVAL_T11BA_SUFFIX" in block and ":?" in block
    assert "aef_probe_mlp" in block and "aef_probe_upernet" in block
    assert "for seed in 17 42 73; do" in block
    assert "unsuffixed" in block  # the first batch's runs are not lane ids
    assert "export RS_EVAL_RERUN_CPU_ONLY=1" in block
    assert "export OMP_NUM_THREADS=8" in block


def test_cpu_only_flag_skips_the_gpu_wait_overall() -> None:
    """Grep-level guard: the wait loop must sit behind the CPU-only guard.

    Round-7 review: T11BA waited on the GPU (and exited 1 when no GPU was
    visible) although it is a pure-CPU lane."""
    text = Path(REPO / "scripts" / "rerun_baselines.sh").read_text(encoding="utf-8")
    wait_start = text.index(r"while true; do" + chr(10) + "        used=$(nvidia-smi")
    wait_end = text.index("GPU free; starting")
    # Flag is tested before the wait loop; the "CPU-only" message lives in the
    # else branch that skips the whole loop.
    before = text[:wait_start]
    assert 'if [ "${RS_EVAL_RERUN_CPU_ONLY:-0}" != "1" ]' in before
    assert text.index("CPU-only lane; skipping the GPU idle wait") > wait_end


def test_t10b2_keeps_its_versioned_budget() -> None:
    block = lane_block("T10B2")
    assert "-v8" in block
    assert "IM_WORKER_TOTAL_STEPS" in block or "RS_EVAL_T10B2_SEEDS" in block
