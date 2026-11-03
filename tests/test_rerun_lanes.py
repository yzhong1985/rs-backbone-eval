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


def test_t10b2_keeps_its_versioned_budget() -> None:
    block = lane_block("T10B2")
    assert "-v8" in block
    assert "IM_WORKER_TOTAL_STEPS" in block or "RS_EVAL_T10B2_SEEDS" in block
