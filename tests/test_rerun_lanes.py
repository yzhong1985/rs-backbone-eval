"""Budget/version guards on the rerun lane script (no worker is started).

The lanes drive long GPU jobs: a missing *_TOTAL_STEPS export silently runs the
worker's default epoch loop (up to 6x the protocol budget on T9), and a missing
run-id version suffix collides with the existing run directory at dispatch
(review 2026-11-03, round 4/5).
"""

import os
import subprocess
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
    assert "for seed in ${RS_EVAL_T11BA_SEEDS:-17 42 73}; do" in block
    assert "unsuffixed" in block  # the first batch's runs are not lane ids
    # the CPU-only gate moved to the top (CPU_LANES set), so the branch only
    # carries the thread cap any more
    assert "export RS_EVAL_RERUN_CPU_ONLY=1" not in block
    assert "export OMP_NUM_THREADS=8" in block


def test_t11bb_lane_is_a_cpu_cell() -> None:
    """Review round 2 (#1 reviewer 1): the lane must not silently grab CUDA.

    The lane clears CUDA_VISIBLE_DEVICES so the device-aware worker tags
    host=cpu for the whole cell; the A100 cell comes from the Colab
    dispatch, never from this lane."""
    block = lane_block("T11BB")
    assert 'export CUDA_VISIBLE_DEVICES=""' in block
    assert "unset OMP_NUM_THREADS CUDA_VISIBLE_DEVICES" in block


def test_t11bb_lane_blocks_unsuffixed_ids_for_both_variants() -> None:
    """Review 2026-11-04 #4: the BB reruns must be lane-governed like BA."""
    block = lane_block("T11BB")
    assert "RS_EVAL_T11BB_SUFFIX" in block and ":?" in block
    assert "lim_s2_single" in block and "lim_s2_median" in block
    assert "lim_s2_20bin" not in block  # the A100 cell never dispatches from here
    assert "for seed in 17 42 73; do" in block
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

    # "T11BA" is a CPU lane by name, so the pre-wait gating must set the flag
    # even when the caller exports nothing (review round-9: the flag was
    # exported only inside the case block, after the check had run).
    assert (
        "$CPU_LANES" in text
        and " T11BA " in text[text.index("CPU_LANES=") : text.index("CPU_LANES=") + 40]
    )


def test_t11ba_lane_starts_without_gpu_with_stub_nvidia(tmp_path):
    """Round-9 review: run the lane for real against a busy stub GPU.

    A fake nvidia-smi reports 9000 MiB used, uv is a no-op stub (no dispatch
    happens), and both roots live in tmp; the log must show the CPU-only skip
    and never the GPU wait."""

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    for tool, body in (
        (
            "nvidia-smi",
            "#!/usr/bin/env bash\necho '9000, 0 %'\n",
        ),
        (
            "uv",
            "#!/usr/bin/env bash\nexit 0\n",
        ),
    ):
        shim = stub_dir / tool
        shim.write_text(body)
        shim.chmod(0o755)
    logdir = tmp_path / "logs"
    logdir.mkdir()
    env = dict(
        RS_EVAL_RERUN_LOGDIR=str(logdir),
        RS_EVAL_RUNS_ROOT=str(tmp_path / "runs"),
        RS_EVAL_T11BA_SUFFIX="v2",
        RS_EVAL_SKIP_ENV="1",  # the repo ./.env must not re-pin roots
        PATH=f"{stub_dir}:{os.environ['PATH']}",
    )
    subprocess.run(
        ["bash", Path(REPO / "scripts" / "rerun_baselines.sh").as_posix(), "T11BA"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    log_path = logdir / "rerun-T11BA-baselines.out"
    assert log_path.is_file(), "the lane must write into the caller-provided log root"
    log = log_path.read_text(encoding="utf-8")
    assert "CPU-only lane; skipping the GPU idle wait" in log
    assert "GPU busy" not in log
    assert "no GPU visible" not in log


def test_t10b2_keeps_its_versioned_budget() -> None:
    block = lane_block("T10B2")
    assert "-v8" in block
    assert "IM_WORKER_TOTAL_STEPS" in block or "RS_EVAL_T10B2_SEEDS" in block


def test_t10b2_run_ids_carry_the_host_suffix() -> None:
    """Two ledgers cannot see each other: v8 must be host-scoped (review 2026-11-05)."""
    block = lane_block("T10B2")
    assert "RS_EVAL_T10B2_SUFFIX" in block
    assert "-v8-${RS_EVAL_T10B2_SUFFIX}" in block


def test_gpu_idle_limit_is_configurable() -> None:
    """A WSL desktop holds ~3.5 GB, above the historical 2000 MiB hard-code."""
    text = (REPO / "scripts" / "rerun_baselines.sh").read_text(encoding="utf-8")
    assert "RS_EVAL_GPU_IDLE_MIB:-2000" in text
    assert '[ "$used" -le "$idle_limit" ]' in text


def test_t10b2_can_stop_starting_seeds_after_a_cutoff() -> None:
    """The second seed must not run into the next working day (round 12)."""
    block = lane_block("T10B2")
    assert "RS_EVAL_RERUN_START_BEFORE" in block
    assert "skipped: past RS_EVAL_RERUN_START_BEFORE" in block
    assert "cutoff=$((midnight + cut_h * 3600 + cut_m * 60))" in block
    assert '"$(date +%s)" -ge "$cutoff"' in block


def test_t11_bbf_lane_pins_the_full_finetune_recipe() -> None:
    block = lane_block("T11BBF4090")
    assert "lim_s2_20bin_finetune" in block
    assert "LIM_T11_BATCH=1" in block
    assert "LIM_T11_ACCUM=8" in block
    assert "LIM_T11_CHECKPOINT_FRAMES=20" in block
    assert "-train-${seed}-${suffix}" in block


def test_t11_ba_lane_allows_a_targeted_seed_rerun() -> None:
    block = lane_block("T11BA")
    assert "RS_EVAL_T11BA_SEEDS:-17 42 73" in block


def test_t11_bbf_lr_lane_declares_both_backbone_lrs() -> None:
    block = lane_block("T11BBFLR4090")
    assert "lim_s2_20bin_finetune_lr5e5" in block
    assert "lim_s2_20bin_finetune_lr1e5" in block
    assert "LIM_T11_CHECKPOINT_FRAMES=20" in block
    assert "RS_EVAL_MANIFEST:?" in block
    assert "for seed in 17 42 73; do" in block


def test_t11_bbf20_lane_declares_the_longer_budget() -> None:
    block = lane_block("T11BBF20")
    assert "RS_EVAL_T11BBF20_PROTOCOL:-lim_s2_20bin_finetune_20ep_lr5e5" in block
    assert "LIM_T11_CHECKPOINT_FRAMES=20" in block
    assert "RS_EVAL_MANIFEST:?" in block
    assert "for seed in 17 42 73; do" in block


def test_t3c_t4c_anysat_lanes_can_extend_to_three_seeds() -> None:
    for lane, env in (("T3C", "RS_EVAL_T3C_SEEDS"), ("T4C", "RS_EVAL_T4C_SEEDS")):
        block = lane_block(lane)
        assert f"{env}:-42" in block
        assert "full_finetune" in block
        assert "run_anysat" in block


def test_t11_lanes_check_disk_before_starting() -> None:
    for lane in ("T11BBF4090", "T11BBFLR4090", "T11BBF20"):
        assert "require_free_gb" in lane_block(lane)


def test_t11o_lane_runs_both_heads_on_cpu() -> None:
    """T11O must reuse the AEF head pair and stay off the GPU like T11BA."""
    block = lane_block("T11O")
    assert "TASK=T11O" in block
    assert "olmoearth_v13_embedding" in block
    assert "olmoearth_v13_probe_mlp" in block and "olmoearth_v13_probe_upernet" in block
    assert "for seed in ${RS_EVAL_T11O_SEEDS:-17 42 73}; do" in block
    assert "-v1" in block
    assert "export OMP_NUM_THREADS=8" in block
    assert "unset OMP_NUM_THREADS" in block


def test_t11o_is_registered_as_a_cpu_lane() -> None:
    text = (REPO / "scripts" / "rerun_baselines.sh").read_text(encoding="utf-8")
    assert 'CPU_LANES=" T11BA T11BA20 T11BB T11O T11O128 T11O12820 T11O20 "' in text


def test_t11o20_lane_is_single_seed_and_versioned() -> None:
    block = lane_block("T11O20")
    assert "TASK=T11O" in block
    assert "olmoearth_v13_probe_mlp_20ep" in block and "olmoearth_v13_probe_upernet_20ep" in block
    assert "RS_EVAL_T11O20_SUFFIX:-v1" in block
    assert "for seed in ${RS_EVAL_T11O20_SEEDS:-17}; do" in block


def test_t11o128_lane_runs_both_heads_on_cpu() -> None:
    block = lane_block("T11O128")
    assert "TASK=T11O128" in block
    assert "olmoearth_v13_probe_mlp_128" in block
    assert "olmoearth_v13_probe_upernet_128" in block
    assert "RS_EVAL_T11O128_SUFFIX:-v1" in block
    assert "for seed in ${RS_EVAL_T11O128_SEEDS:-17 42 73}; do" in block
    assert "export OMP_NUM_THREADS=8" in block


def test_t11ba20_lane_is_the_aef_budget_diagnostic() -> None:
    block = lane_block("T11BA20")
    assert "TASK=T11" in block
    assert "aef_probe_mlp_20ep" in block and "aef_probe_upernet_20ep" in block
    assert "RS_EVAL_T11BA20_SUFFIX:-v1" in block
    assert "for seed in ${RS_EVAL_T11BA20_SEEDS:-17}; do" in block
    assert "export OMP_NUM_THREADS=8" in block


def test_t11o12820_lane_is_the_128dim_budget_diagnostic() -> None:
    block = lane_block("T11O12820")
    assert "TASK=T11O128" in block
    assert "olmoearth_v13_probe_mlp_128_20ep" in block
    assert "olmoearth_v13_probe_upernet_128_20ep" in block
    assert "RS_EVAL_T11O12820_SUFFIX:-v1" in block
    assert "for seed in ${RS_EVAL_T11O12820_SEEDS:-17}; do" in block
