"""Version-ancestry audit for run records.

A fix that changed stored metric values makes every run that predates it
incomparable with a run that includes it. Commit *time* is not enough (two
commits can share a timestamp), so the check asks git whether the fix is an
ancestor of the run's recorded commit. Records with no recorded commit are
"unknown" and never folded into a passing group.

Used by ``scripts/audit_runs.py`` (CLI) and by the report, which must not
present a cell built from pre-fix runs as a finished result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Fixes that changed stored metric values, in landing order.
P0 = "ffdc6db"  # seed_everything + metric selection/aggregation
P0_TRAIN = "7377fb6"  # adapter gradient, Clay B1 unfreeze
B0_RESTORE = "ea9e8a4"  # B0 best-checkpoint restore + day-of-year
T5_GRID = "3b4460a"  # T5 head trained on the full label grid
R6_GRID = "35a972e"  # frozen-path loss on the full label grid
T4_METRIC = "ca27c4c"  # T4 primary metric: fg IoU was renamed to mean_iou
T2_ACCUM = "12844c3"  # AnySat T2: declared gradient accumulation was ignored
T1S_INPUT = "f69e703"  # T1S: Clay S1 metadata, NaN backscatter, AnySat dates
T10_GRAD = "d6f3da8"  # T10: gradient cap on the epoch-tail step (training numerics)
T10_ROI_ORDER = "4b8b197"  # T10: RoI features were permuted by FPN level (mAP 0)
T10_DETECTOR = "0a9993a"  # T10: anchor row order, NMS score indexing, RoI levels
T9_PADDING = "0e88549"  # T9: padding counted as background instead of ignore
T10_RPN_NMS = "e6de76c"  # T10: RPN batch cap and rotated final NMS
NORM_DECISION = "a808302"  # T9/T10: declared (LIM: ImageNet) normalization, seed-independent
T11_TIME = "a948b6c"  # T11: day-offset time positions, AnySat UPerNet decoder, step counts
T11_GPU = "e115ca0"  # T11: numpy standardisation (CPU/CUDA mix), OlmoEarth decoder/input_res

UNKNOWN = "unknown"
OK = "ok"
STALE = "stale"

# Models whose worker had the T4 foreground-IoU/mean_iou rename bug. The bug
# was in the OlmoEarth task wrapper; the other workers never overwrote the
# two-class mIoU (review 2026-10-23, seventh round).
OLMOEARTH_MODELS = {"olmoearth_v1_2_tiny", "olmoearth_v1_2_base"}


def required_fixes(task: str, model: str, regime: str) -> list[str]:
    """Fixes whose absence makes a run's stored metrics incomparable."""
    fixes = [P0, P0_TRAIN]
    if regime == "reference":
        # B0 is the Prithvi reference reproduction; it needs the checkpoint
        # restore fix on top of the training fixes.
        fixes.append(B0_RESTORE)
    if task == "T5":
        # day-of-year encoding (ea9e8a4) plus the full-label grid for the
        # per-date encoders (Prithvi/Clay).
        fixes.append(B0_RESTORE)
        if model in {"prithvi_eo_v2_300_tl", "clay_v1_5"}:
            fixes.append(T5_GRID)
    if task == "T10":
        # The epoch-tail flush used to step without the gradient cap, so a T10 run
        # from before this fix trained slightly differently. Detection only; the
        # T9/T8/baseline records are untouched by it.
        fixes.append(T10_GRAD)
        # RoI features came back grouped by FPN level, so every box was trained
        # against another box's label and every pre-fix run reported mAP ~0.
        fixes.append(T10_ROI_ORDER)
        # Anchors were generated anchor-major while the RPN flattens cell-major,
        # the class-wise NMS mixed local and global score indices, and small boxes
        # were pooled from the wrong FPN level (review 2026-11-02).
        fixes.append(T10_DETECTOR)
        # The RPN oversampled positives past the declared batch (656 of 256 in
        # the review's example) and the final NMS suppressed rotated boxes by
        # their hulls (review 2026-11-02).
        fixes.append(T10_RPN_NMS)
    if task == "T9":
        # Boundless label reads turned tile padding into background class 0, so
        # pre-fix T9 training saw ~32% of tiles with a false background band.
        fixes.append(T9_PADDING)
    if task in {"T9", "T10"}:
        # Before a808302 the T9/T10 workers normalised with statistics sampled
        # per run seed (so the seeds of one cell saw different inputs) and LIM
        # did not use its ImageNet pretraining statistics. preprocessing_id
        # separates the variants, but build_matrix ranks variants rather than
        # following the declared normalization, so the old runs must be
        # excluded here (review 1b3ceae, 2026-11-03).
        fixes.append(NORM_DECISION)
    if task == "T11":
        # a948b6c replaced day-of-year with days-since-2018-09-01 in the 20-bin
        # encoders (LIM + AnySat) and moved AnySat from LiteSpatialHead to
        # UPerNetHead, so pre-fix T11 metrics are not comparable. e115ca0 fixed
        # the CPU/CUDA statistics mix and the OlmoEarth decoder/input_res.
        if model in {"lim_mit_huge", "anysat_base"}:
            fixes.append(T11_TIME)
        if model in {"anysat_base", "clay_v1_5", "olmoearth_v1_2_tiny"}:
            fixes.append(T11_GPU)
    if task == "T3":
        fixes.append(B0_RESTORE)
    if task == "T4" and model in OLMOEARTH_MODELS:
        # Only the OlmoEarth worker stored the foreground IoU under the name
        # mean_iou before ca27c4c; Clay/Prithvi/AnySat always recorded the
        # two-class mIoU, so their pre-fix records remain comparable.
        fixes.append(T4_METRIC)
    if task == "T2" and model == "anysat_base":
        # Before 12844c3 a 1x2 launch stepped on every micro-batch (effective
        # batch 1); runs that did not record their launch batch cannot be told
        # apart from those, so every pre-fix AnySat T2 run is incomparable.
        fixes.append(T2_ACCUM)
    if task == "T1S":
        # No T1S run completed before f69e703; any older record fed NaN and
        # (Clay) an out-of-distribution radar wavelength into the backbone.
        fixes.append(T1S_INPUT)
    if task in {"T1", "T2", "T3"} and model == "prithvi_eo_v2_300_tl" and regime == "frozen_probe":
        fixes.append(R6_GRID)
    if task == "T3" and model == "clay_v1_5" and regime == "frozen_probe":
        fixes.append(R6_GRID)
    # De-duplicate while keeping order.
    seen: list[str] = []
    for fix in fixes:
        if fix not in seen:
            seen.append(fix)
    return seen


class Ancestry:
    """Cached ``git merge-base --is-ancestor`` lookups."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._cache: dict[tuple[str, str], bool] = {}

    def has(self, fix: str, commit: str) -> bool:
        key = (fix, commit)
        if key not in self._cache:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", fix, commit],
                cwd=self.repo,
                capture_output=True,
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"git merge-base failed for {fix} -> {commit}: "
                    f"{result.stderr.decode(errors='replace').strip()}"
                )
            self._cache[key] = result.returncode == 0
        return self._cache[key]

    def verdict(self, commit: str | None, fixes: list[str]) -> str:
        if not commit:
            return UNKNOWN
        # ``_code_version`` appends ``-dirty`` when the tree had uncommitted
        # changes; ancestry is checked against the base commit.
        base = commit.split("-dirty", 1)[0]
        return OK if all(self.has(fix, base) for fix in fixes) else STALE


def stale_run_ids(records, repo: Path) -> set[str]:
    """Run ids whose recorded commit predates a fix that changed their metric."""
    ancestry = Ancestry(repo)
    stale: set[str] = set()
    for record in records:
        training = record.training
        fixes = required_fixes(record.task_id, record.model_id, training.regime)
        if ancestry.verdict(record.git_commit, fixes) == STALE:
            stale.add(record.run_id)
    return stale


def excluded_run_ids(records, repo: Path) -> set[str]:
    """Run ids that must not enter the formal results.

    Two classes are excluded: ``STALE`` (the commit predates a fix that changed
    the stored metric) and ``UNKNOWN`` (no recorded ``git_commit``, so
    comparability cannot be established). The report and the portal both use
    this set; before this, a completed record without a commit silently became
    a result (review 2026-10-22, second round).
    """
    ancestry = Ancestry(repo)
    excluded: set[str] = set()
    for record in records:
        training = record.training
        fixes = required_fixes(record.task_id, record.model_id, training.regime)
        if ancestry.verdict(record.git_commit, fixes) != OK:
            excluded.add(record.run_id)
    return excluded
