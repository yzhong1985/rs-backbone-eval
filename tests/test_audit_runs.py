"""Tests for scripts/audit_runs.py (protocol audit).

The audit must judge each run record individually and by Git ancestry, not by
commit time, and it must never fold an unknown code version into a passing
group.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_audit():
    spec = importlib.util.spec_from_file_location("audit_runs", ROOT / "scripts" / "audit_runs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = load_audit()


def test_required_fixes_cover_b0_restore_and_temporal_paths() -> None:
    # B0 needs the checkpoint-restore fix on top of the training fixes.
    assert audit.B0_RESTORE in audit.required_fixes("T1", "prithvi_eo_v2_300_tl", "reference")
    # AnySat T5 needs the day-of-year fix (shipped with the B0 restore commit).
    assert audit.B0_RESTORE in audit.required_fixes("T5", "anysat_base", "frozen_probe")
    # Prithvi/Clay T5 additionally need the full-label grid.
    assert audit.T5_GRID in audit.required_fixes("T5", "clay_v1_5", "frozen_probe")
    # Prithvi frozen T1 needs the R6 frozen-grid fix.
    assert audit.R6_GRID in audit.required_fixes("T1", "prithvi_eo_v2_300_tl", "frozen_probe")
    # A plain finetune path needs only the training fixes.
    assert audit.required_fixes("T1", "clay_v1_5", "full_finetune") == [audit.P0, audit.P0_TRAIN]
    assert audit.T2_ACCUM in audit.required_fixes("T2", "anysat_base", "frozen_adapter")
    assert audit.T2_ACCUM in audit.required_fixes("T2", "anysat_base", "full_finetune")
    assert audit.T2_ACCUM not in audit.required_fixes("T2", "clay_v1_5", "full_finetune")
    assert audit.T1S_INPUT in audit.required_fixes("T1S", "clay_v1_5", "frozen_probe")
    # The T10 gradient-cap fix must not mark the segmentation cells stale.
    assert audit.T10_GRAD in audit.required_fixes("T10", "lim_mit_huge", "frozen_probe")
    assert audit.T10_GRAD not in audit.required_fixes("T9", "lim_mit_huge", "frozen_probe")
    assert audit.T10_GRAD not in audit.required_fixes("T1", "clay_v1_5", "full_finetune")
    assert audit.T1S_INPUT not in audit.required_fixes("T1", "clay_v1_5", "frozen_probe")
    assert audit.T4_METRIC in audit.required_fixes("T4", "olmoearth_v1_2_tiny", "frozen_adapter")
    assert audit.T4_METRIC in audit.required_fixes("T4", "olmoearth_v1_2_base", "frozen_probe")
    # The rename bug lived only in the OlmoEarth worker; the other T4 paths
    # recorded the two-class mIoU correctly and must not be excluded.
    for model in ("clay_v1_5", "prithvi_eo_v2_300_tl", "anysat_base"):
        assert audit.T4_METRIC not in audit.required_fixes("T4", model, "frozen_probe")


def test_verdict_is_unknown_without_a_commit() -> None:
    ancestry = audit.Ancestry(ROOT)
    assert ancestry.verdict(None, [audit.P0]) == audit.UNKNOWN
    assert ancestry.verdict("", [audit.P0]) == audit.UNKNOWN


def test_verdict_strips_dirty_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    ancestry = audit.Ancestry(ROOT)
    seen: list[str] = []

    def fake_has(fix: str, commit: str) -> bool:
        seen.append(commit)
        return True

    monkeypatch.setattr(ancestry, "has", fake_has)
    assert ancestry.verdict("abc123-def-dirty", [audit.P0]) == audit.OK
    assert seen == ["abc123-def"]


def test_verdict_is_stale_when_any_fix_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    ancestry = audit.Ancestry(ROOT)

    def fake_has(fix: str, commit: str) -> bool:
        return fix != audit.R6_GRID  # the R6 fix is not an ancestor

    monkeypatch.setattr(ancestry, "has", fake_has)
    assert ancestry.verdict("deadbeef", [audit.P0, audit.R6_GRID]) == audit.STALE


def test_ancestry_uses_git_ancestry_not_time() -> None:
    # 1445a0b and the later fixes share a commit timestamp, so a time-based
    # check would wrongly pass; ancestry must say the fixes are not ancestors.
    ancestry = audit.Ancestry(ROOT)
    assert ancestry.has(audit.P0, audit.P0) is True
    assert ancestry.has(audit.P0, "1445a0b") is False


def test_unverifiable_commit_is_excluded_from_the_formal_results(monkeypatch) -> None:
    from types import SimpleNamespace

    def fake_verdict(self, commit, fixes):
        return audit.UNKNOWN if not commit else audit.OK

    monkeypatch.setattr(audit.Ancestry, "verdict", fake_verdict)
    training = SimpleNamespace(regime="frozen_probe")
    verified = SimpleNamespace(
        run_id="with-commit",
        task_id="T1",
        model_id="clay_v1_5",
        git_commit="deadbeef",
        training=training,
    )
    unknown = SimpleNamespace(
        run_id="no-commit",
        task_id="T1",
        model_id="clay_v1_5",
        git_commit=None,
        training=training,
    )
    assert audit.excluded_run_ids([verified, unknown], ROOT) == {"no-commit"}
    # stale_run_ids stays the pre-fix list; unknown is handled separately.
    assert audit.stale_run_ids([verified, unknown], ROOT) == set()
