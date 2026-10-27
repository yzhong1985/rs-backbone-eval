"""Static guards for front-end behaviour that has no DOM test harness.

Each assertion pins a specific review finding (2026-10-27, third round):
a dropped guard would otherwise only show up as a browser bug again.
"""

from __future__ import annotations

from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "web" / "assets" / "app.js"


def source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_every_page_discards_late_responses() -> None:
    """Task, compare and run views all need the token guard (T4 showing T3)."""
    assert source().count("token !== viewToken") >= 3


def test_home_endpoints_are_loaded_independently() -> None:
    """A 500 from /api/downloads must not stop the task page from rendering."""
    assert "Promise.allSettled" in source()
    assert "route();\n  load();" in source()


def test_cells_and_scatter_show_the_experiment_condition() -> None:
    """Random-init and few-shot cells must be distinguishable in the tables."""
    text = source()
    assert text.count("conditionText(cell)") >= 2
    assert "非默认" in text
    assert 'isDefaultCondition(cell) ? "#6aa9ff"' in text
