"""Static wiring checks for epoch-boundary resume in the training workers.

The runtime test (`tests/test_training_checkpoint.py`) needs torch, so it is
skipped in the controller environment that CI runs in. These checks parse the
worker sources instead, which keeps the *placement* of the save covered by the
normal CI run: a save that sits behind `if budget_hit` never runs for a
50-epoch budget, and that failure is invisible to unit tests that drive the
checkpoint object directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

RESUMABLE_LOOPS = {
    "anysat": ["t1_adapter.py", "t3.py", "t4.py", "t5.py"],
    "lim": ["t9.py", "t10.py"],
    "prithvi": ["t9.py"],
}


def _source(worker: str, filename: str) -> Path:
    return ROOT / "workers" / worker / "src" / f"{worker}_worker" / filename


def _epoch_loops(tree: ast.Module) -> list[ast.For]:
    loops = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        call = node.iter
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and call.args
            and isinstance(call.args[0], ast.Attribute)
            and call.args[0].attr == "start_epoch"
        ):
            loops.append(node)
    return loops


def _direct_saves(loop: ast.For) -> list[ast.Call]:
    return [
        statement.value
        for statement in loop.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "save"
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == "checkpoints"
    ]


@pytest.mark.parametrize(
    ("worker", "filename"),
    [(worker, name) for worker, names in RESUMABLE_LOOPS.items() for name in names],
)
def test_resumable_loop_saves_on_every_epoch(worker: str, filename: str) -> None:
    path = _source(worker, filename)
    loops = _epoch_loops(ast.parse(path.read_text(encoding="utf-8")))
    assert loops, f"{path}: no `for epoch in range(checkpoints.start_epoch, ...)` loop"
    for loop in loops:
        assert _direct_saves(loop), (
            f"{path}: no checkpoints.save() directly in the epoch loop body. A save nested "
            "under `if budget_hit:` is dead code for a run that uses its full epoch budget."
        )


@pytest.mark.parametrize(
    ("worker", "filename"),
    [(worker, name) for worker, names in RESUMABLE_LOOPS.items() for name in names],
)
def test_resume_guard_matches_the_wired_tasks(worker: str, filename: str) -> None:
    """Every wired loop must sit behind a runner that accepts --resume-from."""
    runner = _source(worker, "runner.py").read_text(encoding="utf-8")
    assert "resume_from" in runner, f"{worker}: runner never reads resume_from"
