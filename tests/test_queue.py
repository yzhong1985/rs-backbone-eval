"""Queue behaviour without a GPU, a worker, or real dispatch.

The queue exists so a new batch is a file drop instead of a shell edit; the
tests pin the parts that would silently mis-queue work: validation at enqueue,
FIFO order, the state transitions, and that a dry run never moves a file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs_backbone_eval.paths import RuntimePaths
from rs_backbone_eval.queue import (
    QueueError,
    QueueJob,
    cancel,
    enqueue,
    iter_jobs,
    run_queue,
)

REPO = Path(__file__).resolve().parents[1]

JOB = QueueJob(
    run_id="clay_v1_5-T11D-clay_rgbn_probe_s64_e80-train-17",
    task="T11D",
    model="clay_v1_5",
    protocol="clay_rgbn_probe_s64_e80",
    seed=17,
)


@pytest.fixture()
def paths(tmp_path: Path) -> RuntimePaths:
    data_root = tmp_path / "data"
    manifests = data_root / "manifests" / "T11D"
    manifests.mkdir(parents=True)
    (manifests / "manifest.jsonl").write_text("", encoding="utf-8")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    return RuntimePaths(
        data_root=data_root,
        weights_root=tmp_path / "weights",
        runs_root=runs_root,
        cache_root=tmp_path / "cache",
        config_root=REPO / "configs",
    )


def _run(paths: RuntimePaths, queue_dir: Path, **kwargs: object) -> int:
    defaults: dict[str, object] = {
        "once": True,
        "lock_path": queue_dir.parent / "gpu.lock",
        "gpu_used": lambda: 0,
        "sleep": lambda _seconds: None,
        "log": lambda _message: None,
    }
    defaults.update(kwargs)
    return run_queue(queue_dir, paths, **defaults)  # type: ignore[arg-type]


def test_enqueue_rejects_an_unknown_protocol(paths: RuntimePaths, tmp_path: Path) -> None:
    bad = QueueJob(run_id="x", task="T11D", model="clay_v1_5", protocol="no_such_protocol", seed=17)
    with pytest.raises(Exception, match="unknown protocol"):
        enqueue(paths, bad, queue_dir=tmp_path / "queue")


def test_enqueue_rejects_a_duplicate_run_id(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    enqueue(paths, JOB, queue_dir=queue_dir)
    with pytest.raises(QueueError, match="already queued in pending"):
        enqueue(paths, JOB, queue_dir=queue_dir)


def test_enqueue_rejects_foreign_env_keys(paths: RuntimePaths, tmp_path: Path) -> None:
    job = QueueJob(**{**JOB.__dict__, "env": {"PATH": "/tmp"}})
    with pytest.raises(QueueError, match="env keys must start with"):
        enqueue(paths, job, queue_dir=tmp_path / "queue")


def test_enqueue_rejects_an_existing_run_directory(paths: RuntimePaths, tmp_path: Path) -> None:
    (paths.runs_root / JOB.run_id).mkdir()
    with pytest.raises(Exception, match="already exists"):
        enqueue(paths, JOB, queue_dir=tmp_path / "queue")


def test_pending_jobs_are_listed_in_file_name_order(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    second = QueueJob(**{**JOB.__dict__, "run_id": "b-job"})
    first = QueueJob(**{**JOB.__dict__, "run_id": "a-job"})
    enqueue(paths, second, queue_dir=queue_dir)
    enqueue(paths, first, queue_dir=queue_dir)
    run_dir = queue_dir / "pending"
    names = [path.name for path in sorted(run_dir.glob("*.json"))]
    assert names == sorted(names)
    assert [job.run_id for _state, _path, job in iter_jobs(queue_dir)] == ["a-job", "b-job"]


def test_a_successful_job_moves_to_done_with_a_result(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    enqueue(paths, JOB, queue_dir=queue_dir)
    calls: list[dict[str, object]] = []

    def fake_dispatch(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    assert _run(paths, queue_dir, dispatch_fn=fake_dispatch) == 0
    assert calls and calls[0]["run_id"] == JOB.run_id
    assert not list((queue_dir / "pending").glob("*.json"))
    done = sorted((queue_dir / "done").glob("*.json"))
    jobs = [path for path in done if not path.name.endswith(".result.json")]
    sidecars = [path for path in done if path.name.endswith(".result.json")]
    assert len(jobs) == 1 and len(sidecars) == 1
    result = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert result["run_id"] == JOB.run_id and result["state"] == "done" and result["exit_code"] == 0


def test_a_failing_job_moves_to_failed(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    enqueue(paths, JOB, queue_dir=queue_dir)
    assert _run(paths, queue_dir, dispatch_fn=lambda **_kwargs: 3) == 1
    assert list((queue_dir / "failed").glob("*.json"))
    assert not list((queue_dir / "done").glob("*.json"))


def test_dispatch_rejection_is_recorded_not_retried(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    enqueue(paths, JOB, queue_dir=queue_dir)
    (paths.runs_root / JOB.run_id).mkdir()  # appears after enqueue

    def should_not_run(**_kwargs: object) -> int:
        raise AssertionError("dispatch must not be called for an existing run directory")

    assert _run(paths, queue_dir, dispatch_fn=should_not_run) == 1
    assert list((queue_dir / "failed").glob("*.json"))


def test_dry_run_previews_without_moving(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    enqueue(paths, JOB, queue_dir=queue_dir)
    messages: list[str] = []

    def should_not_run(**_kwargs: object) -> int:
        raise AssertionError("dry run must not dispatch")

    assert (
        _run(paths, queue_dir, dry_run=True, dispatch_fn=should_not_run, log=messages.append) == 0
    )
    assert messages and "would run" in messages[0]
    assert list((queue_dir / "pending").glob("*.json"))


def test_job_env_overrides_are_applied_and_restored(
    paths: RuntimePaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RS_EVAL_LABEL_FRACTION", raising=False)
    job = QueueJob(**{**JOB.__dict__, "env": {"RS_EVAL_LABEL_FRACTION": "0.1"}})
    queue_dir = tmp_path / "queue"
    enqueue(paths, job, queue_dir=queue_dir)
    seen: list[str | None] = []

    def fake_dispatch(**_kwargs: object) -> int:
        import os

        seen.append(os.environ.get("RS_EVAL_LABEL_FRACTION"))
        return 0

    _run(paths, queue_dir, dispatch_fn=fake_dispatch)
    import os

    assert seen == ["0.1"]
    assert os.environ.get("RS_EVAL_LABEL_FRACTION") is None


def test_cancel_moves_pending_to_failed(paths: RuntimePaths, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    enqueue(paths, JOB, queue_dir=queue_dir)
    target = cancel(queue_dir, JOB.run_id, reason="wrong manifest")
    assert target.parent.name == "failed"
    sidecar = json.loads(
        (target.parent / f"{target.name[: -len('.json')]}.result.json").read_text(encoding="utf-8")
    )
    assert sidecar["state"] == "cancelled" and sidecar["reason"] == "wrong manifest"
    # the job now lives in failed/, so a second cancel must refuse instead of
    # moving it again
    with pytest.raises(QueueError, match="is failed, not pending"):
        cancel(queue_dir, JOB.run_id)
    with pytest.raises(QueueError, match="no queued job"):
        cancel(queue_dir, "never-existed")
