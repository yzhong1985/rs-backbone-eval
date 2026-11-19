"""Filesystem job queue in front of ``rs-eval dispatch``.

The lanes in ``scripts/rerun_baselines.sh`` hard-code their run lists, so a new
batch means editing shell. This module gives the same guarantees from a
directory of job files: one run at a time (it takes the same
``/tmp/rs-eval-gpu.lock`` the lanes use), skip/reject an existing run, and let
``run.json`` stay the only authority on results.

Layout under ``RS_EVAL_QUEUE_ROOT`` (default ``<runs_root>/../queue``)::

    pending/20261116T103000Z_<run_id>.json   one job; name order is FIFO order
    running/20261116T103000Z_<run_id>.json
    done/...json + ...json.result.json       exit code, timing, state
    failed/...json + ...json.result.json

A job is one dispatch argument set plus optional environment overrides::

    {"run_id": "...", "action": "train", "task": "T11D", "model": "clay_v1_5",
     "protocol": "clay_rgbn_probe_s64_e80", "seed": 17,
     "env": {"RS_EVAL_MANIFEST": "/.../10pct-groups.jsonl"}, "note": "..."}
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dispatch import DispatchError, _gpu_used_mib, dispatch, plan_dispatch
from .paths import RuntimePaths

STATES = ("pending", "running", "done", "failed")
RESULT_SUFFIX = ".result.json"

# Job files may only touch variables the workers already understand. The
# prefixes match the worker env namespaces; anything else is a typo that would
# silently do nothing.
ENV_PREFIXES = (
    "RS_EVAL_",
    "CLAY_WORKER_",
    "ANYSAT_WORKER_",
    "PRITHVI_WORKER_",
    "LIM_WORKER_",
    "AEF_WORKER_",
    "OLMOEARTH_WORKER_",
)


class QueueError(ValueError):
    """Raised for malformed job files, duplicate ids, or invalid combinations."""


@dataclass(frozen=True)
class QueueJob:
    run_id: str
    task: str
    model: str
    protocol: str
    seed: int
    action: str = "train"
    env: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "action": self.action,
            "task": self.task,
            "model": self.model,
            "protocol": self.protocol,
            "seed": self.seed,
        }
        if self.env:
            payload["env"] = dict(self.env)
        if self.note:
            payload["note"] = self.note
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QueueJob:
        missing = [
            key for key in ("run_id", "task", "model", "protocol", "seed") if key not in data
        ]
        if missing:
            raise QueueError(f"job is missing {', '.join(missing)}")
        env = data.get("env") or {}
        if not isinstance(env, Mapping) or not all(
            isinstance(value, str) for value in env.values()
        ):
            raise QueueError("job env must be a mapping of strings")
        validate_env(env)
        seed = data["seed"]
        if not isinstance(seed, int):
            raise QueueError(f"seed must be an integer, got {seed!r}")
        return cls(
            run_id=str(data["run_id"]),
            task=str(data["task"]),
            model=str(data["model"]),
            protocol=str(data["protocol"]),
            seed=seed,
            action=str(data.get("action", "train")),
            env=dict(env),
            note=str(data.get("note", "")),
        )


def validate_env(env: Mapping[str, str]) -> None:
    bad = sorted(key for key in env if not key.startswith(ENV_PREFIXES))
    if bad:
        raise QueueError(
            f"env keys must start with one of {', '.join(ENV_PREFIXES)}; rejected: {bad}"
        )


def default_queue_root(paths: RuntimePaths) -> Path:
    configured = os.environ.get("RS_EVAL_QUEUE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (paths.runs_root.parent / "queue").resolve()


def _ensure_dirs(queue_dir: Path) -> None:
    for state in STATES:
        (queue_dir / state).mkdir(parents=True, exist_ok=True)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


ACTIVE_STATES = ("pending", "running", "done")


def iter_jobs(
    queue_dir: Path, states: tuple[str, ...] = ACTIVE_STATES
) -> Iterator[tuple[str, Path, QueueJob]]:
    """Yield ``(state, path, job)`` for the readable jobs in ``states``, pending first.

    A file that cannot be parsed is logged and skipped. The consumer parks such a
    file in ``failed/``, and listing or queueing must not fail on it afterwards.
    ``failed/`` is read only when a caller names it in ``states``.
    """
    _ensure_dirs(queue_dir)
    for state in states:
        for path in sorted((queue_dir / state).glob("*.json")):
            if path.name.endswith(RESULT_SUFFIX):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = QueueJob.from_dict(data)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                _LOG.warning("skipping unreadable queue file %s/%s: %s", state, path.name, exc)
                continue
            yield state, path, job


def enqueue(paths: RuntimePaths, job: QueueJob, *, queue_dir: Path | None = None) -> Path:
    """Validate a job against the registry and park it in ``pending/``."""
    queue_dir = queue_dir or default_queue_root(paths)
    validate_env(job.env)
    plan_dispatch(
        action=job.action,
        task_id=job.task,
        model_id=job.model,
        protocol_id=job.protocol,
        seed=job.seed,
        paths=paths,
        run_id=job.run_id,
    )
    for state, path, existing in iter_jobs(queue_dir, states=("pending", "running")):
        if existing.run_id == job.run_id:
            raise QueueError(f"run_id already queued in {state}: {path.name}")
    target = queue_dir / "pending" / f"{_stamp()}_{job.run_id}.json"
    if target.exists():  # same second, same id: the duplicate check above covers run_id
        raise QueueError(f"job file already exists: {target}")
    _write_atomic(target, job.to_dict())
    return target


def cancel(queue_dir: Path, run_id: str, *, reason: str = "cancelled") -> Path:
    """Move a pending job to ``failed/`` with a result sidecar (never deletes it)."""
    for state, path, job in iter_jobs(queue_dir, states=STATES):
        if job.run_id != run_id:
            continue
        if state != "pending":
            raise QueueError(f"{run_id} is {state}, not pending")
        target = _move(path, queue_dir / "failed")
        _write_result(target, {"run_id": run_id, "state": "cancelled", "reason": reason})
        return target
    raise QueueError(f"no queued job with run_id {run_id}")


def _move(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    os.replace(source, target)
    return target


def _result_path(job_path: Path) -> Path:
    return job_path.with_name(job_path.name[: -len(".json")] + RESULT_SUFFIX)


def _write_result(job_path: Path, payload: Mapping[str, Any]) -> None:
    _write_atomic(_result_path(job_path), payload)


_LOG = logging.getLogger(__name__)
DEFAULT_GPU_LOCK = "/tmp/rs-eval-gpu.lock"


@contextmanager
def _gpu_lock(lock_path: Path) -> Iterator[None]:
    """Same flock(2) the shell lanes take, so the two mechanisms exclude each other."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # The handle must stay open while the lock is held, so it cannot be a
    # `with open(...)` that closes before the job finishes.
    with open(lock_path, "a+", encoding="utf-8") as handle:  # noqa: SIM117
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _applied_env(overrides: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _wait_for_idle(
    idle_limit: int,
    *,
    poll_seconds: float,
    sleep: Callable[[float], None],
    gpu_used: Callable[[], int | None],
    log: Callable[[str], None],
) -> None:
    while True:
        used = gpu_used()
        if used is not None and used <= idle_limit:
            return
        log(f"GPU busy ({used} MiB, idle limit {idle_limit}); waiting")
        sleep(poll_seconds)


def run_queue(
    queue_dir: Path,
    paths: RuntimePaths,
    *,
    once: bool = False,
    dry_run: bool = False,
    poll_seconds: float = 60.0,
    idle_limit: int | None = None,
    lock_path: Path | None = None,
    dispatch_fn: Callable[..., int] = dispatch,
    gpu_used: Callable[[], int | None] = _gpu_used_mib,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """Drain ``pending/``; returns the number of failed jobs.

    ``once`` handles at most one job, ``dry_run`` only prints the next one and
    never moves files. A job whose run directory already exists is failed
    instead of looping forever on a dispatch rejection. ``dispatch_fn`` and the
    GPU probes are injectable so the queue can be tested without a card.
    """
    _ensure_dirs(queue_dir)
    if lock_path is None:
        # Same resolution as scripts/rerun_baselines.sh: a lane started with a
        # custom RS_EVAL_GPU_LOCK must exclude the queue on that same file.
        lock_path = Path(os.environ.get("RS_EVAL_GPU_LOCK") or DEFAULT_GPU_LOCK)
    limit = idle_limit
    if limit is None:
        limit = int(os.environ.get("RS_EVAL_GPU_IDLE_MIB", "2000") or 2000)
    failures = 0
    while True:
        pending = sorted((queue_dir / "pending").glob("*.json"))
        if not pending:
            if once or dry_run:
                return failures
            sleep(poll_seconds)
            continue
        job_path = pending[0]
        try:
            job = QueueJob.from_dict(json.loads(job_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # A malformed job file must not stop the consumer or block the
            # jobs behind it: park it in failed/ with the reason.
            log(f"{job_path.name}: unreadable job: {exc}")
            if dry_run:
                return failures + 1
            finished = _move(job_path, queue_dir / "failed")
            _write_result(finished, {"state": "failed", "exit_code": 1, "error": str(exc)})
            failures += 1
            if once:
                return failures
            continue
        if dry_run:
            log(f"[dry-run] would run {job.run_id} ({job.task}/{job.model}/{job.protocol})")
            return failures
        started = time.time()
        active = _move(job_path, queue_dir / "running")
        try:
            if (paths.runs_root / job.run_id).exists():
                raise DispatchError(f"run directory already exists: {paths.runs_root / job.run_id}")
            with _gpu_lock(lock_path):
                _wait_for_idle(
                    limit, poll_seconds=poll_seconds, sleep=sleep, gpu_used=gpu_used, log=log
                )
                overrides = dict(job.env)
                overrides.setdefault("RS_EVAL_REQUIRE_IDLE_GPU", "1")
                with _applied_env(overrides):
                    code = dispatch_fn(
                        action=job.action,
                        task_id=job.task,
                        model_id=job.model,
                        protocol_id=job.protocol,
                        seed=job.seed,
                        run_id=job.run_id,
                    )
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the queue
            # Registry/config errors (unknown protocol, invalid YAML, a task the
            # protocol does not allow) used to propagate and kill the consumer.
            code = 1
            log(f"{job.run_id}: {type(exc).__name__}: {exc}")
        finished = _move(active, queue_dir / ("done" if code == 0 else "failed"))
        _write_result(
            finished,
            {
                "run_id": job.run_id,
                "state": "done" if code == 0 else "failed",
                "exit_code": code,
                "seconds": round(time.time() - started, 1),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        if code != 0:
            failures += 1
        if once:
            return failures
