"""Dispatch validated run requests to isolated model workers.

The control layer never imports model code: it validates the
task x model x protocol combination, creates the run directory, writes a
``WorkerRequest`` JSON file, and spawns the worker command configured in the
model registry. Worker progress events (JSONL on stdout) are forwarded live.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .contracts import WorkerAction, WorkerEvent, WorkerRequest
from .paths import RuntimePaths
from .registry import ModelSpec, ProtocolSpec, Registry, TaskSpec

VALID_ACTIONS: tuple[WorkerAction, ...] = (
    "probe",
    "extract_features",
    "train",
    "evaluate",
    "export_previews",
)


class DispatchError(ValueError):
    """Raised for invalid dispatch combinations or missing prerequisites."""


@dataclass(frozen=True)
class DispatchPlan:
    task: TaskSpec
    model: ModelSpec
    protocol: ProtocolSpec
    seed: int
    action: WorkerAction
    run_id: str
    run_dir: Path
    request_path: Path
    manifest_path: Path
    resume_from: Path | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def latest_manifest(data_root: Path, task_id: str) -> Path:
    """Resolve the manifest for a task.

    ``RS_EVAL_MANIFEST`` pins an explicit path (the content hash is what the
    run record stores, so formal experiments should use it). Otherwise the most
    recently *written* manifest wins; sorting by file name would pick by content
    hash, which has nothing to do with recency.
    """
    pinned = os.environ.get("RS_EVAL_MANIFEST")
    if pinned:
        path = Path(pinned)
        if not path.is_file():
            raise DispatchError(f"RS_EVAL_MANIFEST does not exist: {path}")
        return path
    manifest_dir = data_root / "manifests" / task_id
    if not manifest_dir.is_dir():
        raise DispatchError(
            f"no manifest directory for {task_id}; run rs-eval build-manifest first"
        )
    manifests = sorted(manifest_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise DispatchError(f"no manifest found for {task_id}; run rs-eval build-manifest first")
    return manifests[-1]


def plan_dispatch(
    *,
    action: str,
    task_id: str,
    model_id: str,
    protocol_id: str,
    seed: int,
    paths: RuntimePaths,
    run_id: str | None = None,
    resume_from: Path | None = None,
) -> DispatchPlan:
    """Validate the combination and lay out the run directory and request file."""
    registry = Registry(paths.config_root)
    tasks = {spec.task_id: spec for spec in registry.tasks()}
    models = {spec.model_id: spec for spec in registry.models()}
    protocols = {spec.protocol_id: spec for spec in registry.protocols()}

    if task_id not in tasks:
        raise DispatchError(f"unknown task: {task_id}; known: {', '.join(sorted(tasks))}")
    if model_id not in models:
        raise DispatchError(f"unknown model: {model_id}; known: {', '.join(sorted(models))}")
    if protocol_id not in protocols:
        raise DispatchError(
            f"unknown protocol: {protocol_id}; known: {', '.join(sorted(protocols))}"
        )

    task, model, protocol = tasks[task_id], models[model_id], protocols[protocol_id]
    if resume_from is not None:
        supported = model_id == "anysat_base" and task_id in {"T1", "T3", "T4", "T5"}
        if action != "train" or not supported:
            raise DispatchError("resume is supported only for AnySat T1/T3/T4/T5 train")
        resume_from = resume_from.expanduser().resolve()
        if not resume_from.is_file():
            raise DispatchError(f"resume checkpoint does not exist: {resume_from}")
    if not model.enabled:
        raise DispatchError(f"model is disabled: {model_id}")
    if protocol.tasks is not None and task.task_id not in protocol.tasks:
        raise DispatchError(
            f"protocol {protocol_id} is implemented only for tasks {protocol.tasks}; "
            f"the {model.worker_id} worker ignores its semantics on {task.task_id}"
        )
    if protocol.workers is not None and model.worker_id not in protocol.workers:
        raise DispatchError(
            f"protocol {protocol_id} is implemented by workers {protocol.workers}; "
            f"{model_id} runs the {model.worker_id} worker, which would record the run "
            "under this protocol without its semantics"
        )
    if action not in VALID_ACTIONS:
        raise DispatchError(f"unknown action: {action}; known: {', '.join(VALID_ACTIONS)}")
    if action == "train" and seed not in protocol.seeds:
        raise DispatchError(f"seed {seed} not in protocol seeds {protocol.seeds}")

    manifest_path = latest_manifest(paths.data_root, task_id)
    run_id = run_id or f"{model_id}-{task_id}-{protocol_id}-{action}-{seed}"
    run_dir = paths.runs_root / run_id
    if run_dir.exists():
        raise DispatchError(f"run directory already exists: {run_dir}")

    return DispatchPlan(
        task=task,
        model=model,
        protocol=protocol,
        seed=seed,
        action=cast("WorkerAction", action),
        run_id=run_id,
        run_dir=run_dir,
        request_path=run_dir / "request.json",
        manifest_path=manifest_path,
        resume_from=resume_from,
    )


def write_request(plan: DispatchPlan, paths: RuntimePaths) -> WorkerRequest:
    """Materialize the run directory with the WorkerRequest contract."""
    registry = Registry(paths.config_root)
    plan.run_dir.mkdir(parents=True)
    request = WorkerRequest(
        action=plan.action,
        run_id=plan.run_id,
        task_config_path=registry.task_paths()[plan.task.task_id].resolve().as_posix(),
        model_config_path=registry.model_paths()[plan.model.model_id].resolve().as_posix(),
        protocol_config_path=(registry.protocol_paths()[plan.protocol.protocol_id])
        .resolve()
        .as_posix(),
        manifest_path=plan.manifest_path.resolve().as_posix(),
        data_root=paths.data_root.as_posix(),
        weights_root=paths.weights_root.as_posix(),
        cache_root=paths.cache_root.as_posix(),
        runs_root=paths.runs_root.as_posix(),
        seed=plan.seed,
        resume_from=plan.resume_from.as_posix() if plan.resume_from else None,
    )
    plan.request_path.write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return request


def run_worker(
    plan: DispatchPlan,
    *,
    worker_command: list[str] | None = None,
    cwd: Path,
    stream: object = None,
) -> int:
    """Spawn the worker, forward WorkerEvent lines, and return its exit code."""
    command = list(worker_command or plan.model.worker_command or [])
    if not command:
        raise DispatchError(
            f"model {plan.model.model_id} has no worker_command; configure it in its registry yaml"
        )
    command.append(plan.request_path.as_posix())

    print(f"[dispatch] {plan.run_id}: {' '.join(command)}", file=sys.stderr)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        try:
            event = WorkerEvent.model_validate(json.loads(line))
        except ValueError:
            print(f"[dispatch] {line}", file=sys.stderr)
            continue
        print(
            f"[{event.level.upper()}] {event.event}: {event.message}",
            file=stream or sys.stdout,
            flush=True,
        )
    return process.wait()


def dispatch(
    *,
    action: str,
    task_id: str,
    model_id: str,
    protocol_id: str,
    seed: int,
    paths: RuntimePaths | None = None,
    worker_command: list[str] | None = None,
    run_id: str | None = None,
    resume_from: Path | None = None,
) -> int:
    """Plan, write, and execute one worker run; returns the worker exit code."""
    paths = paths or RuntimePaths.from_environment()
    plan = plan_dispatch(
        action=action,
        task_id=task_id,
        model_id=model_id,
        protocol_id=protocol_id,
        seed=seed,
        paths=paths,
        run_id=run_id,
        resume_from=resume_from,
    )
    write_request(plan, paths)
    return run_worker(plan, worker_command=worker_command, cwd=paths.config_root.parent)
