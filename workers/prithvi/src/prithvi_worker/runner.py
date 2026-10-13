"""Prithvi-EO-2.0 worker entry point.

Executes one ``WorkerRequest``: ``probe`` (environment/data/checkpoint checks)
and ``train`` (T1 repo-config baseline via TerraTorch). Runs inside the
isolated worker environment created by ``workers/prithvi/pyproject.toml``;
the shared contracts and audits come from the root ``rs_backbone_eval``
package.

Events are emitted as one JSON ``WorkerEvent`` per stdout line so the
control plane can forward progress without depending on any trainer's log
format.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.registry import ModelSpec, ProtocolSpec, TaskSpec, load_yaml_spec
from rs_backbone_eval.worker_runtime import RunWriter, emit_event

if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prithvi_worker.t1_baseline import train_baseline
    from prithvi_worker.t1_frozen import train_frozen
else:
    from .t1_baseline import train_baseline
    from .t1_frozen import train_frozen


def emit(
    run_id: str, event: str, message: str, level: str = "info", progress: float | None = None
) -> None:
    emit_event(run_id, event, message, level=level, progress=progress)


def _fail(writer: RunWriter, run_id: str, event: str, message: str) -> int:
    emit(run_id, event, message, level="error")
    writer.fail(message)
    return 1


def probe(request: WorkerRequest, writer: RunWriter, event_log) -> int:  # event_log: emit fn
    run_id = request.run_id
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    event_log(run_id, "probe.start", "validating manifest, checkpoint, and environment")

    manifest_path = Path(request.manifest_path)
    audit = audit_task_manifest(manifest_path, task_spec)
    strict_errors = audit.error_count - len(audit.leakage)
    if strict_errors:
        return _fail(
            writer,
            run_id,
            "probe.manifest",
            f"manifest audit failed with {strict_errors} errors: {audit.all_errors()[:3]}",
        )
    event_log(
        run_id,
        "probe.manifest",
        f"manifest OK: {audit.sample_count} samples, splits "
        f"{dict(audit.split_counts)}, {len(audit.leakage)} cross-split groups (official)",
        progress=0.3,
    )
    writer.update(
        tags={
            "manifest_samples": str(audit.sample_count),
            "manifest_leakage_groups": str(len(audit.leakage)),
        }
    )

    checkpoint = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint:
        return _fail(
            writer,
            run_id,
            "probe.checkpoint",
            f"environment variable {model_spec.checkpoint_env} is not set",
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        return _fail(writer, run_id, "probe.checkpoint", f"checkpoint not found: {checkpoint_path}")
    from rs_backbone_eval.dispatch import file_sha256

    digest = file_sha256(checkpoint_path)
    writer.update(model_checkpoint_sha256=digest)
    event_log(
        run_id,
        "probe.checkpoint",
        f"checkpoint OK: {checkpoint_path.name} ({checkpoint_path.stat().st_size / 1e6:.0f} MB)",
        progress=0.6,
    )

    try:
        import torch
    except ImportError as exc:
        return _fail(writer, run_id, "probe.torch", f"torch unavailable in worker env: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer.update(tags={"torch_version": torch.__version__, "device": device})
    event_log(
        run_id,
        "probe.torch",
        f"torch {torch.__version__}, device {device}",
        progress=0.8,
    )

    try:
        import terratorch  # noqa: F401
    except ImportError as exc:
        return _fail(writer, run_id, "probe.terrat", f"terratorch unavailable in worker env: {exc}")
    event_log(
        run_id,
        "probe.terrat",
        f"terratorch {getattr(terratorch, '__version__', 'unknown')} available",
        progress=1.0,
    )
    writer.complete("probe passed: manifest, checkpoint, torch, terratorch")
    return 0


def train_action(request: WorkerRequest, writer: RunWriter, event_log) -> int:
    run_id = request.run_id
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    if task_spec.task_id != "T1":
        message = f"prithvi worker only implements T1 training, got {task_spec.task_id}"
        return _fail(writer, run_id, "train.task", message)

    checkpoint = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint or not Path(checkpoint).is_file():
        return _fail(
            writer,
            run_id,
            "train.checkpoint",
            f"checkpoint missing via {model_spec.checkpoint_env}: {checkpoint}",
        )

    try:
        protocol = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
        if protocol.protocol_id == "frozen_probe":
            return train_frozen(request, writer, event_log, checkpoint=Path(checkpoint))
        return train_baseline(request, writer, event_log, checkpoint=Path(checkpoint))
    except Exception as exc:  # noqa: BLE001 - worker must record every failure
        return _fail(writer, run_id, "train.crash", f"{type(exc).__name__}: {exc}")


ACTIONS = {"probe": probe, "train": train_action}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: runner.py <request.json>", file=sys.stderr)
        return 2
    request = WorkerRequest.model_validate(
        json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    )

    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    protocol_spec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    model_spec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    manifest_digest = Path(request.manifest_path).name.removesuffix(".jsonl")

    writer = RunWriter(
        Path(request.runs_root) / request.run_id,
        run_id=request.run_id,
        task_spec=task_spec,
        protocol_spec=protocol_spec,
        model_id=model_spec.model_id,
        seed=request.seed,
        manifest_sha256=manifest_digest,
        worker="prithvi_worker",
    )
    writer.persist()

    emit(request.run_id, "worker.start", f"action {request.action}, seed {request.seed}")
    handler = ACTIONS.get(request.action)
    if handler is None:
        message = f"action not implemented yet in prithvi worker: {request.action}"
        return _fail(writer, request.run_id, "worker.action", message)
    code = handler(request, writer, emit)
    emit(request.run_id, "worker.exit", f"exit code {code}", level="debug")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
