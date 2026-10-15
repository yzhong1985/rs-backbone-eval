"""OlmoEarth worker entry point (stage A: probe only).

Executes one ``WorkerRequest``. The worker runs inside the isolated environment
created by ``workers/olmoearth/pyproject.toml``; the shared contracts come from
the root ``rs_backbone_eval`` package. See ``docs/OlmoEarth接入进展.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.registry import ModelSpec, ProtocolSpec, TaskSpec, load_yaml_spec
from rs_backbone_eval.worker_runtime import RunWriter, emit_event

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from olmoearth_worker.t1_frozen import probe, train_action
else:
    from .t1_frozen import probe, train_action


def event_log(run_id: str, event: str, message: str, **kwargs) -> None:
    emit_event(run_id, event, message, **kwargs)


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

    writer = RunWriter(
        Path(request.runs_root) / request.run_id,
        run_id=request.run_id,
        task_spec=task_spec,
        protocol_spec=protocol_spec,
        model_id=model_spec.model_id,
        seed=request.seed,
        manifest_sha256=manifest_sha256(Path(request.manifest_path)),
        worker="olmoearth_worker",
    )
    writer.persist()

    event_log(request.run_id, "worker.start", f"action {request.action}, seed {request.seed}")
    handler = {"probe": probe, "train": train_action}.get(request.action)
    if handler is None:
        from rs_backbone_eval.worker_runtime import emit_event

        message = f"action not implemented in olmoearth worker: {request.action}"
        emit_event(request.run_id, "worker.action", message, level="error")
        writer.fail(message)
        return 1
    code = handler(request, writer, event_log)
    event_log(request.run_id, "worker.exit", f"exit code {code}", level="debug")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
