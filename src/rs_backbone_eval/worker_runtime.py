"""Shared worker runtime: atomic run.json writer and event emission.

Workers own the run record lifecycle: create it as ``running``, update fields
as the action progresses, and settle on ``completed`` or ``failed``. Every
write goes through a ``run.json.tmp`` file followed by an atomic replace so
the Portal never observes a partially written record.

The three model workers share this module through the root package so their
run records and progress events follow one contract.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .contracts import (
    ArtifactRecord,
    CurvePoint,
    DatasetRecord,
    InputRecord,
    MetricRecord,
    RunRecord,
    TrainingRecord,
)
from .registry import ProtocolSpec, TaskSpec


def emit_event(
    run_id: str,
    event: str,
    message: str,
    *,
    level: str = "info",
    progress: float | None = None,
) -> None:
    """Write one WorkerEvent JSON line to stdout for the control plane."""
    payload = {"run_id": run_id, "level": level, "event": event, "message": message}
    if progress is not None:
        payload["progress"] = progress
    print(json.dumps(payload, ensure_ascii=False), flush=True)


class RunWriter:
    """Incremental, atomic updates for one run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        run_id: str,
        task_spec: TaskSpec,
        protocol_spec: ProtocolSpec,
        model_id: str,
        seed: int,
        manifest_sha256: str,
        worker: str = "worker",
    ) -> None:
        self.run_dir = run_dir
        self.worker = worker
        self.path = run_dir / "run.json"
        self._record = RunRecord(
            run_id=run_id,
            status="running",
            task_id=task_spec.task_id,
            model_id=model_id,
            protocol_id=protocol_spec.protocol_id,
            seed=seed,
            dataset=DatasetRecord(
                dataset_id=task_spec.dataset_id,
                version=None,
                manifest_sha256=manifest_sha256,
                split="all",
            ),
            input=InputRecord(
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                time_steps=task_spec.time_steps,
                gsd_m=None,
                preprocessing_id="none",
            ),
            training=TrainingRecord(
                regime=protocol_spec.regime,
                head_id="none",
            ),
        )

    @property
    def record(self) -> RunRecord:
        return self._record

    def update(self, **changes) -> None:
        """Apply top-level updates (e.g. status=, notes=, model_checkpoint_sha256=).

        ``tags`` entries are merged into the existing dict; all other fields
        are replaced.
        """
        for name, value in changes.items():
            if name == "tags" and isinstance(value, dict):
                self._record.tags.update(value)
            else:
                setattr(self._record, name, value)
        self._persist()

    def update_training(self, **fields) -> None:
        """Merge fields into the nested training record."""
        self._record.training = self._record.training.model_copy(update=fields)
        self._persist()

    def update_input(self, **fields) -> None:
        """Merge fields into the nested input record."""
        self._record.input = self._record.input.model_copy(update=fields)
        self._persist()

    def add_metrics(self, metrics: list[MetricRecord]) -> None:
        self._record.metrics.extend(metrics)
        self._persist()

    def add_curve(self, name: str, split: str, step: int, value: float) -> None:
        self._record.curves.append(CurvePoint(name=name, split=split, step=step, value=value))
        self._persist()

    def add_artifact(self, kind: str, path: str, title: str | None = None) -> None:
        self._record.artifacts.append(ArtifactRecord(kind=kind, path=path, title=title))
        self._persist()

    def complete(self, notes: str | None = None) -> None:
        self.update(status="completed", notes=notes)

    def fail(self, message: str) -> None:
        self.update(status="failed", notes=message)

    def persist(self) -> None:
        """Write the current record atomically (also used for the initial write)."""
        self._record.updated_at = datetime.now(UTC)
        self._record.tags["updated_by"] = self.worker
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(self._record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def _persist(self) -> None:
        self.persist()

    @classmethod
    def load(cls, path: Path) -> RunRecord:
        with path.open("r", encoding="utf-8") as handle:
            return RunRecord.model_validate(json.load(handle))
