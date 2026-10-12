"""Atomic run.json writer implementing the control-plane run contract.

Workers own the run record lifecycle: create it as ``running``, update fields
as the action progresses, and settle on ``completed`` or ``failed``. Every
write goes through a ``run.json.tmp`` file followed by an atomic replace so
the Portal never observes a partially written record.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rs_backbone_eval.contracts import (
    DatasetRecord,
    InputRecord,
    MetricRecord,
    RunRecord,
    TrainingRecord,
)
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec


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
    ) -> None:
        self.run_dir = run_dir
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

    def update(self, **changes: Any) -> None:
        """Apply attribute updates (e.g. status=, notes=) and persist atomically.

        ``tags`` entries are merged into the existing dict; all other fields
        are replaced.
        """
        for name, value in changes.items():
            if name == "tags" and isinstance(value, dict):
                self._record.tags.update(value)
            else:
                setattr(self._record, name, value)
        self._record.updated_at = datetime.now(UTC)
        self._record.tags["updated_by"] = "prithvi_worker"
        self.flush()

    def add_metrics(self, metrics: list[MetricRecord]) -> None:
        self._record.metrics.extend(metrics)
        self._record.updated_at = datetime.now(UTC)
        self.flush()

    def complete(self, notes: str | None = None) -> None:
        self.update(status="completed", notes=notes)

    def fail(self, message: str) -> None:
        self.update(status="failed", notes=message)

    def flush(self) -> None:
        if not self._record.run_id:
            raise ValueError("run_id is required before flushing")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(self._record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    @classmethod
    def load(cls, path: Path) -> RunRecord:
        with path.open("r", encoding="utf-8") as handle:
            return RunRecord.model_validate(json.load(handle))
