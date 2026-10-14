"""Shared worker runtime: atomic run.json writer and event emission.

Workers own the run record lifecycle: create it as ``running``, update fields
as the action progresses, and settle on ``completed`` or ``failed``. Every
write goes through a ``run.json.tmp`` file followed by an atomic replace so
the Portal never observes a partially written record.

The three model workers share this module through the root package so their
run records and progress events follow one contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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


def _code_version() -> str | None:
    """Commit of the repository this code runs from; ``-dirty`` when uncommitted changes exist."""
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if status else commit


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
            git_commit=_code_version(),
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
        label_fraction = os.environ.get("RS_EVAL_LABEL_FRACTION")
        if label_fraction:
            self._record.tags["label_fraction"] = label_fraction

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

    def add_confusion(
        self,
        groups: dict[str, list[list[int]]],
        num_classes: int,
        filename: str = "confusion_per_scene.json",
    ) -> None:
        """Persist one confusion matrix per group so IoU intervals can be bootstrapped.

        Storing per-group (region/tile/event) matrices lets the report resample
        groups and re-accumulate a global matrix instead of bootstrapping
        per-sample values, which answers a different question.
        """
        payload = {"num_classes": num_classes, "groups": groups}
        (self.run_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        self.add_artifact("confusion_matrix", filename, "per-group confusion matrix")

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

    def attach_checkpoint(self, path: Path) -> None:
        """Record the SHA-256 of the model checkpoint actually loaded by this run."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        self._record.model_checkpoint_sha256 = digest.hexdigest()
        self.persist()

    @classmethod
    def load(cls, path: Path) -> RunRecord:
        with path.open("r", encoding="utf-8") as handle:
            return RunRecord.model_validate(json.load(handle))


def load_confusion(path: Path) -> tuple[int, list[tuple[str, list[list[int]]]]]:
    """Read a per-group confusion artifact written by ``RunWriter.add_confusion``.

    Returns ``(num_classes, [(group_id, counts), ...])`` sorted by group id so
    two runs can be aligned group-by-group for a paired bootstrap.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["num_classes"]), sorted(payload["groups"].items())
