"""Filesystem-backed immutable run repository used by the CLI and Portal."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import RunRecord


class RunNotFoundError(KeyError):
    pass


class RunRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _record_paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*/run.json"))

    @staticmethod
    def load_path(path: Path) -> RunRecord:
        with path.open("r", encoding="utf-8") as handle:
            return RunRecord.model_validate(json.load(handle))

    def list_records(self) -> tuple[list[RunRecord], list[dict[str, str]]]:
        records: list[RunRecord] = []
        errors: list[dict[str, str]] = []
        for path in self._record_paths():
            try:
                records.append(self.load_path(path))
            except Exception as exc:  # Portal must keep serving valid runs.
                errors.append({"path": str(path), "error": str(exc)})
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records, errors

    def get(self, run_id: str) -> RunRecord:
        candidate = (self.root / run_id / "run.json").resolve()
        if candidate.parent.parent != self.root or not candidate.is_file():
            raise RunNotFoundError(run_id)
        return self.load_path(candidate)

    def artifact_path(self, run_id: str, relative_path: str) -> Path:
        run_root = (self.root / run_id).resolve()
        candidate = (run_root / relative_path).resolve()
        if run_root not in candidate.parents or not candidate.is_file():
            raise RunNotFoundError(f"{run_id}/{relative_path}")
        return candidate

    def summary(self) -> dict[str, Any]:
        records, errors = self.list_records()
        return {
            "run_count": len(records),
            "invalid_run_count": len(errors),
            "status_counts": dict(Counter(record.status for record in records)),
            "task_ids": sorted({record.task_id for record in records}),
            "model_ids": sorted({record.model_id for record in records}),
            "protocol_ids": sorted({record.protocol_id for record in records}),
            "errors": errors,
        }
