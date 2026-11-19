"""Manifest writing, content hashing, and task-aware split audits.

Manifests are canonical JSONL files stored outside the repository. The hash of
the file content is the ``manifest_sha256`` recorded in every run record, so
writing must be deterministic: rows are sorted by ``(task_id, sample_id)`` and
serialized with stable separators.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ..contracts import ManifestSample
from ..registry import TaskSpec


@dataclass
class ManifestAudit:
    sample_count: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    task_counts: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    leakage: dict[str, list[str]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors and not self.leakage


@dataclass
class TaskManifestAudit:
    task_id: str
    sample_count: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    schema_errors: list[str] = field(default_factory=list)
    duplicate_samples: list[str] = field(default_factory=list)
    band_mismatches: list[str] = field(default_factory=list)
    sensor_mismatches: list[str] = field(default_factory=list)
    time_step_mismatches: list[str] = field(default_factory=list)
    unknown_splits: list[str] = field(default_factory=list)
    missing_splits: list[str] = field(default_factory=list)
    leakage: dict[str, list[str]] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return (
            len(self.schema_errors)
            + len(self.duplicate_samples)
            + len(self.band_mismatches)
            + len(self.sensor_mismatches)
            + len(self.time_step_mismatches)
            + len(self.unknown_splits)
            + len(self.missing_splits)
            + len(self.leakage)
        )

    @property
    def valid(self) -> bool:
        return self.error_count == 0

    def all_errors(self) -> list[str]:
        errors = [
            *self.schema_errors,
            *(f"duplicate sample {item}" for item in self.duplicate_samples),
            *(f"band mismatch {item}" for item in self.band_mismatches),
            *(f"sensor mismatch {item}" for item in self.sensor_mismatches),
            *(f"time-step mismatch {item}" for item in self.time_step_mismatches),
            *(f"unknown split {item}" for item in self.unknown_splits),
            *(f"missing split {item}" for item in self.missing_splits),
            *(f"leakage {group}: {', '.join(splits)}" for group, splits in self.leakage.items()),
        ]
        return errors


def _iter_rows(path: Path, errors: list[str], max_errors: int):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                yield line_number, ManifestSample.model_validate(json.loads(raw_line))
            except (json.JSONDecodeError, ValidationError) as exc:
                if len(errors) < max_errors:
                    errors.append(f"line {line_number}: {exc}")
                yield line_number, None


def audit_manifest(path: Path, max_errors: int = 100) -> ManifestAudit:
    """Stream a manifest, validate every row, and reject group leakage."""
    audit = ManifestAudit()
    sample_ids: set[tuple[str, str]] = set()
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for line_number, sample in _iter_rows(path, audit.errors, max_errors):
        if sample is None:
            continue
        key = (sample.task_id, sample.sample_id)
        if key in sample_ids and len(audit.errors) < max_errors:
            audit.errors.append(f"line {line_number}: duplicate sample {key}")
        sample_ids.add(key)
        group_splits[(sample.task_id, sample.group_id)].add(sample.split)
        audit.sample_count += 1
        audit.split_counts[sample.split] += 1
        audit.task_counts[sample.task_id] += 1

    audit.leakage = {
        f"{task_id}/{group_id}": sorted(splits)
        for (task_id, group_id), splits in group_splits.items()
        if len(splits) > 1
    }
    return audit


def audit_task_manifest(
    path: Path, task_spec: TaskSpec, max_errors: int = 100
) -> TaskManifestAudit:
    """Validate a manifest against its TaskSpec: bands, sensor, time steps, splits."""
    audit = TaskManifestAudit(task_id=task_spec.task_id)
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)

    for line_number, sample in _iter_rows(path, audit.schema_errors, max_errors):
        if sample is None:
            continue
        if sample.task_id != task_spec.task_id:
            audit.schema_errors.append(
                f"line {line_number}: task_id {sample.task_id} does not match {task_spec.task_id}"
            )
            continue
        if sample.sample_id in sample_ids:
            audit.duplicate_samples.append(sample.sample_id)
        sample_ids.add(sample.sample_id)
        group_splits[sample.group_id].add(sample.split)
        audit.sample_count += 1
        audit.split_counts[sample.split] += 1

        if sample.sensor != task_spec.sensor:
            audit.sensor_mismatches.append(
                f"line {line_number} {sample.sample_id}: {sample.sensor} != {task_spec.sensor}"
            )
        if sample.bands != task_spec.bands:
            audit.band_mismatches.append(
                f"line {line_number} {sample.sample_id}: {sample.bands} != {task_spec.bands}"
            )
        if len(sample.timestamps) != task_spec.time_steps:
            audit.time_step_mismatches.append(
                f"line {line_number} {sample.sample_id}: "
                f"{len(sample.timestamps)} dates != {task_spec.time_steps}"
            )
        if sample.split not in task_spec.allowed_splits:
            audit.unknown_splits.append(f"line {line_number} {sample.sample_id}: {sample.split}")

    audit.missing_splits = [
        split for split in task_spec.required_splits if audit.split_counts.get(split, 0) == 0
    ]
    audit.leakage = {
        group_id: sorted(splits) for group_id, splits in group_splits.items() if len(splits) > 1
    }
    return audit


class GroupLeakageError(ValueError):
    """A group of one task has rows in more than one split."""


class ManifestRowError(ValueError):
    """A row of the task being trained is malformed, or its task cannot be read."""


def _task_rows_strict(path: Path, task_id: str) -> list[ManifestSample]:
    """Every row of ``task_id``, each validated; other tasks' rows are skipped.

    Unlike ``audit_manifest``, which drops rows that fail the schema, a malformed row
    here is an error: its group could otherwise be trained on and tested unseen.
    A line that is not a JSON object has no readable task_id, so it is also an error.
    """
    rows: list[ManifestSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ManifestRowError(
                    f"{path.name} line {line_number}: not valid JSON ({exc.msg}); "
                    "its task cannot be read"
                ) from exc
            if not isinstance(payload, dict):
                raise ManifestRowError(f"{path.name} line {line_number}: row is not a JSON object")
            if payload.get("task_id") != task_id:
                continue
            try:
                rows.append(ManifestSample.model_validate(payload))
            except ValidationError as exc:
                raise ManifestRowError(
                    f"{path.name} line {line_number}: {task_id} row fails the manifest "
                    f"schema, refusing to train on it: {exc}"
                ) from exc
    return rows


def require_no_group_leakage(path: Path, task_id: str, max_groups: int = 10) -> None:
    """Refuse a manifest in which a group of ``task_id`` spans more than one split.

    The group (sheet, AOI, ...) is the unit that must stay in one split, as in
    ``audit_manifest``. Training entry points call this before using rows. Rows of
    the task that fail the schema raise ``ManifestRowError``; rows of other tasks
    in a shared manifest are ignored.
    """
    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample in _task_rows_strict(path, task_id):
        group_splits[sample.group_id].add(sample.split)
    leaks = [(group, sorted(splits)) for group, splits in group_splits.items() if len(splits) > 1]
    if not leaks:
        return
    shown = "; ".join(f"{group} in {', '.join(splits)}" for group, splits in leaks[:max_groups])
    raise GroupLeakageError(
        f"{task_id} manifest {path.name}: {len(leaks)} group(s) span several splits "
        f"({shown}); refusing to train on leaked groups"
    )


def _canonical_row(sample: ManifestSample) -> str:
    return json.dumps(sample.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def write_manifest(path: Path, samples: Iterable[ManifestSample]) -> str:
    """Write a canonical manifest and return the SHA-256 of the file content."""
    rows = sorted(_canonical_row(sample) for sample in samples)
    payload = "\n".join(rows) + "\n" if rows else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return manifest_sha256(path)


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
