"""Unified sample schema and manifest-driven sequence loading.

``GeoSequenceSample`` is the in-memory object described by the evaluation
design: every date of one sample is a single unit, carries its own temporal
mask, and never leaks across splits. Raster-level label masks (nodata, invalid
pixels) are derived by model workers when they open the label files; the
control plane never opens payloads.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from ..contracts import ManifestSample
from ..registry import TaskSpec


class GeoSequenceSample(ManifestSample):
    """In-memory sample with all dates of one location kept together."""

    @property
    def time_count(self) -> int:
        """Number of dates. A single stacked raster may hold all of them."""
        return len(self.timestamps)

    @property
    def valid_timestamps(self) -> list[str | int]:
        return [
            stamp for stamp, valid in zip(self.timestamps, self.temporal_mask, strict=True) if valid
        ]


def validate_sample(sample: GeoSequenceSample, task_spec: TaskSpec) -> list[str]:
    """Return every reason this sample cannot enter training for the task."""
    errors: list[str] = []
    if sample.task_id != task_spec.task_id:
        errors.append(f"task_id {sample.task_id} != {task_spec.task_id}")
    if sample.sensor != task_spec.sensor:
        errors.append(f"sensor {sample.sensor} != {task_spec.sensor}")
    if sample.bands != task_spec.bands:
        errors.append(f"bands {sample.bands} != {task_spec.bands}")
    if sample.time_count != task_spec.time_steps:
        errors.append(f"{sample.time_count} frames != {task_spec.time_steps}")
    if sample.split not in task_spec.allowed_splits:
        errors.append(f"split {sample.split} not in {task_spec.allowed_splits}")
    return errors


class SampleValidationError(ValueError):
    """Raised when a manifest row violates the task registry contract."""


class SequenceDataset:
    """Stream validated samples from one manifest, optionally split-filtered."""

    def __init__(self, manifest_path: Path, task_spec: TaskSpec | None = None) -> None:
        self.manifest_path = manifest_path
        self.task_spec = task_spec

    def __iter__(self) -> Iterator[GeoSequenceSample]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    sample = GeoSequenceSample.model_validate(json.loads(raw_line))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise SampleValidationError(f"line {line_number}: {exc}") from exc
                if self.task_spec is not None:
                    errors = validate_sample(sample, self.task_spec)
                    if errors:
                        raise SampleValidationError(
                            f"line {line_number} {sample.sample_id}: {'; '.join(errors)}"
                        )
                yield sample

    def by_split(self, split: str) -> Iterator[GeoSequenceSample]:
        for sample in self:
            if sample.split == split:
                yield sample

    def split_counts(self) -> Counter[str]:
        return Counter(sample.split for sample in self)

    def sample_count(self) -> int:
        return sum(1 for _ in self)
