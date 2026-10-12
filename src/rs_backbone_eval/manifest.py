"""Streaming validation and split leakage audit for external JSONL manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .contracts import ManifestSample


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


def audit_manifest(path: Path, max_errors: int = 100) -> ManifestAudit:
    audit = ManifestAudit()
    sample_ids: set[tuple[str, str]] = set()
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                sample = ManifestSample.model_validate(json.loads(raw_line))
            except (json.JSONDecodeError, ValidationError) as exc:
                if len(audit.errors) < max_errors:
                    audit.errors.append(f"line {line_number}: {exc}")
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
