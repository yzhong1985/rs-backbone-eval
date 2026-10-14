"""Track x model x task summary for the Portal matrix and the Markdown report.

Tasks and models are discovered from the run records, so new tasks or models
appear without code changes. Tracks are fixed by the design (B0-B3).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .contracts import RunRecord

PROTOCOL_TRACKS = {
    "prithvi_repo_config": "B0",
    "prithvi_paper_protocol": "B0",
    "full_finetune": "B1",
    "frozen_probe": "B2",
    "frozen_adapter": "B2",
    "native_input": "B3",
}
TRACK_NAMES = {
    "B0": "Prithvi 参考复现",
    "B1": "全量微调（主结果）",
    "B2": "冻结表征（辅助）",
    "B3": "原生输入（能力上限）",
}
TRACK_ORDER = ["B0", "B1", "B2", "B3"]
TASK_NAMES = {
    "T1": "洪水水体（Sen1Floods11）",
    "T2": "烧毁区（HLS Burn Scars）",
    "T5": "多时相作物（US Multi-temporal Crop）",
}


def track_of(record: RunRecord) -> str:
    return PROTOCOL_TRACKS.get(record.protocol_id, "?")


def model_names(config_root: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for path in sorted((config_root / "models").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("model_id"):
            names[data["model_id"]] = data.get("display_name", data["model_id"])
    return names


def variant_key(record: RunRecord) -> tuple[str, str, str, str]:
    return (
        record.training.head_id,
        record.input.preprocessing_id,
        record.tags.get("learning_rate", "-"),
        record.tags.get("encoder_input_px", "-"),
    )


def _rank(candidate: dict[str, Any]) -> tuple:
    """Validation-selected variants win; then the variant with the most seeds."""
    return (candidate["selected"], candidate["n"], -len(candidate["runs"]))


def build_matrix(records: list[RunRecord], names: dict[str, str]) -> dict[str, Any]:
    """Pick one headline variant per cell: the one with the most distinct seeds."""
    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    tasks: dict[str, str] = {}
    for record in records:
        metric = record.primary_metric()
        if record.status != "completed" or metric is None:
            continue
        tasks.setdefault(record.task_id, metric.name)
        key = (track_of(record), record.model_id, record.task_id, record.protocol_id, metric.name)
        groups[key + variant_key(record)].append(record)

    cells: dict[tuple, dict[str, Any]] = {}
    variant_count: dict[tuple, int] = defaultdict(int)
    for key, runs in groups.items():
        track, model_id, task_id, protocol_id, metric_name = key[:5]
        head_id, preprocessing_id, lr, input_px = key[5:]
        seeds = sorted({r.seed for r in runs})
        values = [r.primary_metric().value for r in runs]
        candidate = {
            "protocol": protocol_id,
            "head": head_id,
            "preprocessing": preprocessing_id,
            "learning_rate": lr,
            "input_px": input_px,
            "seeds": seeds,
            "values": {
                str(r.seed): r.primary_metric().value for r in sorted(runs, key=lambda r: r.seed)
            },
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(seeds),
            "runs": [r.run_id for r in runs],
            "selected": any(r.tags.get("selection") for r in runs),
        }
        cell_key = (track, model_id, task_id)
        variant_count[cell_key] += 1
        current = cells.get(cell_key)
        if current is None or _rank(candidate) > _rank(current):
            cells[cell_key] = candidate
            candidate["metric"] = metric_name
    for cell_key, value in cells.items():
        value["variants"] = variant_count[cell_key]

    models = sorted({model for _, model, _ in cells})
    rows = [
        {"track": track, "model": model, "name": names.get(model, model)}
        for track in TRACK_ORDER
        for model in models
        if any(t == track and m == model for t, m, _ in cells)
    ]
    return {
        "tracks": [{"id": t, "name": TRACK_NAMES[t]} for t in TRACK_ORDER],
        "tasks": [
            {"id": task, "name": TASK_NAMES.get(task, task), "metric": metric}
            for task, metric in sorted(tasks.items())
        ],
        "rows": rows,
        "cells": [
            {"track": t, "model": m, "task": k, **value}
            for (t, m, k), value in sorted(cells.items())
        ],
    }
