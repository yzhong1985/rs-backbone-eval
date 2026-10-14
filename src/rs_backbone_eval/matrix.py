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


def canonical_runs(runs: list[RunRecord]) -> list[RunRecord]:
    """Keep one run per seed: a retry of a seed is not an extra observation.

    When a seed ran more than once the most recently updated run is canonical,
    so the mean and standard deviation are never inflated by duplicates and are
    never chosen on the test score.
    """
    by_seed: dict[int, RunRecord] = {}
    for record in runs:
        current = by_seed.get(record.seed)
        if current is None or (record.updated_at or "") >= (current.updated_at or ""):
            by_seed[record.seed] = record
    return [by_seed[seed] for seed in sorted(by_seed)]


def mean_validation(runs: list[RunRecord]) -> float:
    """Mean of the validation scores recorded by checkpoint selection."""
    scores = [
        float(record.tags["validation_mean_iou"])
        for record in runs
        if record.tags.get("validation_mean_iou")
    ]
    return statistics.mean(scores) if scores else float("-inf")


def _rank(candidate: dict[str, Any]) -> tuple:
    """Best recorded validation score wins, then the variant with most seeds.

    ``selected`` only means the track selected a checkpoint on validation;
    among candidates that all did, the recorded validation score decides, so
    the headline variant is not chosen by seed count or by input order.
    """
    return (
        candidate["selected"],
        candidate["mean_validation"],
        candidate["n"],
        -len(candidate["runs"]),
    )


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
        runs = canonical_runs(runs)
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
            "mean_validation": mean_validation(runs),
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


DIMENSIONS = [
    {
        "id": "D1",
        "name": "任务效果",
        "measure": "各任务上相对最优模型的比例（均值），1 = 每个任务都是最优",
    },
    {
        "id": "D2",
        "name": "输入灵活性",
        "measure": "输入扰动（波段缺失、分辨率、时相）下的性能保持率",
    },
    {
        "id": "D3",
        "name": "适配成本",
        "measure": "B1 训练耗时的相对值（越省越高），并列出参数量、显存、微调增益",
    },
    {"id": "D4", "name": "稳定性", "measure": "B1 各任务种子标准差的相对值（越稳越高）"},
]


def scorecard(
    matrix: dict[str, Any], records: list[RunRecord], names: dict[str, str]
) -> dict[str, Any]:
    """Quantify the four backbone dimensions from the same cells the matrix shows.

    Scores are relative within the compared models: they change when models are
    added, so they are only comparable inside one report.
    """
    by_id = {record.run_id: record for record in records}
    cells = matrix["cells"]
    models = sorted({cell["model"] for cell in cells})
    best: dict[tuple[str, str], float] = {}
    for cell in cells:
        key = (cell["track"], cell["task"])
        best[key] = max(best.get(key, float("-inf")), cell["mean"])

    def track_cells(model: str, track: str) -> list[dict[str, Any]]:
        return [c for c in cells if c["model"] == model and c["track"] == track]

    def run_list(model: str, track: str) -> list[RunRecord]:
        found = []
        for cell in track_cells(model, track):
            found.extend(by_id[run_id] for run_id in cell["runs"] if run_id in by_id)
        return found

    rows = []
    for model in models:
        b1 = track_cells(model, "B1")
        b2 = track_cells(model, "B2")
        all_tasks = sorted({c["task"] for c in cells})
        d1_b1 = [c["mean"] / best[("B1", c["task"])] for c in b1 if best[("B1", c["task"])] > 0]
        d1_b2 = [c["mean"] / best[("B2", c["task"])] for c in b2 if best[("B2", c["task"])] > 0]
        b1_runs = run_list(model, "B1")
        b2_runs = run_list(model, "B2")
        wall = [r.training.wall_time_seconds for r in b1_runs if r.training.wall_time_seconds]
        peak = [r.training.peak_gpu_memory_mb for r in b1_runs if r.training.peak_gpu_memory_mb]
        sds = [c["sd"] for c in b1 if c["sd"] is not None]
        gains = {}
        for c in b1:
            match = [x for x in b2 if x["task"] == c["task"]]
            if match:
                gains[c["task"]] = round(c["mean"] - match[0]["mean"], 4)
        trainable = [
            r.training.trainable_parameters for r in b2_runs if r.training.trainable_parameters
        ]
        total = [r.training.total_parameters for r in b2_runs if r.training.total_parameters]
        rows.append(
            {
                "model": model,
                "name": names.get(model, model),
                "d1_b1": {
                    "score": round(sum(d1_b1) / len(d1_b1), 4) if d1_b1 else None,
                    "coverage": f"{len(b1)}/{len(all_tasks)}",
                },
                "d1_b2": {
                    "score": round(sum(d1_b2) / len(d1_b2), 4) if d1_b2 else None,
                    "coverage": f"{len(b2)}/{len(all_tasks)}",
                },
                "d2": {"score": None, "note": "待测：需要输入扰动实验"},
                "d3": {
                    "b1_wall_min_mean": round(statistics.mean(wall) / 60, 1) if wall else None,
                    "b1_peak_mem_mb": round(max(peak)) if peak else None,
                    "b2_trainable_params": max(trainable) if trainable else None,
                    "b2_total_params": max(total) if total else None,
                    "adaptation_gain": gains,
                },
                "d4": {"mean_sd_b1": round(statistics.mean(sds), 4) if sds else None},
            }
        )

    walls = [r["d3"]["b1_wall_min_mean"] for r in rows if r["d3"]["b1_wall_min_mean"]]
    sds_all = [r["d4"]["mean_sd_b1"] for r in rows if r["d4"]["mean_sd_b1"] is not None]
    for row in rows:
        wall = row["d3"]["b1_wall_min_mean"]
        row["d3"]["score"] = round(1 - wall / max(walls), 4) if wall and walls else None
        sd = row["d4"]["mean_sd_b1"]
        row["d4"]["score"] = round(1 - sd / max(sds_all), 4) if sd is not None and sds_all else None
    return {"dimensions": DIMENSIONS, "models": rows}
