"""Track x model x task summary for the Portal matrix and the Markdown report.

Tasks and models are discovered from the run records, so new tasks or models
appear without code changes. Tracks are fixed by the design (B0-B3).
"""

from __future__ import annotations

import contextlib
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


def metric_direction(metric_name: str, records: list[RunRecord]) -> str:
    """Direction of a task metric, read from the run records (default maximize)."""
    for record in records:
        metric = record.primary_metric()
        if metric is not None and metric.name == metric_name:
            return metric.direction
    return "maximize"


def is_better(value: float, other: float, direction: str) -> bool:
    """Whether ``value`` beats ``other`` under the metric direction."""
    return value < other if direction == "minimize" else value > other


def ratio_to_best(value: float, best: float, direction: str) -> float | None:
    """Relative-to-best score in (0, 1]; 1 means best on that task, direction aware."""
    if direction == "minimize":
        return best / value if value else None
    return value / best if best else None


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
            "direction": metric_direction(metric_name, runs),
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
        "name": "表征与迁移效果",
        "measure": "各任务相对最优模型的比例（方向感知，1 = 该任务最优），"
        "冻结与微调分列；辅以少标注曲线与监督基线增益",
    },
    {
        "id": "D2",
        "name": "泛化与输入鲁棒性",
        "measure": "扰动后的绝对成绩、下降幅度与保持率；缺波段/分辨率/时相，以及地理与时间 holdout",
    },
    {
        "id": "D3",
        "name": "使用效率",
        "measure": "分阶段的 GPU 小时、峰值显存、参数量与缓存体积；不做相对归零",
    },
    {
        "id": "D4",
        "name": "稳定性与可靠性",
        "measure": "各种子原始成绩、标准差、失败率与最差表现；不做相对归零",
    },
]


def perturbation_summary(records: list[RunRecord]) -> list[dict[str, Any]]:
    """Paired perturbation results: absolute score, drop and retention.

    A perturbation run records the clean evaluation as its primary metric and
    each perturbation as a non-primary metric on the same run, so baseline and
    perturbed score come from the same trained model (a paired comparison).
    Direction is respected for minimize metrics such as RMSE.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.status != "completed":
            continue
        primary = record.primary_metric()
        if primary is None:
            continue
        for metric in record.metrics:
            if metric.is_primary or metric.name != primary.name or metric.split == primary.split:
                continue
            baseline, perturbed = primary.value, metric.value
            if primary.direction == "minimize":
                drop = perturbed - baseline
                retention = baseline / perturbed if perturbed else None
            else:
                drop = baseline - perturbed
                retention = perturbed / baseline if baseline else None
            rows.append(
                {
                    "model": record.model_id,
                    "task": record.task_id,
                    "track": track_of(record),
                    "seed": record.seed,
                    "run_id": record.run_id,
                    "perturbation": metric.split,
                    "metric": metric.name,
                    "direction": primary.direction,
                    "baseline": baseline,
                    "perturbed": perturbed,
                    "drop": drop,
                    "retention": retention,
                }
            )
    return rows


def _d2_by_model(perturbations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in perturbations:
        buckets[(row["model"], row["task"], row["perturbation"])].append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, task, perturbation), items in sorted(buckets.items()):
        grouped[model].append(
            {
                "task": task,
                "perturbation": perturbation,
                "baseline": round(statistics.mean(i["baseline"] for i in items), 4),
                "perturbed": round(statistics.mean(i["perturbed"] for i in items), 4),
                "drop": round(statistics.mean(i["drop"] for i in items), 4),
                "retention": (
                    round(statistics.mean(i["retention"] for i in items), 4)
                    if all(i["retention"] is not None for i in items)
                    else None
                ),
                "seeds": sorted({i["seed"] for i in items}),
            }
        )
    return grouped


def _mark_pareto(rows: list[dict[str, Any]]) -> None:
    """Flag accuracy-vs-cost Pareto-optimal models instead of a forced-zero score."""
    points = [
        (row["d1_b1"]["score"], row["d3"]["b1_wall_min_mean"])
        for row in rows
        if row["d1_b1"]["score"] is not None and row["d3"]["b1_wall_min_mean"] is not None
    ]
    for row in rows:
        accuracy, cost = row["d1_b1"]["score"], row["d3"]["b1_wall_min_mean"]
        if accuracy is None or cost is None:
            row["pareto"] = None
            continue
        row["pareto"] = not any(
            other_accuracy >= accuracy
            and other_cost <= cost
            and (other_accuracy > accuracy or other_cost < cost)
            for other_accuracy, other_cost in points
        )


def scorecard(
    matrix: dict[str, Any], records: list[RunRecord], names: dict[str, str]
) -> dict[str, Any]:
    """Quantify the four dimensions from the same cells the matrix shows.

    D1 is relative to the best model per task and changes when models are added,
    so it is only comparable inside one report. D3 and D4 report raw values (no
    "worst maps to zero" rescaling) and the accuracy-cost Pareto front is shown
    instead of a composite total.
    """
    by_id = {record.run_id: record for record in records}
    cells = matrix["cells"]
    models = sorted({cell["model"] for cell in cells})
    all_tasks = sorted({cell["task"] for cell in cells})
    perturbations = _d2_by_model(perturbation_summary(records))

    best: dict[tuple[str, str], tuple[float, str]] = {}
    for cell in cells:
        key = (cell["track"], cell["task"])
        direction = cell.get("direction", "maximize")
        current = best.get(key)
        if current is None or is_better(cell["mean"], current[0], direction):
            best[key] = (cell["mean"], direction)

    def track_cells(model: str, track: str) -> list[dict[str, Any]]:
        return [c for c in cells if c["model"] == model and c["track"] == track]

    def run_list(model: str, track: str) -> list[RunRecord]:
        found = []
        for cell in track_cells(model, track):
            found.extend(by_id[run_id] for run_id in cell["runs"] if run_id in by_id)
        return found

    def d1_ratios(model: str, track: str) -> list[float]:
        out = []
        for cell in track_cells(model, track):
            reference = best.get((track, cell["task"]))
            if reference and reference[0]:
                ratio = ratio_to_best(cell["mean"], reference[0], reference[1])
                if ratio is not None:
                    out.append(ratio)
        return out

    rows = []
    for model in models:
        b1, b2 = track_cells(model, "B1"), track_cells(model, "B2")
        d1_b1, d1_b2 = d1_ratios(model, "B1"), d1_ratios(model, "B2")
        b1_runs, b2_runs = run_list(model, "B1"), run_list(model, "B2")
        wall = [r.training.wall_time_seconds for r in b1_runs if r.training.wall_time_seconds]
        peak = [r.training.peak_gpu_memory_mb for r in b1_runs if r.training.peak_gpu_memory_mb]
        sds = [c["sd"] for c in b1 if c["sd"] is not None]
        gains = {}
        for cell in b1:
            match = [x for x in b2 if x["task"] == cell["task"]]
            if match:
                gains[cell["task"]] = round(cell["mean"] - match[0]["mean"], 4)
        trainable = [
            r.training.trainable_parameters for r in b2_runs if r.training.trainable_parameters
        ]
        total = [r.training.total_parameters for r in b2_runs if r.training.total_parameters]
        model_runs = [r for r in records if r.model_id == model]
        failed = [r for r in model_runs if r.status == "failed"]
        rows.append(
            {
                "model": model,
                "name": names.get(model, model),
                "d1_b1": {
                    "score": round(statistics.mean(d1_b1), 4) if d1_b1 else None,
                    "coverage": f"{len(b1)}/{len(all_tasks)}",
                    "per_task": {c["task"]: c["mean"] for c in b1},
                },
                "d1_b2": {
                    "score": round(statistics.mean(d1_b2), 4) if d1_b2 else None,
                    "coverage": f"{len(b2)}/{len(all_tasks)}",
                    "per_task": {c["task"]: c["mean"] for c in b2},
                },
                "d2": {
                    "results": perturbations.get(model, []),
                    "note": None if perturbations.get(model) else "未测：需要输入扰动实验",
                },
                "d3": {
                    "b1_wall_min_mean": round(statistics.mean(wall) / 60, 1) if wall else None,
                    "b1_peak_mem_mb": round(max(peak)) if peak else None,
                    "b2_trainable_params": max(trainable) if trainable else None,
                    "b2_total_params": max(total) if total else None,
                    "adaptation_gain": gains,
                },
                "d4": {
                    "mean_sd_b1": round(statistics.mean(sds), 4) if sds else None,
                    "per_task_sd": {c["task"]: c["sd"] for c in b1},
                    "failed_runs": len(failed),
                    "total_runs": len(model_runs),
                },
            }
        )

    _mark_pareto(rows)
    return {"dimensions": DIMENSIONS, "models": rows}


def compatibility_matrix(records: list[RunRecord], names: dict[str, str]) -> list[dict[str, Any]]:
    """Discrete input-compatibility matrix: sensor/bands/timesteps per model.

    ``support`` is "adapter" when a run trained an explicit input adapter
    (recorded ``adapter_parameters`` > 0), "native" when a completed run used
    the configuration directly, and "unknown" otherwise; forward execution
    alone is not treated as validated support.
    """
    buckets: dict[tuple, dict[str, Any]] = {}
    for record in records:
        key = (
            record.model_id,
            record.input.sensor,
            tuple(record.input.bands),
            record.input.time_steps,
        )
        entry = buckets.setdefault(
            key,
            {
                "model": record.model_id,
                "name": names.get(record.model_id, record.model_id),
                "sensor": record.input.sensor,
                "bands": list(record.input.bands),
                "time_steps": record.input.time_steps,
                "tasks": set(),
                "tracks": set(),
                "adapter_parameters": None,
                "completed": False,
            },
        )
        entry["tasks"].add(record.task_id)
        entry["tracks"].add(track_of(record))
        adapter = record.tags.get("adapter_parameters")
        if adapter is not None:
            with contextlib.suppress(ValueError):
                entry["adapter_parameters"] = int(adapter)
        if record.status == "completed":
            entry["completed"] = True

    rows = []
    for key in sorted(buckets):
        entry = buckets[key]
        entry["support"] = (
            "adapter"
            if entry["adapter_parameters"]
            else ("native" if entry["completed"] else "unknown")
        )
        entry["tasks"] = sorted(entry["tasks"])
        entry["tracks"] = sorted(entry["tracks"])
        rows.append(entry)
    return rows
