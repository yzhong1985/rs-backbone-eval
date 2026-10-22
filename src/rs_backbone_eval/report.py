"""Markdown experiment report built from run records, manifests and the download catalog.

The report is a snapshot: it reads the external run repository and never writes to it.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .audit import stale_run_ids
from .contracts import RunRecord
from .evaluation.bootstrap import bootstrap_confusion, bootstrap_confusion_paired
from .evaluation.segmentation import multiclass_metrics
from .matrix import (
    PROTOCOL_TRACKS as TRACKS,
)
from .matrix import (
    TRACK_NAMES,
    build_matrix,
    canonical_runs,
    compatibility_matrix,
    decode_variant_tail,
    is_primary_panel,
    is_superseded,
    mean_validation,
    model_names,
    scorecard,
    variant_key,
)
from .results import RunRepository
from .worker_runtime import load_confusion

TASK_NAMES = {
    "T1": "洪水水体分割（Sen1Floods11）",
    "T2": "烧毁区分割（HLS Burn Scars）",
    "T3": "烧毁强度分类（Burn Intensity）",
    "T4": "滑坡检测（Landslide4Sense）",
    "T5": "多时相作物分类（US Multi-temporal Crop）",
}
TASK_DATASET_IDS = {
    "T1": "data_sen1floods11",
    "T2": "data_hls_burn_scars",
    "T3": "data_burn_intensity",
    "T4": "data_landslide4sense",
    "T5": "data_us_multicrop",
}
OFFICIAL_EPOCHS = {"T1": 50, "T2": 50, "T3": 50, "T4": 50, "T5": 125}
MODEL_NAMES = {
    "prithvi_eo_v2_300_tl": "Prithvi-EO-2.0 300M-TL",
    "clay_v1_5": "Clay v1.5",
    "anysat_base": "AnySat (base)",
    "olmoearth_v1_2_tiny": "OlmoEarth v1.2 Tiny",
    "olmoearth_v1_2_base": "OlmoEarth v1.2 Base",
}


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest_summary(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Per task: manifest file name plus sample counts per split."""
    result: dict[str, list[dict[str, Any]]] = {}
    root = data_root / "manifests"
    if not root.exists():
        return result
    for path in sorted(root.glob("*/*.jsonl")):
        counts: dict[str, int] = defaultdict(int)
        task_id = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    counts[json.loads(line)["split"]] += 1
        result.setdefault(task_id, []).append(
            {"file": path.name, "sha256_prefix": path.stem[:16], "counts": dict(counts)}
        )
    return result


EVAL_SPLITS = {
    "T1": "test",
    "T2": "official_validation",
    "T3": "test",
    "T4": "test",
    "T5": "official_validation",
}


def _split_overlap(data_root: Path, task_id: str) -> dict[str, int] | None:
    """How many evaluation groups also occur in the training split.

    A shared group id means the same scene/tile/event contributes to training
    and evaluation, so the score cannot support an "unseen geography" claim.
    """
    root = data_root / "manifests" / task_id
    if not root.is_dir():
        return None
    paths = sorted(root.glob("*.jsonl"))
    if not paths:
        return None
    eval_split = EVAL_SPLITS.get(task_id)
    if eval_split is None:
        return None
    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    train_images: set[str] = set()
    eval_images: set[str] = set()
    with paths[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "train":
                train_groups.add(row.get("group_id", ""))
                train_images.update(row.get("image_paths") or [])
            elif row.get("split") == eval_split:
                eval_groups.add(row.get("group_id", ""))
                eval_images.update(row.get("image_paths") or [])
    return {
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "shared_groups": len(eval_groups & train_groups),
        "shared_images": len(eval_images & train_images),
    }


def _catalog(config_root: Path) -> list[dict[str, Any]]:
    path = config_root / "downloads" / "catalog.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("resources", []))


def _track(record: RunRecord) -> str:
    return TRACKS.get(record.protocol_id, "?")


def _variant_key(record: RunRecord) -> tuple[str, ...]:
    return variant_key(record)


def _is_reduced(record: RunRecord) -> bool:
    if "REDUCED-BUDGET" in (record.notes or "") or "SMOKE" in (record.notes or ""):
        return True
    official = OFFICIAL_EPOCHS.get(record.task_id)
    return official is not None and 0 < record.training.epochs_completed < official


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _confusion_metric(num_classes: int):
    """Metric callable for a confusion matrix, matching the task's headline."""
    if num_classes == 2:
        return lambda cm: cm.iou(1)
    return lambda cm: multiclass_metrics(cm)["mean_iou"]


def _run_confusion(
    runs_root: Path, record: RunRecord
) -> tuple[int, list[tuple[str, list[list[int]]]]] | None:
    """Load the per-group confusion artifact a run wrote, if any."""
    for artifact in record.artifacts:
        if artifact.kind == "confusion_matrix":
            path = runs_root / record.run_id / artifact.path
            if path.is_file():
                return load_confusion(path)
    return None


def _confusion_section(
    records: list[RunRecord], runs_root: Path, matrix: dict[str, Any], add
) -> None:
    """Grouped and paired bootstrap intervals from the workers' per-group matrices."""
    by_id = {record.run_id: record for record in records}
    cells = [cell for cell in matrix["cells"] if is_primary_panel(cell)]

    pooled: dict[tuple[str, str, str], tuple[int, list[np.ndarray], list[np.ndarray]]] = {}
    representative: dict[tuple[str, str], dict[str, tuple[int, dict[str, list[list[int]]]]]] = (
        defaultdict(dict)
    )
    for cell in cells:
        key = (cell["track"], cell["model"], cell["task"])
        counts: list[np.ndarray] = []
        group_sums: dict[str, np.ndarray] = {}
        num_classes: int | None = None
        runs = sorted(
            (by_id[run_id] for run_id in cell["runs"] if run_id in by_id), key=lambda r: r.seed
        )
        for record in runs:
            loaded = _run_confusion(runs_root, record)
            if loaded is None:
                continue
            num_classes, groups = loaded
            counts.extend(np.asarray(group_counts, dtype=np.int64) for _, group_counts in groups)
            for group_id, group_counts in groups:
                matrix = np.asarray(group_counts, dtype=np.int64)
                group_sums[group_id] = (
                    group_sums[group_id] + matrix if group_id in group_sums else matrix
                )
            representative[(cell["track"], cell["task"])].setdefault(
                cell["model"], (num_classes, dict(groups))
            )
        if counts and num_classes is not None:
            pooled[key] = (num_classes, counts, list(group_sums.values()))

    if not pooled:
        add("（尚无 worker 写出逐组混淆矩阵；接入 T1/T5 后会在此自动出现区间。）")
        add("")
        return

    add(
        "同一地理组在多个种子下都会出现，把“组×种子”当成独立单元会低估不确定性。"
        "下表同时给出两种区间：左列沿用组×种子（把种子间差异也纳入），"
        "右列先把同一种子下相同 group 的混淆矩阵相加、再按地理组重采样（种子不作为独立样本）。"
        "两者都只覆盖评测集抽样，不包含训练种子本身的不确定性；后者见各单元的标准差。"
    )
    add("")
    add(
        "| 任务 | 轨道 | 模型 | 指标 | 点估计 | 95% 区间（组×种子） "
        "| 95% 区间（地理组，种子合并） | 单元数（组×种子 / 组） |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for (track, model, task), (num_classes, counts, grouped) in sorted(pooled.items()):
        interval = bootstrap_confusion(counts, _confusion_metric(num_classes))
        grouped_interval = bootstrap_confusion(grouped, _confusion_metric(num_classes))
        name = "foreground_iou" if num_classes == 2 else "mean_iou"
        add(
            f"| {task} | {track} | {MODEL_NAMES.get(model, model)} | {name} | "
            f"{interval.point:.4f} | [{interval.low:.4f}, {interval.high:.4f}] "
            f"| [{grouped_interval.low:.4f}, {grouped_interval.high:.4f}] "
            f"| {len(counts)} / {len(grouped)} |"
        )
    add("")

    add("#### 2.3.1 配对区间（同任务同轨道，按 group 对齐）")
    add("")
    add(
        "取每个模型该单元最小种子的那次运行，按相同 group 对齐做配对重采样，"
        "区间是“模型差”的区间，group 难度被抵消；group 集合不一致的模型对不列。"
    )
    add("")
    add("| 任务 | 轨道 | 模型差 | 差值点估计 | 95% 区间 | 共同组数 |")
    add("|---|---|---|---|---|---|")
    any_paired = False
    for (track, task), models in sorted(representative.items()):
        names = sorted(models)
        for left_index in range(len(names)):
            for right_index in range(left_index + 1, len(names)):
                left, right = names[left_index], names[right_index]
                left_classes, left_groups = models[left]
                right_classes, right_groups = models[right]
                if left_classes != right_classes:
                    continue
                shared = sorted(set(left_groups) & set(right_groups))
                if not shared:
                    continue
                counts_left = [np.asarray(left_groups[g], dtype=np.int64) for g in shared]
                counts_right = [np.asarray(right_groups[g], dtype=np.int64) for g in shared]
                interval = bootstrap_confusion_paired(
                    counts_left, counts_right, _confusion_metric(left_classes)
                )
                any_paired = True
                add(
                    f"| {task} | {track} | "
                    f"{MODEL_NAMES.get(left, left)} − {MODEL_NAMES.get(right, right)} | "
                    f"{interval.point:+.4f} | [{interval.low:+.4f}, {interval.high:+.4f}] | "
                    f"{len(shared)} |"
                )
    if not any_paired:
        add("| — | — | 暂无可配对的模型对 | — | — | — |")
    add("")


def _common_seed_note(matrix: dict[str, Any], add, names: dict[str, str]) -> None:
    """Flag cells whose seed counts differ inside the same (task, track).

    A six-seed mean is not directly comparable with a three-seed mean. When the
    seed sets differ, show each model's mean over the shared seeds so the
    comparison is like-for-like (review 2026-10-21).

    The numbers come from the *matrix cells*, i.e. the same variant the headline
    table shows. Reading the raw variant groups instead let the last learning
    rate win, so the table could compare a configuration that was not selected
    anywhere else (review 2026-10-22).
    """
    by_cell: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(dict)
    for cell in matrix["cells"]:
        if not is_primary_panel(cell):
            continue
        by_cell[(cell["task"], cell["track"])][cell["model"]] = {
            int(seed): value for seed, value in cell["values"].items()
        }

    rows: list[str] = []
    for (task_id, track), models in sorted(by_cell.items()):
        if len(models) < 2:
            continue
        seed_sets = [set(scores) for scores in models.values()]
        if len({frozenset(seeds) for seeds in seed_sets}) == 1:
            continue
        shared = sorted(set.intersection(*seed_sets))
        if len(shared) < 2:
            continue
        cells = []
        for model_id, scores in sorted(models.items()):
            values = [scores[seed] for seed in shared]
            cells.append(
                f"{names.get(model_id, model_id)} {statistics.mean(values):.4f}"
                f"（{len(scores)} 个种子中取 {len(shared)} 个）"
            )
        rows.append(
            f"| {task_id} | {track} | {','.join(map(str, shared))} | " + "；".join(cells) + " |"
        )

    if not rows:
        return
    add("#### 2.0.1 共同种子对照（种子数不一致的单元）")
    add("")
    add(
        "同任务同轨道内若各模型的种子数不同，直接比较均值会混入样本数差异。"
        "下表按共同种子重算均值，供对照；完整种子结果仍以上表为准。"
    )
    add("")
    add("| 任务 | 轨道 | 共同种子 | 各模型共同种子均值 |")
    add("|---|---|---|---|")
    for row in rows:
        add(row)
    add("")


def build_markdown(
    runs_root: Path,
    data_root: Path,
    config_root: Path,
    repo_root: Path,
) -> str:
    repository = RunRepository(runs_root)
    records, errors = repository.list_records()
    quarantine = runs_root.parent / "runs_quarantine"
    quarantined = (
        sorted(p.parent.name for p in quarantine.glob("*/run.json")) if quarantine.exists() else []
    )

    scored = [r for r in records if r.primary_metric() is not None and not is_superseded(r)]
    unscored = [r for r in records if r.primary_metric() is None and not is_superseded(r)]
    # Review 2026-10-21: a run that predates a value-changing fix must not appear
    # in the main table, the scorecard or the Pareto front as a finished result.
    stale = stale_run_ids(records, repo_root)
    live_scored = [r for r in scored if r.run_id not in stale]
    stale_scored = [r for r in scored if r.run_id in stale]

    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for record in live_scored:
        metric = record.primary_metric()
        key = (record.task_id, _track(record), record.model_id, record.protocol_id, metric.name)
        groups[key + _variant_key(record)].append(record)
    # A stale run only costs a cell its value when the cell would have selected
    # it. Rebuild the matrix as if stale runs were allowed and keep the ones it
    # picks: an older duplicate already replaced by a live run is harmless.
    by_id = {record.run_id: record for record in records}
    matrix_all = build_matrix(
        [r for r in records if not is_superseded(r)], model_names(config_root)
    )
    canonical_stale = [
        by_id[run_id]
        for cell in matrix_all["cells"]
        for run_id in cell["runs"]
        if run_id in stale and run_id in by_id
    ]

    lines: list[str] = []
    add = lines.append
    add("# 多光谱骨干模型评测：当前实验报告")
    add("")
    add(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"- 代码版本：`{_git_commit(repo_root)}`")
    superseded = sum(1 for r in records if is_superseded(r))
    completed_live = sum(1 for r in records if r.status == "completed" and not is_superseded(r))
    failed_live = sum(1 for r in records if r.status == "failed" and not is_superseded(r))
    running_live = sum(1 for r in records if r.status == "running" and not is_superseded(r))
    add(
        f"- 账本：{len(records)} 条可解析（{len(errors)} 条解析失败），其中 "
        f"**{completed_live} 条未取代的已完成运行**可用于比较；"
        f"{superseded} 条已取代、{failed_live} 条失败、{running_live} 条运行中；"
        f"{len(quarantined)} 条已隔离"
    )
    add(
        "  “可解析记录数”不等于“可用于正式比较的实验数”：失败的探测/中断、"
        "被取代的旧版和修复前的记录都不参与结论。"
    )
    add("")

    add("## 1. 实验设置")
    add("")
    add("### 1.1 轨道")
    add("")
    add("| 轨道 | 含义 | 主干 | 头 | 训练方式 |")
    add("|---|---|---|---|---|")
    add(
        "| B0 | Prithvi 参考复现：论文协议或官方仓库配置 | Prithvi-EO-2.0 "
        "300M-TL | UPerNet（TerraTorch） | 微调 |"
    )
    add(
        "| B1 | 全量下游适配（主结果） | Prithvi / Clay / AnySat（OlmoEarth 未接入） | "
        "各自任务头（分割为轻量解码器，分类为 MLP） | 主干和头一起训练 |"
    )
    add("| B2 | 冻结表征（辅助） | 四种模型 | 统一轻量头（LiteSpatialHead 或时序头） | 只训练头 |")
    add("| B3 | 原生输入（能力上限） | 三种模型 | - | 仅完成前向探针，未训练 |")
    add("")
    add("### 1.2 任务与数据")
    add("")
    manifests = _manifest_summary(data_root)
    catalog = {entry.get("resource_id", ""): entry for entry in _catalog(config_root)}
    add("| 任务 | 数据集 | 许可 | 来源 | 样本数（按划分） |")
    add("|---|---|---|---|---|")
    for task_id in ("T1", "T2", "T3", "T4", "T5"):
        dataset_id = TASK_DATASET_IDS[task_id]
        entry = catalog.get(dataset_id, {})
        source = entry.get("source_page", "未记录")
        license_ = entry.get("license", "未记录")
        counts = manifests.get(task_id, [{}])[0].get("counts", {})
        count_text = "，".join(f"{k} {v}" for k, v in sorted(counts.items())) or "未建 manifest"
        add(
            f"| {task_id} {TASK_NAMES[task_id]} | {entry.get('display_name', dataset_id)} | "
            f"{license_} | {source} | {count_text} |"
        )
    add("")
    add(
        "T3、T4 目前只有 OlmoEarth v1.2 Tiny/Base 的冻结头结果（各 3 个种子）；"
        "其余模型与全量微调（B1）尚未开始。T7 未建 manifest；T6（Sen4Map）等待新硬盘。"
    )
    add("")
    add("manifest 文件：")
    add("")
    for task_id in ("T1", "T2", "T3", "T4", "T5"):
        for item in manifests.get(task_id, []):
            add(f"- {task_id}：`{item['file']}`（前 16 位 sha256：`{item['sha256_prefix']}`）")
    add("")
    add("### 1.3 划分边界与可主张的泛化范围")
    add("")
    add("| 任务 | 训练组 | 评测组 | 评测组同时出现在训练 | 相同影像路径 |")
    add("|---|---:|---:|---:|---:|")
    for task_id in ("T1", "T2", "T3", "T4", "T5"):
        overlap = _split_overlap(data_root, task_id)
        if overlap is None:
            continue
        add(
            f"| {task_id} | {overlap['train_groups']} | {overlap['eval_groups']} "
            f"| {overlap['shared_groups']} | {overlap['shared_images']} |"
        )
    add("")
    add(
        "评测集没有与训练集共享同一影像路径，但**分组（场景/事件/瓦片）大量重合**，"
        "因此这些分数**不能用来主张“未见地理区域/事件”的泛化**；它们是同分布留出集的成绩。"
        "T1 的 Bolivia holdout 与训练集不共享分组，应作为独立的跨域证据单独评价；"
        "T2/T5 若要验证域外能力，需要另建地理或时间独立的协议。"
    )
    add("")
    add("### 1.4 模型与权重")
    add("")
    add("| 模型 | 权重来源 | 许可 | 备注 |")
    add("|---|---|---|---|")
    model_notes = {
        "prithvi_eo_v2_300_tl": "Prithvi_EO_V2_300M_TL.pt，ViT-L 300M，TL 版本",
        "clay_v1_5": "clay-v1.5.ckpt",
        "anysat_base": "AnySat.pth，base 版本",
        "olmoearth_v1_2_tiny": "v1.2 Tiny，冻结主干（encoder 约 12.5M）",
        "olmoearth_v1_2_base": "v1.2 Base，冻结主干（encoder 约 114M）",
    }
    for model_id, note in model_notes.items():
        entry = catalog.get(f"model_{model_id}", {})
        source = entry.get("source_page", "未记录")
        license_ = entry.get("license", "未记录")
        add(f"| {MODEL_NAMES[model_id]} | {source} | {license_} | {note} |")
    add("")
    add("### 1.5 训练与评测约定")
    add("")
    seeds = sorted({record.seed for record in records})
    add(f"- 种子：{', '.join(str(seed) for seed in seeds) or '—'}（主结果）。")
    tasks_present = sorted({record.task_id for record in records})
    add(f"- 已跑任务：{', '.join(tasks_present) or '—'}；各任务的轨道覆盖见 2.0 矩阵。")
    reduced = sorted({record.task_id for record in records if _is_reduced(record)})
    if reduced:
        add(f"- 缩减预算任务（非官方 epoch，三家相同）：{', '.join(reduced)}。")
    compat = compatibility_matrix(records, model_names(config_root))
    adapted = sorted({entry["name"] for entry in compat if entry["support"] == "adapter"})
    if adapted:
        add(f"- 使用显式输入适配器的模型：{', '.join(adapted)}（适配方式见各 run 的 tags）。")
    native = sorted({entry["name"] for entry in compat if entry["support"] == "native"})
    if native:
        add(f"- 原生输入（未训练适配器）：{', '.join(native)}。")
    native_time = sorted(
        {
            record.model_id
            for record in records
            if record.tags.get("temporal_fusion", "").startswith("native")
        }
    )
    if native_time:
        add(
            f"- 原生时序融合：{', '.join(native_time)}；其余为逐日期编码 + 统一时序头，"
            "结构不同需分别说明。"
        )
    crops = {
        record.model_id: record.tags["train_crop_px"]
        for record in records
        if record.tags.get("train_crop_px")
    }
    if crops:
        add("- 训练裁剪：" + "，".join(f"{m} {c}px" for m, c in sorted(crops.items())) + "。")
    add("- 选模与评测划分逐 run 记录在 `select_split`/`eval_split`，不在此汇总。")
    add("")

    add("## 2. 结果")
    add("")
    task_metric = {
        record.task_id: record.primary_metric().name
        for record in records
        if record.primary_metric() is not None
    }
    if task_metric:
        add(
            "指标说明："
            + "，".join(f"{tid} 为 {m}" for tid, m in sorted(task_metric.items()))
            + "。"
        )
    add(
        "均值 ± 标准差基于不同种子的运行（同一 seed 的重跑只取最新一次）；"
        "只有 1 个种子时不报告标准差。"
    )
    add("")
    if canonical_stale:
        affected = sorted(
            {
                f"{r.task_id} {_track(r)} {MODEL_NAMES.get(r.model_id, r.model_id)}"
                f"（seed {r.seed}）"
                for r in canonical_stale
            }
        )
        add(
            "> ⚠️ **本轮不能作为最终排名。** 下列单元的当前最新记录早于改变结果的修复，"
            "已从主表、评分卡与 Pareto 中剔除（明细见 3.4）：" + "；".join(affected) + "。"
        )
        add("")
    add("")
    add(
        "| 任务 | 轨道 | 模型 | 协议 | 指标 | 头 / 预处理 "
        "| 种子数 | 均值 ± 标准差 | 各种子结果 | 备注 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5:])):
        runs = sorted(canonical_runs(groups[key]), key=lambda r: r.seed)
        task_id, track, model_id, protocol_id, metric_name = key[:5]
        v = decode_variant_tail(key, protocol_id)
        head_id = v["head"]
        preprocessing_id = v["preprocessing"]
        lr = v["learning_rate"]
        input_px = v["input_px"]
        init = v["init"]
        label_fraction = v["label_fraction"]
        _manifest = v["manifest"]
        _checkpoint = v["checkpoint"]
        _crop = v["train_crop_px"]
        values = [r.primary_metric().value for r in runs]
        seeds = sorted({r.seed for r in runs})
        if len(values) > 1:
            spread = f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"
        else:
            spread = f"{values[0]:.4f}（单次）"
        detail = "，".join(f"s{r.seed}={r.primary_metric().value:.4f}" for r in runs)
        notes = []
        if any(_is_reduced(r) for r in runs):
            notes.append("非官方预算")
        if lr != "-":
            notes.append(f"lr={lr}")
        if input_px != "-":
            notes.append(f"编码输入 {input_px}px")
        if init != "pretrained":
            notes.append(f"init={init}")
        if label_fraction != "1.0":
            notes.append(f"标注={label_fraction}")
        if any(is_superseded(r) for r in runs):
            notes.append("含已取代的旧版")
        if any(r.run_id in stale for r in runs):
            notes.append("⚠️ 含修复前记录（待重跑）")
        add(
            f"| {task_id} | {track} {TRACK_NAMES.get(track, '')} | "
            f"{MODEL_NAMES.get(model_id, model_id)} | "
            f"`{protocol_id}` | {metric_name} | {head_id} / {preprocessing_id} | {len(seeds)} | "
            f"{spread} | {detail} | {'；'.join(notes) or '-'} |"
        )
    add("")
    live_records = [r for r in records if r.run_id not in stale]
    matrix = build_matrix(live_records, model_names(config_root))
    _common_seed_note(matrix, add, model_names(config_root))
    card = scorecard(matrix, live_records, model_names(config_root))
    add("### 2.1 Backbone 评分卡（四个维度）")
    add("")
    add("分数是相对值：只在本报告内比较，加入新模型后会变化。覆盖任务数不全时，该维度只能参考。")
    add("")
    add(
        "| 模型 | D1 表征/迁移 B1（覆盖） | D1 表征/迁移 B2（覆盖） | D2 泛化/鲁棒性 "
        "| D3 使用效率（B1 耗时/显存/参数量） | D4 稳定性（sd/运行事故率） | Pareto |"
    )
    add("|---|---|---|---|---|---|---|")
    for row in card["models"]:
        d1b1 = (
            "—"
            if row["d1_b1"]["score"] is None
            else f"{row['d1_b1']['score']:.3f}（{row['d1_b1']['coverage']}）"
        )
        d1b2 = (
            "—"
            if row["d1_b2"]["score"] is None
            else f"{row['d1_b2']['score']:.3f}（{row['d1_b2']['coverage']}）"
        )
        d2 = row["d2"]
        if d2["results"]:

            def _d2_label(item: dict) -> str:
                bits = [str(item.get("track", "?")), str(item.get("task", "?"))]
                if item.get("init", "pretrained") != "pretrained":
                    bits.append(f"init={item['init']}")
                if item.get("label_fraction", "1.0") != "1.0":
                    bits.append(f"标注={item['label_fraction']}")
                return "/".join(bits)

            d2_text = "；".join(
                f"[{_d2_label(item)}] "
                + (
                    f"{item['perturbation']}：{item['perturbed']:.3f}（降 {item['drop']:.3f}，"
                    f"保持 {item['retention']:.2f}）"
                    if item["retention"] is not None
                    else f"{item['perturbation']}：{item['perturbed']:.3f}（降 {item['drop']:.3f}）"
                )
                for item in d2["results"]
            )
        else:
            d2_text = d2["note"] or "—"
        d3 = row["d3"]
        d3_text = (
            f"{d3['b1_wall_min_mean'] or '—'} 分钟，{d3['b1_peak_mem_mb'] or '—'} MB，"
            f"B1 可训练 {d3['b1_trainable_params'] or '—'} / 总 {d3['b1_total_params'] or '—'}"
            f"（B2 可训练 {d3['b2_trainable_params'] or '—'}）"
        )
        d4 = row["d4"]
        failure = f"{d4['failure_rate']:.1%}" if d4["failure_rate"] is not None else "—"
        incidents = d4.get("incident_classes") or {}
        incident_text = "，".join(
            f"{name} {count}" for name, count in sorted(incidents.items(), key=lambda kv: -kv[1])
        )
        d4_text = (
            f"sd {d4['mean_sd_b1'] if d4['mean_sd_b1'] is not None else '—'}，"
            f"运行事故率 {failure}（{d4['failed_runs']}/{d4['training_runs']}）"
            + (f"，其中 {incident_text}" if incident_text else "")
            + f"，排除 running {d4['excluded_running']} / probe {d4['excluded_probes']}"
        )
        pareto = "—" if row.get("pareto") is None else ("是" if row["pareto"] else "否")
        add(f"| {row['name']} | {d1b1} | {d1b2} | {d2_text} | {d3_text} | {d4_text} | {pareto} |")
    add("")
    add("各维度的计算方式：")
    add("")
    for dim in card["dimensions"]:
        add(f"- **{dim['id']} {dim['name']}**：{dim['measure']}")
    add("")
    add(
        "D3/D4 只列原始值，不做“最差归零”的换算；`Pareto` 表示在（D1 B1 精度，B1 训练耗时）"
        "上不被任何模型同时更准且更省。综合总分暂不提供，避免参照系随参赛模型变化。"
    )
    add(
        "D3 的耗时/显存/参数量与 D4 的标准差都取 **B1** 单元；尚无 B1 的模型"
        "（当前为 OlmoEarth）在这些列留空，不用冻结头（B2）的数值代替——"
        "两者可训练参数相差两个数量级，混用会把“只训头”读成“全量微调很便宜”。"
    )
    add("")
    add(
        "D4 的“运行事故率”是**运行层面**的统计，包含配置拦截（协议不支持）、显存探针、"
        "缓存读取故障和人工中断，不等于模型本身的训练稳定性；按类别拆分见括号。"
    )
    add("")
    compat = compatibility_matrix(records, model_names(config_root))
    if compat:
        add("### 2.1.1 输入兼容性矩阵")
        add("")
        add(
            "| 模型 | 传感器 | 波段数 | 时相 | 接收方式 | 验证状态 | 适配器参数 | 已跑轨道 | 任务 |"
        )
        add("|---|---|---|---|---|---|---|---|---|")
        for entry in compat:
            add(
                f"| {entry['name']} | {entry['sensor']} | {len(entry['bands'])} | "
                f"{entry['time_steps']} | {entry['received']} | {entry['validated']} | "
                f"{entry['adapter_parameters'] or '—'} | {', '.join(entry['tracks']) or '—'} | "
                f"{', '.join(entry['tasks']) or '—'} |"
            )
        add("")
    fractions = sorted(
        {record.tags["label_fraction"] for record in records if record.tags.get("label_fraction")}
    )
    if fractions:
        add("### 2.1.2 少标注曲线（标注量—精度）")
        add("")
        add("固定验证/测试集，按标注预算（`label_fraction`）重训；单位是切片或地块。")
        add("")
        add("| 任务 | 轨道 | 模型 | " + " | ".join(f"标注={f}" for f in fractions) + " |")
        add("|---|---|---|" + "---|" * len(fractions))
        budget_groups: dict[tuple, list[RunRecord]] = defaultdict(list)
        for record in records:
            if record.status != "completed" or record.primary_metric() is None:
                continue
            if not record.tags.get("label_fraction"):
                continue
            budget_groups[
                (
                    record.task_id,
                    _track(record),
                    record.model_id,
                    record.tags["label_fraction"],
                )
                + _variant_key(record)
            ].append(record)
        budget: dict[tuple, dict] = {}
        for key, runs in budget_groups.items():
            canonical = canonical_runs(runs)
            values = [record.primary_metric().value for record in canonical]
            candidate = {
                "mean": statistics.mean(values),
                "n": len({record.seed for record in canonical}),
                "validation": mean_validation(canonical),
            }
            panel = key[:4]
            current = budget.get(panel)
            if current is None or (candidate["validation"], candidate["n"]) > (
                current["validation"],
                current["n"],
            ):
                budget[panel] = candidate
        budget_rows = sorted({key[:3] for key in budget})
        for task, track, model in budget_rows:
            cells = []
            for fraction in fractions:
                entry = budget.get((task, track, model, fraction))
                cells.append(f"{entry['mean']:.4f}（n={entry['n']}）" if entry else "—")
            add(f"| {task} | {track} | {model} | " + " | ".join(cells) + " |")
        add("")
    add("### 2.2 单次运行明细")
    add("")
    add(
        "| run_id | 任务 | 轨道 | 模型 | 种子 | 状态 | 指标值 "
        "| 训练时长（分钟） | 峰值显存（MB） | 代码版本 | 权重哈希 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    detail = [r for r in records if not is_superseded(r) and r.run_id not in stale]
    for record in sorted(detail, key=lambda r: (r.task_id, _track(r), r.model_id, r.seed)):
        metric = record.primary_metric()
        wall = record.training.wall_time_seconds
        wall_text = "-" if wall is None else f"{wall / 60:.1f}"
        mem = record.training.peak_gpu_memory_mb
        mem_text = "-" if mem is None else f"{mem:.0f}"
        add(
            f"| `{record.run_id}` | {record.task_id} | {_track(record)} | "
            f"{MODEL_NAMES.get(record.model_id, record.model_id)} | "
            f"{record.seed} | {record.status} | {_fmt(metric.value) if metric else '无指标'} | "
            f"{wall_text} | {mem_text} | "
            f"`{record.git_commit or '未记录'}` | "
            f"`{(record.model_checkpoint_sha256 or '未记录')[:16]}` |"
        )
    add("")

    add("### 2.2.1 分阶段成本与缓存（有记录时）")
    add("")
    phase_rows = [r for r in records if r.training.phases or r.training.cache_bytes]
    if phase_rows:
        add(
            "| run_id | 任务 | 轨道 | 模型 | 种子 | optimizer steps "
            "| 分阶段耗时（秒） | 缓存（GB） |"
        )
        add("|---|---|---|---|---|---|---|---|")
        for record in sorted(phase_rows, key=lambda r: (r.task_id, _track(r), r.model_id, r.seed)):
            phases = record.training.phases
            phase_text = (
                "，".join(f"{name}={value:.1f}" for name, value in sorted(phases.items())) or "—"
            )
            cache = record.training.cache_bytes
            cache_text = f"{cache / 1e9:.2f}" if cache else "—"
            add(
                f"| `{record.run_id}` | {record.task_id} | {_track(record)} | "
                f"{MODEL_NAMES.get(record.model_id, record.model_id)} | {record.seed} | "
                f"{record.training.optimizer_steps} | {phase_text} | {cache_text} |"
            )
    else:
        add("（尚无运行记录分阶段成本或缓存大小。）")
    add("")

    add("### 2.3 分组置信区间")
    add("")
    _confusion_section(records, runs_root, matrix, add)

    add("### 2.4 解释边界")
    add("")
    add(
        "- **轨道不同不可直接归因。** T5 的 AnySat 是 B2S（先学适配器再冻结），"
        "Clay/Prithvi 是 B2（冻结骨干只训头）；任务头结构、时序融合方式和更新次数都不同"
        "（T5 两阶段 125 轮约 17.4 万次更新，冻结头路径约 4.3 万次），"
        "分差是**方案差**，不能写成“同协议骨干领先”。"
    )
    add(
        "- **成本字段口径不全。** 部分旧运行的 `phases` 为空、`cache_bytes` 未记录；"
        "冻结头路径的计时从特征就绪开始，不含首次提取。跨模型的端到端成本倍数需要"
        "按原始日志重建，不能用 `wall_time_seconds` 直接相除。"
    )
    add(
        "- **置信区间只覆盖评测集抽样。** 两种区间都不包含训练种子本身的不确定性；"
        "跨模型比较应同时看种子标准差与共同种子对照（2.0.1）。"
    )
    add("")

    add("## 3. 异常、隔离与待办")
    add("")
    add("### 3.1 已隔离的运行（不计入结果）")
    add("")
    if quarantined:
        for name in quarantined:
            add(
                f"- `{name}`：早于全忽略 batch 修复，loss 出现 "
                f"NaN，代码版本缺失，已移入 `runs_quarantine/`。"
            )
    else:
        add("- 无")
    add("")
    add("### 3.1b 已取代的运行（保留记录，不计入最终结论）")
    add("")
    superseded = [r for r in records if is_superseded(r)]
    for record in sorted(superseded, key=lambda r: r.run_id):
        add(f"- `{record.run_id}`：{record.tags['superseded']}")
    if not superseded:
        add("- 无")
    add("")
    add("### 3.2 无法解析的 run 记录")
    add("")
    if errors:
        for item in errors:
            add(f"- `{item['path']}`：{item['error'].splitlines()[0]}")
    else:
        add("- 无")
    add("")
    add("### 3.3 无指标的运行")
    add("")
    if unscored:
        for record in unscored:
            add(f"- `{record.run_id}`（{record.status}）：{(record.notes or '无说明')[:120]}")
    else:
        add("- 无")
    add("")
    add("### 3.4 修复前的记录（已从主表剔除，待重跑）")
    add("")
    if stale_scored:
        add(
            f"这些运行早于某个改变指标语义的修复（按代码祖先判定），数值只代表旧语义。"
            f"其中 **{len(canonical_stale)} 条仍是该单元当前最新的记录**（这些单元当前缺值），"
            f"其余 {len(stale_scored) - len(canonical_stale)} 条已被更新的有效记录取代。"
            "T1FIX 队列会重跑其中缺失的 T1 B1 单元。"
        )
        add("")
        add("| run_id | 任务 | 轨道 | 模型 | 种子 | 主指标 | 代码版本 |")
        add("|---|---|---|---|---:|---|---|")
        for record in sorted(stale_scored, key=lambda r: r.run_id):
            metric = record.primary_metric()
            add(
                f"| `{record.run_id}` | {record.task_id} | {_track(record)} | "
                f"{MODEL_NAMES.get(record.model_id, record.model_id)} | {record.seed} | "
                f"{metric.name}={metric.value:.4f} | `{(record.git_commit or '-')[:8]}` |"
            )
    else:
        add("- 无")
    add("")
    add("### 3.5 记录缺失")
    add("")
    no_hash = [r.run_id for r in records if not r.model_checkpoint_sha256]
    no_commit = [r.run_id for r in records if not r.git_commit]
    add(f"- 未记录权重哈希的运行：{len(no_hash)} 条（需在 worker 中补充写入）")
    add(f"- 未记录代码版本的运行：{len(no_commit)} 条（需修复 dispatch 的版本记录）")
    add("")
    add("### 3.6 尚未完成")
    add("")
    add("- T2、T5 的全量微调（B1）未运行。")
    b2s = [r for r in live_records if r.protocol_id == "frozen_adapter_two_stage"]
    b2s_done = sorted({r.seed for r in b2s if r.status == "completed"})
    b2s_running = [r for r in b2s if r.status == "running"]
    if b2s:
        note = f"- T5 两阶段（B2S）已完成 {len(b2s_done)} 个种子（{b2s_done}）"
        if b2s_running:
            note += "；进行中：" + "，".join(
                f"`{r.run_id}`（seed {r.seed}，epoch {r.training.epochs_completed}）"
                for r in b2s_running
            )
        add(note + "。")
    other_running = [r for r in live_records if r.status == "running" and r not in b2s_running]
    for record in other_running:
        add(
            f"- 进行中的运行：`{record.run_id}`（{record.task_id} "
            f"{MODEL_NAMES.get(record.model_id, record.model_id)}，"
            f"epoch {record.training.epochs_completed}）。"
        )
    add("- B3 只有前向探针，无训练结果。")
    add(
        "- T3、T4 目前只有 OlmoEarth Tiny/Base 的冻结头结果（各 3 种子）；"
        "其他模型和 B1 未开始，且 T3 的分数接近平凡背景解，暂不能作为能力结论。"
        "T7 未建 manifest；T6 等待硬盘。"
    )
    add(
        "- T1 B1 的 9 个 cell 值中 4 个来自修复前（stale）的 run："
        "AnySat seed 42、Prithvi 全部三个种子；Clay 在调优 lr 下从未真正跑过微调。"
    )
    add("- 论文对照：T1 的 B0 复现低于论文约 6 个点，原因未确认（论文用了超参搜索和 10 个种子）。")
    add("")
    return "\n".join(lines)
