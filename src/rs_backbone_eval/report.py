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

import yaml

from .contracts import RunRecord
from .matrix import build_matrix, canonical_runs, compatibility_matrix, model_names, scorecard
from .results import RunRepository

TRACKS = {
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
TASK_NAMES = {
    "T1": "洪水水体分割（Sen1Floods11）",
    "T2": "烧毁区分割（HLS Burn Scars）",
    "T5": "多时相作物分类（US Multi-temporal Crop）",
}
TASK_DATASET_IDS = {
    "T1": "data_sen1floods11",
    "T2": "data_hls_burn_scars",
    "T5": "data_us_multicrop",
}
OFFICIAL_EPOCHS = {"T1": 50, "T2": 50, "T5": 125}
SUPERSEDED = {
    "prithvi-t5-frozen-42": "lr 1e-4 的旧版，已由 prithvi-t5-frozen-42-v2（lr 1e-3）取代",
}
MODEL_NAMES = {
    "prithvi_eo_v2_300_tl": "Prithvi-EO-2.0 300M-TL",
    "clay_v1_5": "Clay v1.5",
    "anysat_base": "AnySat (base)",
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


def _catalog(config_root: Path) -> list[dict[str, Any]]:
    path = config_root / "downloads" / "catalog.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("resources", []))


def _track(record: RunRecord) -> str:
    return TRACKS.get(record.protocol_id, "?")


def _variant_key(record: RunRecord) -> tuple[str, str, str, str]:
    return (
        record.training.head_id,
        record.input.preprocessing_id,
        record.tags.get("learning_rate", "-"),
        record.tags.get("encoder_input_px", "-"),
    )


def _is_reduced(record: RunRecord) -> bool:
    if "REDUCED-BUDGET" in (record.notes or "") or "SMOKE" in (record.notes or ""):
        return True
    official = OFFICIAL_EPOCHS.get(record.task_id)
    return official is not None and 0 < record.training.epochs_completed < official


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


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

    scored = [r for r in records if r.primary_metric() is not None]
    unscored = [r for r in records if r.primary_metric() is None]

    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for record in scored:
        metric = record.primary_metric()
        key = (record.task_id, _track(record), record.model_id, record.protocol_id, metric.name)
        groups[key + _variant_key(record)].append(record)

    lines: list[str] = []
    add = lines.append
    add("# 多光谱骨干模型评测：当前实验报告")
    add("")
    add(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"- 代码版本：`{_git_commit(repo_root)}`")
    add(f"- run 记录：{len(records)} 条有效，{len(errors)} 条无效，{len(quarantined)} 条已隔离")
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
        "| B1 | 全量下游适配（主结果） | 三种模型 | "
        "各自任务头（分割为轻量解码器，分类为 MLP） | 主干和头一起训练 |"
    )
    add("| B2 | 冻结表征（辅助） | 三种模型 | 统一轻量头（LiteSpatialHead 或时序头） | 只训练头 |")
    add("| B3 | 原生输入（能力上限） | 三种模型 | - | 仅完成前向探针，未训练 |")
    add("")
    add("### 1.2 任务与数据")
    add("")
    manifests = _manifest_summary(data_root)
    catalog = {entry.get("resource_id", ""): entry for entry in _catalog(config_root)}
    add("| 任务 | 数据集 | 许可 | 来源 | 样本数（按划分） |")
    add("|---|---|---|---|---|")
    for task_id in ("T1", "T2", "T5"):
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
    add("未纳入：T3、T4、T7 尚未建 manifest；T6（Sen4Map）等待新硬盘。")
    add("")
    add("manifest 文件：")
    add("")
    for task_id in ("T1", "T2", "T5"):
        for item in manifests.get(task_id, []):
            add(f"- {task_id}：`{item['file']}`（前 16 位 sha256：`{item['sha256_prefix']}`）")
    add("")
    add("### 1.3 模型与权重")
    add("")
    add("| 模型 | 权重来源 | 许可 | 备注 |")
    add("|---|---|---|---|")
    model_notes = {
        "prithvi_eo_v2_300_tl": "Prithvi_EO_V2_300M_TL.pt，ViT-L 300M，TL 版本",
        "clay_v1_5": "clay-v1.5.ckpt",
        "anysat_base": "AnySat.pth，base 版本",
    }
    for model_id, note in model_notes.items():
        entry = catalog.get(f"model_{model_id}", {})
        source = entry.get("source_page", "未记录")
        license_ = entry.get("license", "未记录")
        add(f"| {MODEL_NAMES[model_id]} | {source} | {license_} | {note} |")
    add("")
    add("### 1.4 训练与评测约定")
    add("")
    add("- 三个种子：17、42、73（主结果）。")
    add(
        "- T2、T5 的评测划分为 "
        "`official_validation`（锁定，不参与调参）；项目验证集为官方训练集的 10%。"
    )
    add("- T5 为**非官方预算**：官方 125 epoch，本报告统一为 20 epoch，三家相同。")
    add(
        "- AnySat 的 B1 使用 224 训练裁剪（256 在 "
        "24GB 显卡上 OOM），与其他模型不完全一致，需在对比时说明。"
    )
    add("- AnySat 的六波段输入通过显式光谱适配器映射到十波段入口，不做零填充。")
    add(
        "- T5 中 AnySat 使用原生时序融合 + 空间头；Clay "
        "与 Prithvi 为逐日期编码 + 统一时序头，两者结构不同。"
    )
    add("")

    add("## 2. 结果")
    add("")
    add(
        "指标说明：T1 为水体 IoU，T2 为烧毁区 IoU，T5 "
        "为 13 类 mIoU。均值 ± 标准差基于不同种子的运行；"
    )
    add("只有 1 个种子时不报告标准差。")
    add("")
    add(
        "| 任务 | 轨道 | 模型 | 协议 | 指标 | 头 / 预处理 "
        "| 种子数 | 均值 ± 标准差 | 各种子结果 | 备注 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5:])):
        runs = sorted(canonical_runs(groups[key]), key=lambda r: r.seed)
        task_id, track, model_id, protocol_id, metric_name = key[:5]
        head_id, preprocessing_id, lr, input_px = key[5:]
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
        if any(r.run_id in SUPERSEDED for r in runs):
            notes.append("含已取代的旧版")
        add(
            f"| {task_id} | {track} {TRACK_NAMES.get(track, '')} | "
            f"{MODEL_NAMES.get(model_id, model_id)} | "
            f"`{protocol_id}` | {metric_name} | {head_id} / {preprocessing_id} | {len(seeds)} | "
            f"{spread} | {detail} | {'；'.join(notes) or '-'} |"
        )
    add("")
    matrix = build_matrix(records, model_names(config_root))
    card = scorecard(matrix, records, model_names(config_root))
    add("### 2.1 Backbone 评分卡（四个维度）")
    add("")
    add("分数是相对值：只在本报告内比较，加入新模型后会变化。覆盖任务数不全时，该维度只能参考。")
    add("")
    add(
        "| 模型 | D1 表征/迁移 B1（覆盖） | D1 表征/迁移 B2（覆盖） | D2 泛化/鲁棒性 "
        "| D3 使用效率（B1 耗时/显存/参数量） | D4 稳定性（sd/失败率） | Pareto |"
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
            d2_text = "；".join(
                (
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
            f"B2 可训练 {d3['b2_trainable_params'] or '—'}，总 {d3['b2_total_params'] or '—'}"
        )
        d4 = row["d4"]
        d4_text = (
            f"sd {d4['mean_sd_b1'] if d4['mean_sd_b1'] is not None else '—'}，"
            f"失败 {d4['failed_runs']}/{d4['total_runs']}"
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
    add("")
    compat = compatibility_matrix(records, model_names(config_root))
    if compat:
        add("### 2.1.1 输入兼容性矩阵")
        add("")
        add("| 模型 | 传感器 | 波段数 | 时相 | 支持方式 | 适配器参数 | 已跑轨道 | 任务 |")
        add("|---|---|---|---|---|---|---|---|")
        for entry in compat:
            add(
                f"| {entry['name']} | {entry['sensor']} | {len(entry['bands'])} | "
                f"{entry['time_steps']} | {entry['support']} | "
                f"{entry['adapter_parameters'] or '—'} | {', '.join(entry['tracks']) or '—'} | "
                f"{', '.join(entry['tasks']) or '—'} |"
            )
        add("")
    add("### 2.2 单次运行明细")
    add("")
    add(
        "| run_id | 任务 | 轨道 | 模型 | 种子 | 状态 | 指标值 "
        "| 训练时长（分钟） | 峰值显存（MB） | 代码版本 | 权重哈希 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for record in sorted(records, key=lambda r: (r.task_id, _track(r), r.model_id, r.seed)):
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
    for record in records:
        if record.run_id in SUPERSEDED:
            add(f"- `{record.run_id}`：{SUPERSEDED[record.run_id]}")
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
    add("### 3.4 记录缺失")
    add("")
    no_hash = [r.run_id for r in records if not r.model_checkpoint_sha256]
    no_commit = [r.run_id for r in records if not r.git_commit]
    add(f"- 未记录权重哈希的运行：{len(no_hash)} 条（需在 worker 中补充写入）")
    add(f"- 未记录代码版本的运行：{len(no_commit)} 条（需修复 dispatch 的版本记录）")
    add("")
    add("### 3.5 尚未完成")
    add("")
    add("- T2、T5 的全量微调（B1）未运行。")
    add("- T5 仅 seed 42，尚无误差范围；完整种子和 125 epoch 预算未做。")
    add("- B3 只有前向探针，无训练结果。")
    add("- T3、T4、T7 的 manifest 与 worker 未完成；T6 等待硬盘。")
    add("- 论文对照：T1 的 B0 复现低于论文约 6 个点，原因未确认（论文用了超参搜索和 10 个种子）。")
    add("")
    return "\n".join(lines)
