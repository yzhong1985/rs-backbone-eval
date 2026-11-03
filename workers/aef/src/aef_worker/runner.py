"""AEF worker entry point: frozen-embedding training for the T11 registry runs.

The worker consumes contract manifests in which ``image_paths[0]`` is a
precomputed AEF embedding grid (the task declares ``input_kind:
aef_features``); the per-sample value encoding comes from
``aux_paths.encoding`` and must match the declared one
(``raw_aef_int8_as_float16``). Deconding applies the signed-square transform
``sign(x) * (x/127.5)**2`` and zeroes invalid pixels; void (class 19 in the
PASTIS nomenclature) is remapped to the task's ignore_index.

Actions: ``probe`` (contract audit + decode + one forward on cpu) and
``train`` (declared head over the pre-declared 8-epoch budget on the declared
folds), writing run.json through the shared writer."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aef_worker.heads import FALLBACK_IGNORE, HEADS, IoUEvaluator
else:
    from .heads import FALLBACK_IGNORE, HEADS, IoUEvaluator

from rs_backbone_eval.contracts import (
    ManifestSample,
    MetricRecord,
    ProtocolSpec,
    TaskSpec,
    WorkerRequest,
)
from rs_backbone_eval.data.manifest import audit_task_manifest, manifest_sha256
from rs_backbone_eval.registry import load_yaml_spec
from rs_backbone_eval.worker_runtime import RunWriter, emit_event

DECLARED_ENCODING = "raw_aef_int8_as_float16"
VOID_CLASS = 19
FORMAL_EPOCHS = 8
STATS_PATCHES = 80
BATCH = 8
TRAIN_MIN = 1400  # Fold 1-3 full fold is 1,455 rows; blocks silently shrunk folds


class WorkerFailure(Exception):
    """Gates convert to a failed run with a readable reason."""


def _rows(manifest_path: Path) -> list[ManifestSample]:
    import json

    return [
        ManifestSample.model_validate(json.loads(line))
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decode(
    row: ManifestSample, request: WorkerRequest, ignore: int
) -> tuple[np.ndarray, np.ndarray]:
    encoding = (row.aux_paths or {}).get("encoding", DECLARED_ENCODING)
    if encoding != DECLARED_ENCODING:
        raise WorkerFailure(
            f"sample {row.sample_id}: aux encoding {encoding!r} is not {DECLARED_ENCODING!r}"
        )
    with np.load(request.data_root / row.image_paths[0]) as archive:
        features = archive["features"].astype(np.float32)
        valid = np.asarray(archive["valid"], dtype=bool)
    if features.ndim != 3 or features.shape[0] != 64 or valid.shape != features.shape[1:]:
        raise WorkerFailure(f"sample {row.sample_id}: malformed grid {features.shape}")
    features = np.sign(features) * (features / 127.5) ** 2
    features[:, ~valid] = 0.0
    labels = np.load(request.data_root / row.label_path)[0].astype(np.int64)
    if labels.shape != valid.shape:
        raise WorkerFailure(f"sample {row.sample_id}: label grid {labels.shape} != {valid.shape}")
    labels = np.where(labels == VOID_CLASS, ignore, labels)
    return features, labels


def _stats(
    rows: list[ManifestSample], request: WorkerRequest, ignore: int, count: int = STATS_PATCHES
) -> tuple[np.ndarray, np.ndarray]:
    """Tile-balanced feature statistics over the train fold (review 2026-11-03, #2)."""
    split = sorted(
        (row for row in rows if row.split == "train"), key=lambda row: int(row.sample_id)
    )
    by_tile: dict[str, list[ManifestSample]] = {}
    for row in split:
        by_tile.setdefault(row.aux_paths.get("tile", ""), []).append(row)
    chosen: list[ManifestSample] = []
    index = 0
    while len(chosen) < min(count, len(split)):
        advanced = False
        for values in sorted(by_tile.values()):
            if index < len(values):
                chosen.append(values[index])
                advanced = True
        index += 1
        if not advanced:
            break
    total = np.zeros(64, dtype=np.float64)
    squares = np.zeros(64, dtype=np.float64)
    seen = 0
    for row in chosen[:count]:
        features, _ = _decode(row, request, ignore)
        total += features.sum(axis=(1, 2))
        squares += (features**2).sum(axis=(1, 2))
        seen += features.shape[1] * features.shape[2]
    mean = total / max(seen, 1)
    std = np.sqrt(np.maximum(squares / max(seen, 1) - mean**2, 0.0)) + 1e-3
    return mean.astype(np.float32), std.astype(np.float32)


def _batched(
    rows: list[ManifestSample],
    request: WorkerRequest,
    ignore: int,
    mean: np.ndarray,
    std: np.ndarray,
):
    mean_t = torch.from_numpy(mean)[:, None, None]
    std_t = torch.from_numpy(std)[:, None, None]
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(rows), BATCH):
        features, labels = [], []
        for row in rows[start : start + BATCH]:
            values, target = _decode(row, request, ignore)
            features.append(torch.from_numpy(values))
            labels.append(torch.from_numpy(target))
        stacked = (torch.stack(features) - mean_t) / std_t
        batches.append((stacked, torch.stack(labels)))
    return batches


def train(request: WorkerRequest, writer: RunWriter, emit) -> int:
    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    protocol_spec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    head_name = {"aef_probe_mlp": "mlp", "aef_probe_upernet": "upernet"}[protocol_spec.protocol_id]
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE

    manifest_path = Path(request.manifest_path)
    rows = _rows(manifest_path)
    audit = audit_task_manifest(manifest_path, task_spec)
    if not audit.valid:
        raise WorkerFailure(f"manifest audit failed: {audit.all_errors()[:5]}")
    emit("manifest.ok", f"{audit.sample_count} rows, splits {dict(audit.split_counts)}")

    torch.manual_seed(request.seed)
    np.random.seed(request.seed)

    mean, std = _stats(rows, request, ignore)
    groups = {
        split: sorted(
            (row for row in rows if row.split == split), key=lambda row: int(row.sample_id)
        )
        for split in ("train", "validation", "test")
    }
    if len(groups["train"]) < TRAIN_MIN:
        raise WorkerFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs >= {TRAIN_MIN}"
        )
    batches = {split: _batched(groups[split], request, ignore, mean, std) for split in groups}

    started = time.time()
    model: nn.Module = HEADS[head_name]()
    writer.update_training(
        head_id=head_name,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
    writer.update_input(preprocessing_id="aef_signed_square_imagenetfree")
    lr = 3e-3 if head_name == "mlp" else 1e-3
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(FORMAL_EPOCHS):
        model.train()
        order = torch.randperm(
            len(batches["train"]), generator=torch.Generator().manual_seed(1_000 + epoch)
        ).tolist()
        total = 0.0
        for index in order:
            stacked, labels = batches["train"][index]
            loss = F.cross_entropy(model(stacked), labels, ignore_index=ignore)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        emit(
            "train.epoch",
            f"epoch {epoch}: loss {total / len(order):.4f}",
            progress=(epoch + 1) / FORMAL_EPOCHS,
        )
    writer.add_curve("loss", "train", FORMAL_EPOCHS, total / len(order))

    for split, split_batches in batches.items():
        evaluator = IoUEvaluator()
        model.eval()
        with torch.no_grad():
            for stacked, labels in split_batches:
                evaluator.add(labels, model(stacked).argmax(dim=1))
        value = evaluator.value()
        writer.add_metrics(
            [
                MetricRecord(
                    name="mean_iou",
                    direction=task_spec.primary_metric.direction,
                    value=round(value["miou"], 6),
                    split=split,
                    is_primary=split == "test",
                ),
                MetricRecord(
                    name="mean_iou_foreground",
                    direction=task_spec.primary_metric.direction,
                    value=round(value["foreground"], 6),
                    split=split,
                    is_primary=False,
                ),
            ]
        )
    test = next(metric for metric in writer.record.metrics if metric.is_primary)
    writer.update_training(
        epochs_completed=FORMAL_EPOCHS,
        optimizer_steps=len(batches["train"]) * FORMAL_EPOCHS,
        wall_time_seconds=round(time.time() - started, 1),
    )
    foreground = next(
        metric.value
        for metric in writer.record.metrics
        if metric.name == "mean_iou_foreground" and metric.split == "test"
    )
    writer.complete(
        notes=(
            f"frozen AEF 2019 embedding; head {head_name}, full fold split, "
            "value encoding raw_aef_int8_as_float16 (sign(x)*(x/127.5)^2); "
            "labels 2019 pending LPIS-year confirmation"
        )
    )
    emit(
        "train.done",
        f"head {head_name}: test mIoU {test.value:.4f}, foreground {foreground:.4f}",
        progress=1.0,
    )
    return 0


def probe(request: WorkerRequest, writer: RunWriter, emit) -> int:
    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    manifest_path = Path(request.manifest_path)
    rows = _rows(manifest_path)
    audit = audit_task_manifest(manifest_path, task_spec)
    if not audit.valid:
        raise WorkerFailure(f"manifest audit failed: {audit.all_errors()[:5]}")
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE
    started = time.time()
    model = HEADS["upernet"]()
    features, _ = _decode(rows[0], request, ignore)
    with torch.no_grad():
        _ = model(torch.from_numpy(features[None]))
    writer.complete(
        notes=(
            f"probe ok: decoded sample {rows[0].sample_id} via {rows[0].image_paths[0]}; "
            f"forward within {time.time() - started:.2f}s"
        )
    )
    emit("probe.ok", f"manifest {audit.sample_count} rows; decode+forward ok")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: runner.py <request.json>", file=sys.stderr)
        return 2
    request = WorkerRequest.model_validate_json(Path(argv[0]).read_text(encoding="utf-8"))
    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    protocol_spec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    writer = RunWriter(
        Path(request.runs_root) / request.run_id,
        run_id=request.run_id,
        task_spec=task_spec,
        protocol_spec=protocol_spec,
        model_id="aef_annual_2019",
        seed=request.seed,
        manifest_sha256=manifest_sha256(Path(request.manifest_path)),
        worker="aef_worker",
    )
    writer.record.model_checkpoint_sha256 = None
    writer.record.notes = "AEF track: no upstream checkpoint; payloads come from the manifest"
    writer.persist()

    def emit(event, message, level="info", progress=None):  # noqa: ANN001
        emit_event(request.run_id, event, message, level=level, progress=progress)

    actions = {"train": train, "probe": probe}
    handler = actions.get(request.action)
    if handler is None:
        message = f"aef worker implements probe/train only, got {request.action}"
        emit("worker.fail", message, level="error")
        writer.fail(message)
        return 1
    try:
        code = handler(request, writer, emit)
    except WorkerFailure as failure:
        message = str(failure)
        emit("worker.fail", message, level="error")
        writer.fail(message)
        return 1
    emit("worker.exit", f"exit code {code}", level="debug")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
