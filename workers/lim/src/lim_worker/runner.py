"""LIM worker entry point (MiT-Huge / MixMIM, iSAID T9).

``probe`` validates the manifest, the checkpoint hash, torch/GPU and the encoder
weights, then runs one real forward pass on a T9 tile and records the feature
pyramid, the effective stride and the cost. ``train`` runs the T9 protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest, manifest_sha256
from rs_backbone_eval.registry import ModelSpec, ProtocolSpec, TaskSpec, load_yaml_spec
from rs_backbone_eval.worker_runtime import RunWriter, emit_event

if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lim_worker.mit import load_encoder, mit_huge
    from lim_worker.t9 import (
        LimSegmentation,
        _read_png,
        load_rows,
        rgb_only,
        tile_geometry,
        train_t9,
    )
else:
    from .mit import load_encoder, mit_huge
    from .t9 import (
        LimSegmentation,
        _read_png,
        load_rows,
        rgb_only,
        tile_geometry,
        train_t9,
    )

TASK_IDS = {"T9"}


def emit(
    run_id: str, event: str, message: str, level: str = "info", progress: float | None = None
) -> None:
    emit_event(run_id, event, message, level=level, progress=progress)


def _fail(writer: RunWriter, run_id: str, event: str, message: str) -> int:
    emit(run_id, event, message, level="error")
    writer.fail(message)
    return 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def probe(request: WorkerRequest, writer: RunWriter, event_log) -> int:
    import torch

    run_id = request.run_id
    run_dir = Path(request.runs_root) / run_id
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    event_log(run_id, "probe.start", "validating manifest, checkpoint, environment")
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    strict_errors = audit.error_count - len(audit.leakage)
    if strict_errors:
        return _fail(
            writer,
            run_id,
            "probe.manifest",
            f"manifest audit failed with {strict_errors} errors: {audit.all_errors()[:3]}",
        )
    writer.update(tags={"manifest_samples": str(audit.sample_count)})
    event_log(
        run_id,
        "probe.manifest",
        f"manifest OK: {audit.sample_count} samples, splits {dict(audit.split_counts)}",
        progress=0.2,
    )

    checkpoint_value = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint_value or not Path(checkpoint_value).is_file():
        return _fail(
            writer,
            run_id,
            "probe.checkpoint",
            f"checkpoint missing via {model_spec.checkpoint_env}: {checkpoint_value}",
        )
    checkpoint = Path(checkpoint_value)
    writer.update(model_checkpoint_sha256=_sha256(checkpoint))
    event_log(
        run_id,
        "probe.checkpoint",
        f"checkpoint OK: {checkpoint.name} ({checkpoint.stat().st_size / 1e6:.0f} MB)",
        progress=0.35,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer.update(tags={"torch_version": torch.__version__, "device": device})
    event_log(run_id, "probe.torch", f"torch {torch.__version__}, device {device}", progress=0.5)

    encoder = mit_huge()
    stats = load_encoder(checkpoint, encoder)
    if stats["missing"] or stats["unexpected"]:
        return _fail(writer, run_id, "probe.checkpoint", f"encoder load mismatch: {stats}")
    model = LimSegmentation(encoder, task_spec.num_outputs).to(device).eval()
    event_log(
        run_id,
        "probe.encoder",
        f"MiT-Huge loaded: {stats['loaded']} keys, {stats['dropped']} decoder keys dropped",
        progress=0.65,
    )

    rows = load_rows(Path(request.manifest_path), "test")
    if not rows:
        return _fail(writer, run_id, "probe.manifest", "manifest has no test tiles")
    crop, stride = tile_geometry(rows)
    row = rows[0]
    data_root = Path(request.data_root)
    pixels = rgb_only(_read_png(data_root / row["image_paths"][0], tuple(row["window"]))).astype(
        "float32"
    )
    # Probe normalisation only (/255, mean=std=0.5): the real run computes the
    # training-tile mean/std, which the probe cannot know before it trains.
    pixels = (pixels / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).to(device)[None]

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        with torch.inference_mode():
            features = model.encoder(tensor)
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        return _fail(
            writer,
            run_id,
            "probe.forward",
            f"{pixels.shape[0]}x{pixels.shape[1]} tile does not fit in GPU memory",
        )
    latency = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    finite = bool(all(torch.isfinite(feat).all() for feat in features))
    shape = "x".join(str(dim) for dim in features[-1].shape)
    event_log(
        run_id,
        "probe.forward",
        f"pyramid {[tuple(f.shape[1:]) for f in features]} ({shape}), "
        f"strides {[crop // f.shape[-1] for f in features]}, {latency:.2f}s, "
        f"peak {peak_mb:.0f} MB, finite={finite}",
        progress=1.0,
    )
    report = {
        "tile": row["sample_id"],
        "crop_px": crop,
        "stride_px": stride,
        "pyramid": [list(feat.shape) for feat in features],
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "head_params": sum(p.numel() for p in model.head.parameters()),
        "normalisation": "probe-only /255 then mean=std=0.5",
        "latency_s": round(latency, 3),
        "peak_gpu_mb": round(peak_mb, 1),
        "finite": finite,
    }
    (run_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    writer.add_artifact("report", "probe_report.json", "LIM probe report (pyramid and cost)")
    if not finite:
        return _fail(writer, run_id, "probe.forward", "non-finite values in LIM features")
    writer.update(
        tags={
            "pyramid": shape,
            "crop_px": str(crop),
            "stride_px": str(stride),
            "peak_gpu_mb": f"{peak_mb:.0f}",
        }
    )
    writer.complete(
        f"probe passed: LIM MiT-Huge on {row['sample_id']} ({crop}px tile) -> "
        f"{[tuple(f.shape[1:]) for f in features]}, peak {peak_mb:.0f} MB"
    )
    return 0


def train_action(request: WorkerRequest, writer: RunWriter, event_log) -> int:
    run_id = request.run_id
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)
    if task_spec.task_id not in TASK_IDS:
        return _fail(
            writer,
            run_id,
            "train.task",
            f"lim worker only implements {sorted(TASK_IDS)}, got {task_spec.task_id}",
        )
    protocol: ProtocolSpec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    if protocol.protocol_id not in {"upernet_frozen", "upernet_finetune"}:
        return _fail(
            writer,
            run_id,
            "train.protocol",
            f"the lim worker implements the UPerNet tracks only, got {protocol.protocol_id}",
        )
    checkpoint_value = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint_value or not Path(checkpoint_value).is_file():
        return _fail(
            writer,
            run_id,
            "train.checkpoint",
            f"checkpoint missing via {model_spec.checkpoint_env}: {checkpoint_value}",
        )
    checkpoint = Path(checkpoint_value)
    writer.attach_checkpoint(checkpoint)
    try:
        return train_t9(
            request,
            writer,
            event_log,
            checkpoint=checkpoint,
            freeze_backbone=not protocol.train_backbone,
            num_classes=task_spec.num_outputs,
            metric_name=task_spec.primary_metric.name,
        )
    except Exception as exc:  # noqa: BLE001 - the worker must record every failure
        return _fail(writer, run_id, "train.crash", f"{type(exc).__name__}: {exc}")


ACTIONS = {"probe": probe, "train": train_action}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: runner.py <request.json>", file=sys.stderr)
        return 2
    request = WorkerRequest.model_validate(
        json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    )
    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    protocol_spec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    model_spec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    writer = RunWriter(
        Path(request.runs_root) / request.run_id,
        run_id=request.run_id,
        task_spec=task_spec,
        protocol_spec=protocol_spec,
        model_id=model_spec.model_id,
        seed=request.seed,
        manifest_sha256=manifest_sha256(Path(request.manifest_path)),
        worker="lim_worker",
    )
    writer.persist()

    emit(request.run_id, "worker.start", f"action {request.action}, seed {request.seed}")
    handler = ACTIONS.get(request.action)
    if handler is None:
        return _fail(
            writer,
            request.run_id,
            "worker.action",
            f"action not implemented in lim worker: {request.action}",
        )
    code = handler(request, writer, emit)
    emit(request.run_id, "worker.exit", f"exit code {code}", level="debug")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
