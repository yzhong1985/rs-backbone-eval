"""Clay v1.5 worker entry point.

The ``probe`` action validates the contract (manifest audit, checkpoint hash,
torch/GPU, pinned upstream source, sensor metadata) and runs one real forward
pass on a T1 chip through Clay's own segmentation encoder, measuring the
patch-token grid, stride, latency and peak memory.
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
    from clay_worker.clay_probe import (  # noqa: F401
        S2_BAND_INDICES,
        SENSOR,
        add_upstream_to_path,
        load_sensor_metadata,
        read_chip,
        upstream_commit,
    )
    from clay_worker.t1 import train_clay_t1
    from clay_worker.t2 import train_t2
    from clay_worker.t3 import train_t3
    from clay_worker.t4 import train_t4
    from clay_worker.t5 import train_t5
    from clay_worker.t8 import train_t8
else:
    from .clay_probe import (  # noqa: F401
        S2_BAND_INDICES,
        SENSOR,
        add_upstream_to_path,
        load_sensor_metadata,
        read_chip,
        upstream_commit,
    )
    from .t1 import train_clay_t1
    from .t2 import train_t2
    from .t5 import train_t5
    from .t8 import train_t8

CLAY_LARGE = {
    "mask_ratio": 0,
    "patch_size": 8,
    "shuffle": False,
    "dim": 1024,
    "depth": 24,
    "heads": 16,
    "dim_head": 64,
    "mlp_ratio": 4,
}


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
    import numpy as np
    import torch

    run_id = request.run_id
    run_dir = Path(request.runs_root) / run_id
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    event_log(
        run_id, "probe.start", "validating manifest, checkpoint, source, metadata, environment"
    )

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

    try:
        source_root = add_upstream_to_path()
        sensor_meta = load_sensor_metadata(source_root)
    except RuntimeError as exc:
        return _fail(writer, run_id, "probe.upstream", str(exc))
    commit = upstream_commit(source_root)
    writer.update(tags={"clay_src": str(source_root), "clay_commit": commit, "sensor": SENSOR})
    event_log(
        run_id,
        "probe.upstream",
        f"Clay source {commit[:12]} at {source_root}; {SENSOR} gsd {sensor_meta['gsd']}m",
        progress=0.5,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer.update(tags={"torch_version": torch.__version__, "device": device})
    event_log(run_id, "probe.torch", f"torch {torch.__version__}, device {device}", progress=0.6)

    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    encoder = SegmentEncoder(**CLAY_LARGE, ckpt_path=str(checkpoint))
    encoder = encoder.to(device).eval()
    event_log(
        run_id,
        "probe.encoder",
        f"SegmentEncoder loaded (dim {CLAY_LARGE['dim']}, patch {CLAY_LARGE['patch_size']}px)",
        progress=0.7,
    )

    data_root = Path(request.data_root)
    with Path(request.manifest_path).open("r", encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    chip = read_chip(data_root / row["image_paths"][0])

    band_order = sensor_meta["band_order"]
    means = np.asarray(
        [sensor_meta["bands"]["mean"][band] for band in band_order], dtype=np.float32
    )
    stds = np.asarray([sensor_meta["bands"]["std"][band] for band in band_order], dtype=np.float32)
    normalized = (chip - means[:, None, None]) / stds[:, None, None]

    waves = torch.tensor(
        [sensor_meta["bands"]["wavelength"][band] for band in band_order], dtype=torch.float32
    )
    # Clay expects 8 metadata dimensions: time [week, hour] x 4 and latlon
    # [lat, lon] x 4. The manifest currently carries no chip centroid or
    # acquisition hour, so they are zero-filled exactly as Clay does for 20%
    # of its own training chips (a documented follow-up is deriving week
    # sin/cos from s2_date and lat/lon from the GeoTIFF transform).
    datacube = {
        "pixels": torch.from_numpy(normalized)[None].to(device),
        "time": torch.zeros(1, 4, device=device),
        "latlon": torch.zeros(1, 4, device=device),
        "gsd": torch.tensor(sensor_meta["gsd"], device=device),
        "waves": waves.to(device),
        "platform": SENSOR,
    }
    _, _, height, width = datacube["pixels"].shape
    event_log(
        run_id,
        "probe.input",
        f"chip {row['sample_id']} {tuple(datacube['pixels'].shape)}, "
        f"S2 date {row['timestamps'][0]}",
        progress=0.8,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        with torch.no_grad():
            tokens = encoder(datacube)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _fail(
            writer,
            run_id,
            "probe.forward",
            f"{SENSOR} {height}x{width} does not fit in GPU memory "
            f"with patch {CLAY_LARGE['patch_size']}",
        )
    latency = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    patch = CLAY_LARGE["patch_size"]
    grid_h, grid_w = height // patch, width // patch
    finite = bool(torch.isfinite(tokens).all())
    event_log(
        run_id,
        "probe.forward",
        f"tokens {tuple(tokens.shape)} -> grid {grid_h}x{grid_w}, stride {patch}px "
        f"({patch * sensor_meta['gsd']:g}m), {latency:.2f}s, "
        f"peak {peak_mb:.0f} MB, finite={finite}",
        progress=1.0,
    )
    report = {
        "sensor": SENSOR,
        "chip": row["sample_id"],
        "input_shape": list(datacube["pixels"].shape),
        "token_shape": list(tokens.shape),
        "grid": [grid_h, grid_w],
        "stride_px": patch,
        "stride_m": patch * sensor_meta["gsd"],
        "latency_s": round(latency, 3),
        "peak_gpu_mb": round(peak_mb, 1),
        "finite": finite,
    }
    (run_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    writer.add_artifact("report", "probe_report.json", "Clay probe report (token grid and cost)")
    if not finite:
        return _fail(writer, run_id, "probe.forward", "non-finite values in Clay tokens")

    writer.update(
        tags={
            "token_shape": "x".join(str(dim) for dim in tokens.shape),
            "grid": f"{grid_h}x{grid_w}",
            "stride_m": f"{patch * sensor_meta['gsd']:g}",
            "peak_gpu_mb": f"{peak_mb:.0f}",
        }
    )
    writer.complete(
        f"probe passed: Clay {SENSOR} ten-band forward on {row['sample_id']} -> tokens "
        f"{tuple(tokens.shape)} grid {grid_h}x{grid_w} at stride {patch * sensor_meta['gsd']:g}m"
    )
    return 0


def train_action(request: WorkerRequest, writer: RunWriter, event_log) -> int:
    run_id = request.run_id
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    if task_spec.task_id not in {"T1", "T1S", "T2", "T3", "T4", "T5", "T8"}:
        return _fail(
            writer,
            run_id,
            "train.task",
            f"clay worker only implements T1/T1S/T2/T3/T4/T5/T8, got {task_spec.task_id}",
        )
    checkpoint = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint or not Path(checkpoint).is_file():
        return _fail(
            writer,
            run_id,
            "train.checkpoint",
            f"checkpoint missing via {model_spec.checkpoint_env}: {checkpoint}",
        )
    writer.attach_checkpoint(Path(checkpoint))
    protocol: ProtocolSpec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    if protocol.protocol_id == "frozen_adapter_two_stage":
        # Review P1: this worker has no stage boundary or feature cache, so the
        # run would be a plain frozen probe recorded as a two-stage result.
        return _fail(
            writer,
            run_id,
            "train.protocol",
            "the clay worker does not implement two-stage training (frozen_adapter_two_stage)",
        )
    try:
        if task_spec.task_id == "T8":
            return train_t8(
                request,
                writer,
                event_log,
                checkpoint=Path(checkpoint),
                freeze_backbone=not protocol.train_backbone,
            )
        if task_spec.task_id == "T4":
            return train_t4(
                request,
                writer,
                event_log,
                checkpoint=Path(checkpoint),
                task_bands=list(task_spec.bands),
                freeze_backbone=not protocol.train_backbone,
                metric_name=task_spec.primary_metric.name,
            )
        if task_spec.task_id == "T3":
            return train_t3(
                request,
                writer,
                event_log,
                checkpoint=Path(checkpoint),
                freeze_backbone=not protocol.train_backbone,
                metric_name=task_spec.primary_metric.name,
            )
        if task_spec.task_id == "T5":
            return train_t5(
                request,
                writer,
                event_log,
                checkpoint=Path(checkpoint),
                freeze_backbone=not protocol.train_backbone,
                metric_name=task_spec.primary_metric.name,
            )
        if task_spec.task_id == "T2":
            return train_t2(
                request,
                writer,
                event_log,
                checkpoint=Path(checkpoint),
                task_bands=list(task_spec.bands),
                freeze_backbone=not protocol.train_backbone,
                metric_name=task_spec.primary_metric.name,
            )
        return train_clay_t1(
            request,
            writer,
            event_log,
            checkpoint=Path(checkpoint),
            sensor_kind="sar" if task_spec.task_id == "T1S" else "optical",
            task_bands=list(task_spec.bands),
            freeze_backbone=not protocol.train_backbone,
        )
    except Exception as exc:  # noqa: BLE001 - worker must record every failure
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
        worker="clay_worker",
    )
    writer.persist()

    emit(request.run_id, "worker.start", f"action {request.action}, seed {request.seed}")
    handler = ACTIONS.get(request.action)
    if handler is None:
        return _fail(
            writer,
            request.run_id,
            "worker.action",
            f"action not implemented yet in clay worker: {request.action}",
        )
    code = handler(request, writer, emit)
    emit(request.run_id, "worker.exit", f"exit code {code}", level="debug")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
