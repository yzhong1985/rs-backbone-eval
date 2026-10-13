"""AnySat worker entry point.

Executes one ``WorkerRequest``. The ``probe`` action validates the contract
(manifest audit, checkpoint hash, torch/GPU, pinned upstream source) and runs
one real forward pass on a T1 chip through AnySat's native Sentinel-2 entry
(ten bands), measuring output stride, latency and peak memory. Training
actions follow once input adaptation for the six-band protocol lands.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.registry import ModelSpec, ProtocolSpec, TaskSpec, load_yaml_spec
from rs_backbone_eval.worker_runtime import RunWriter, emit_event

if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anysat_worker.s2_probe import (  # noqa: F401
        S2_BAND_INDICES,
        S2_BAND_NAMES,
        add_upstream_to_path,
        read_chip,
        training_normalization,
        upstream_commit,
    )
    from anysat_worker.t1_adapter import train_adapter
else:
    from .s2_probe import (  # noqa: F401
        S2_BAND_INDICES,
        S2_BAND_NAMES,
        add_upstream_to_path,
        read_chip,
        training_normalization,
        upstream_commit,
    )
    from .t1_adapter import train_adapter


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


def _build_model(checkpoint: Path, source_root: Path):
    import torch

    add_upstream_to_path()
    from hubconf import AnySat  # type: ignore[import-not-found]  # upstream checkout

    model = AnySat(model_size="base", flash_attn=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.model.load_state_dict(state)
    return model


def probe(request: WorkerRequest, writer: RunWriter, event_log) -> int:
    import torch

    run_id = request.run_id
    run_dir = Path(request.runs_root) / run_id
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    event_log(
        run_id, "probe.start", "validating manifest, checkpoint, upstream source, environment"
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
    except RuntimeError as exc:
        return _fail(writer, run_id, "probe.upstream", str(exc))
    commit = upstream_commit(source_root)
    writer.update(tags={"anysat_src": str(source_root), "anysat_commit": commit})
    event_log(
        run_id, "probe.upstream", f"AnySat source {commit[:12]} at {source_root}", progress=0.5
    )

    import torch as _torch

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    writer.update(tags={"torch_version": _torch.__version__, "device": device})
    event_log(run_id, "probe.torch", f"torch {_torch.__version__}, device {device}", progress=0.6)

    model = _build_model(checkpoint, source_root)
    model = model.to(device).eval()

    data_root = Path(request.data_root)
    mean, std, chips_used, norm_file = training_normalization(
        Path(request.manifest_path),
        data_root,
        Path(request.cache_root) / "anysat_norm",
        limit=int(os.environ.get("ANYSAT_NORM_CHIPS", "48")),
    )
    writer.update(tags={"norm_stats": Path(norm_file).name, "norm_chips": str(chips_used)})
    event_log(
        run_id,
        "probe.norm",
        f"train-split S2 statistics ready ({chips_used or 'cached'} chips)",
        progress=0.7,
    )

    with Path(request.manifest_path).open("r", encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    chip = read_chip(data_root / row["image_paths"][0], band_indices=S2_BAND_INDICES)
    chip = (chip - mean[:, None, None]) / std[:, None, None]
    day_of_year = row["timestamps"][0]
    event_log(
        run_id,
        "probe.input",
        f"chip {row['sample_id']} {tuple(chip.shape)}, S2 date {day_of_year}",
        progress=0.8,
    )

    grid_spec = os.environ.get("ANYSAT_PROBE_GRID", "320:512,160:512,80:256")
    attempts: list[dict] = []
    best: dict | None = None
    for index, spec in enumerate(grid_spec.split(",")):
        patch_size, crop = (int(part) for part in spec.split(":"))
        patch = chip[:, :crop, :crop]
        tensor = torch.from_numpy(patch)[None, None].to(device)
        dates = torch.tensor([[0]], dtype=torch.float32, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            with torch.no_grad():
                dense = model(
                    {"s2": tensor, "s2_dates": dates},
                    patch_size=patch_size,
                    output="dense",
                    output_modality="s2",
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            attempts.append({"patch_size_m": patch_size, "crop_px": crop, "status": "oom"})
            event_log(
                run_id,
                "probe.forward",
                f"patch {patch_size}m on {crop}px chip: out of memory",
                level="warning",
                progress=0.8 + 0.05 * (index + 1),
            )
            continue
        latency = time.time() - started
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        finite = bool(torch.isfinite(dense).all())
        stride = crop / dense.shape[1]
        record = {
            "patch_size_m": patch_size,
            "crop_px": crop,
            "dense_shape": list(dense.shape),
            "output_stride_px": stride,
            "latency_s": round(latency, 3),
            "peak_gpu_mb": round(peak_mb, 1),
            "finite": finite,
            "status": "ok" if finite else "non_finite",
        }
        attempts.append(record)
        event_log(
            run_id,
            "probe.forward",
            f"patch {patch_size}m on {crop}px: dense {tuple(dense.shape)}, "
            f"stride {stride:g}px, {latency:.2f}s, peak {peak_mb:.0f} MB, finite={finite}",
            progress=0.8 + 0.05 * (index + 1),
        )
        if finite and best is None:
            best = record

    report_path = run_dir / "probe_report.json"
    report_path.write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")
    writer.add_artifact("report", report_path.name, "AnySat probe sweep (patch size x crop)")
    if best is None:
        return _fail(
            writer, run_id, "probe.forward", "no patch size/crop combination fit in memory"
        )
    writer.update(
        tags={
            "dense_shape": "x".join(str(dim) for dim in best["dense_shape"]),
            "stride_px": f"{best['output_stride_px']:g}",
            "peak_gpu_mb": f"{best['peak_gpu_mb']:g}",
        }
    )
    writer.complete(
        f"probe passed: AnySat native S2 ten-band forward on {row['sample_id']}; "
        f"largest fitting config patch {best['patch_size_m']}m on {best['crop_px']}px "
        f"-> dense {tuple(best['dense_shape'])} at stride {best['output_stride_px']:g}px"
    )
    return 0


def train_action(request: WorkerRequest, writer: RunWriter, event_log) -> int:
    run_id = request.run_id
    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)

    if task_spec.task_id != "T1":
        return _fail(
            writer,
            run_id,
            "train.task",
            f"anysat worker only implements T1, got {task_spec.task_id}",
        )
    checkpoint = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint or not Path(checkpoint).is_file():
        return _fail(
            writer,
            run_id,
            "train.checkpoint",
            f"checkpoint missing via {model_spec.checkpoint_env}: {checkpoint}",
        )

    try:
        return train_adapter(
            request,
            writer,
            event_log,
            checkpoint=Path(checkpoint),
            task_bands=list(task_spec.bands),
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
        manifest_sha256=Path(request.manifest_path).name.removesuffix(".jsonl"),
        worker="anysat_worker",
    )
    writer.persist()

    emit(request.run_id, "worker.start", f"action {request.action}, seed {request.seed}")
    handler = ACTIONS.get(request.action)
    if handler is None:
        return _fail(
            writer,
            request.run_id,
            "worker.action",
            f"action not implemented yet in anysat worker: {request.action}",
        )
    code = handler(request, writer, emit)
    emit(request.run_id, "worker.exit", f"exit code {code}", level="debug")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
