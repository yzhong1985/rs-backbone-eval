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
else:
    from .s2_probe import (  # noqa: F401
        S2_BAND_INDICES,
        S2_BAND_NAMES,
        add_upstream_to_path,
        read_chip,
        training_normalization,
        upstream_commit,
    )


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
    tensor = torch.from_numpy(chip)[None, None].to(device)  # [B=1, T=1, C=10, H, W]
    dates = torch.tensor([[0]], dtype=torch.float32, device=device)
    day_of_year = row["timestamps"][0]
    event_log(
        run_id,
        "probe.input",
        f"chip {row['sample_id']} {tuple(tensor.shape)}, S2 date {day_of_year}",
        progress=0.8,
    )

    started = time.time()
    with torch.no_grad():
        dense = model(
            {"s2": tensor, "s2_dates": dates}, patch_size=40, output="dense", output_modality="s2"
        )
    latency = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    stride_y = tensor.shape[-2] / dense.shape[-2]
    stride_x = tensor.shape[-1] / dense.shape[-1]
    finite = bool(torch.isfinite(dense).all())
    writer.update(
        tags={
            "dense_shape": "x".join(str(dim) for dim in dense.shape),
            "stride": f"{stride_y:g}",
            "latency_s": f"{latency:.3f}",
        }
    )
    event_log(
        run_id,
        "probe.forward",
        f"dense {tuple(dense.shape)}, stride {stride_y:g}x{stride_x:g} px, "
        f"{latency:.2f}s, peak {peak_mb:.0f} MB, finite={finite}",
        progress=1.0,
    )
    if not finite:
        return _fail(writer, run_id, "probe.forward", "non-finite values in dense features")

    writer.complete(
        f"probe passed: AnySat native S2 ten-band forward on {row['sample_id']}, "
        f"dense {tuple(dense.shape)} at stride {stride_y:g}"
    )
    return 0


ACTIONS = {"probe": probe}


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
