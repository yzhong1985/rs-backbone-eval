"""Locked test evaluation of the DOFA F3 injector with its validation-selected head.

Loads one selected ``dofa_plugin.pt``, checks its source run and file SHA-256, then
scores the test split once. Records mean IoU (primary) and Boundary IoU.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.evaluation import to_metric_records

from .arms import ArmModel
from .boundary import add_counts, boundary_counts, boundary_metrics, mean_boundary_iou
from .mit import mit_huge
from .t13 import (
    CHANNELS,
    EFFECTIVE_BATCH,
    NUM_CLASSES,
    Inputs,
    _int_env,
    load_full_checkpoint,
    score_sheets,
)
from .t13_data import load_rows
from .t13_dofa_train import (
    DOFA_FORMAT,
    DOFA_PROTOCOL,
    DOFA_WAVE_UM,
    F3_LEVEL,
    DofaF3Plugin,
    load_dofa,
)
from .t13_fusion import FT_B0_BASELINES, find_full_dependency
from .t13_fusion_eval import PLUGIN_RUN_ENV, check_source_run, verify_plugin_bytes
from .t13_fusion_train import REPO_ROOT, _require_finite

PLUGIN_FILE = "dofa_plugin.pt"


def check_dofa_plugin(plugin: dict, *, pin: dict, manifest_sha: str) -> None:
    if plugin.get("format") != DOFA_FORMAT:
        raise RuntimeError(f"plugin format is {plugin.get('format')!r}, not {DOFA_FORMAT!r}")
    if plugin.get("protocol") != DOFA_PROTOCOL:
        raise RuntimeError(f"plugin was trained for {plugin.get('protocol')}, not {DOFA_PROTOCOL}")
    if plugin.get("baseline_run") != pin["run_id"]:
        raise RuntimeError("plugin baseline differs from the pinned baseline")
    if plugin.get("baseline_checkpoint_sha256") != pin["checkpoint_sha256"]:
        raise RuntimeError("plugin baseline checkpoint SHA-256 differs from the pinned baseline")
    if plugin.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("plugin was trained on another manifest than this request's")
    config = plugin.get("config") or {}
    if not config.get("dofa") or not config.get("head") or config.get("level") != F3_LEVEL:
        raise RuntimeError("plugin is not the DOFA F3 injector with a trained head")
    if "head_state" not in plugin or plugin["head_state"] is None:
        raise RuntimeError("plugin carries no head weights")


def evaluate_dofa(request, writer, emit, *, protocol, task_spec, model_id: str) -> int:
    if protocol.protocol_id != DOFA_PROTOCOL:
        raise ValueError(f"no DOFA locked evaluate for {protocol.protocol_id}")
    plugin_run = os.environ.get(PLUGIN_RUN_ENV, "")
    if not plugin_run:
        raise ValueError(f"set {PLUGIN_RUN_ENV} to the run ID of the trained plugin")
    runs_root = Path(request.runs_root)
    plugin_path = runs_root / plugin_run / PLUGIN_FILE
    if not plugin_path.is_file():
        raise FileNotFoundError(f"no trained plugin at {plugin_path}")
    raw = plugin_path.read_bytes()
    plugin_sha = verify_plugin_bytes(runs_root, plugin_run, plugin_file=PLUGIN_FILE, raw=raw)
    plugin = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    manifest = Path(request.manifest_path)
    manifest_sha = manifest_sha256(manifest)
    pin = FT_B0_BASELINES[request.seed]
    check_dofa_plugin(plugin, pin=pin, manifest_sha=manifest_sha)
    check_source_run(
        runs_root,
        plugin_run,
        plugin_file=PLUGIN_FILE,
        epochs=int(protocol.epochs or 0),
        plugin=plugin,
    )
    test_rows = load_rows(manifest, "test")
    if not test_rows:
        raise ValueError("the manifest has no T13 test tiles")

    dependency = find_full_dependency(
        runs_root,
        seed=request.seed,
        manifest_sha=manifest_sha,
        model_id=model_id,
        batch=EFFECTIVE_BATCH,
        repo=REPO_ROOT,
    )
    writer.attach_checkpoint(dependency)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = mit_huge(in_chans=3)
    model = ArmModel("B0", CHANNELS, NUM_CLASSES)
    load_full_checkpoint(dependency, encoder, model)
    encoder = encoder.to(device).eval()
    model = model.to(device).eval()
    model.head.load_state_dict(plugin["head_state"], strict=True)
    dofa = load_dofa().to(device)
    injector = DofaF3Plugin().to(device).eval()
    injector.load_state_dict(plugin["state_dict"], strict=True)
    inputs = Inputs(
        encoder,
        np.asarray(plugin["band_mean"], dtype=np.float32),
        np.asarray(plugin["band_std"], dtype=np.float32),
        device,
        finetune=True,
    )

    def logits_for(image_u8: torch.Tensor) -> torch.Tensor:
        image = image_u8.to(device)
        feats = list(inputs.features(image, grad=False))
        nir = inputs.plugin_image(image)[:, 3:4]
        tokens = dofa.forward_map(nir, [DOFA_WAVE_UM])
        feats[F3_LEVEL] = feats[F3_LEVEL].float() + injector.dofa.proj(tokens.float())
        with inputs.autocast():
            logits = model.head(feats, output_size=tuple(image.shape[-2:]))
        logits = logits.float()
        _require_finite("test logits", logits)
        return logits

    boundary_totals: dict[int, list[int]] = {}

    def on_sheet(label: np.ndarray, prediction: np.ndarray) -> None:
        add_counts(boundary_totals, boundary_counts(label, prediction, range(1, NUM_CLASSES)))

    started = time.time()
    with torch.inference_mode():
        metrics, groups = score_sheets(
            test_rows,
            Path(request.data_root),
            lambda image, _codes: logits_for(image),
            with_aef=False,
            batch=_int_env("LIM_WORKER_MICRO_BATCH", 2),
            on_sheet=on_sheet,
        )
    boundary = mean_boundary_iou(boundary_totals)
    records = to_metric_records(
        metrics, primary_name="mean_iou", direction="maximize", split="test"
    )
    records += to_metric_records(
        boundary_metrics(boundary_totals),
        primary_name="boundary_iou",
        direction="maximize",
        split="test",
        primary=False,
    )
    writer.add_metrics(records)
    writer.add_confusion(groups, NUM_CLASSES)
    writer.update(
        tags={
            "test_evaluated": "yes",
            "plugin_run": plugin_run,
            "plugin_sha256": plugin_sha,
            "baseline_run": pin["run_id"],
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "boundary_band": "2 px band: boundary plus one layer inside, labelled neighbours only",
            "test_sheets": str(len({r["group_id"] for r in test_rows})),
            "evaluate_seconds": f"{time.time() - started:.1f}",
        }
    )
    summary = (
        f"test mean_iou={metrics['mean_iou']:.4f} boundary_iou={boundary:.4f} "
        f"({protocol.protocol_id}, plugin {plugin_run})"
    )
    writer.complete(summary)
    emit(request.run_id, "evaluate.metrics", summary)
    return 0
