"""Locked test evaluation of a trained T13 fusion plugin (plan v3 §6).

The trainer selects the plugin on validation tiles and never reads the test split.
This step loads one selected plugin, checks that it belongs to the pinned FT-B0
baseline and manifest, and scores the test split once. It records mean IoU (primary)
and Boundary IoU, and sets ``test_evaluated`` on its own run.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.evaluation import to_metric_records

from .arms import ArmModel
from .boundary import add_counts, boundary_counts, mean_boundary_iou
from .fusion_branches import FusionBranches
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
from .t13_fusion import FT_B0_BASELINES, FUSION_RECIPES, find_full_dependency, fused_logits
from .t13_fusion_train import FUSION_FORMAT, REPO_ROOT, _require_finite

PLUGIN_FILE = "fusion_plugin.pt"
PLUGIN_RUN_ENV = "LIM_WORKER_PLUGIN_RUN"


def check_plugin(
    plugin: dict,
    *,
    pin: dict,
    manifest_sha: str,
    protocol_id: str,
    use_nir: bool,
) -> None:
    """Refuse a plugin that is not the validated result of this protocol and baseline."""
    if plugin.get("format") != FUSION_FORMAT:
        raise RuntimeError(f"plugin format is {plugin.get('format')!r}, not {FUSION_FORMAT!r}")
    if plugin.get("protocol") != protocol_id:
        raise RuntimeError(f"plugin was trained for {plugin.get('protocol')}, not {protocol_id}")
    if plugin.get("baseline_run") != pin["run_id"]:
        raise RuntimeError(
            f"plugin baseline is {plugin.get('baseline_run')}, the pin is {pin['run_id']}"
        )
    if plugin.get("baseline_checkpoint_sha256") != pin["checkpoint_sha256"]:
        raise RuntimeError("plugin baseline checkpoint SHA-256 differs from the pinned baseline")
    if plugin.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("plugin was trained on another manifest than this request's")
    config = plugin.get("config") or {}
    if bool(config.get("nir")) != use_nir:
        raise RuntimeError("plugin branches differ from the protocol's recipe")


def evaluate_fusion(request, writer, emit, *, protocol, task_spec, model_id: str) -> int:
    recipe = FUSION_RECIPES.get(protocol.protocol_id)
    if recipe is None or recipe["sar"]:
        raise ValueError(f"no locked evaluate for {protocol.protocol_id} yet")
    plugin_run = os.environ.get(PLUGIN_RUN_ENV, "")
    if not plugin_run:
        raise ValueError(f"set {PLUGIN_RUN_ENV} to the run ID of the trained plugin")
    runs_root = Path(request.runs_root)
    plugin_path = runs_root / plugin_run / PLUGIN_FILE
    if not plugin_path.is_file():
        raise FileNotFoundError(f"no trained plugin at {plugin_path}")
    plugin = torch.load(plugin_path, map_location="cpu")

    manifest = Path(request.manifest_path)
    manifest_sha = manifest_sha256(manifest)
    pin = FT_B0_BASELINES[request.seed]
    check_plugin(
        plugin,
        pin=pin,
        manifest_sha=manifest_sha,
        protocol_id=protocol.protocol_id,
        use_nir=recipe["nir"],
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
    inputs = Inputs(
        encoder,
        np.asarray(plugin["band_mean"], dtype=np.float32),
        np.asarray(plugin["band_std"], dtype=np.float32),
        device,
        finetune=True,
    )
    branches = FusionBranches(CHANNELS, use_nir=recipe["nir"], use_sar=False).to(device).eval()
    branches.load_state_dict(plugin["state_dict"], strict=True)

    def logits_for(image_u8: torch.Tensor) -> torch.Tensor:
        image = image_u8.to(device)
        feats = inputs.features(image, grad=False)
        nir = nir_valid = None
        if recipe["nir"]:
            nir = inputs.plugin_image(image)[:, 3:4]
            nir_valid = torch.ones_like(nir)
        with inputs.autocast():
            logits = fused_logits(
                model.head, feats, branches, tuple(image.shape[-2:]), nir=nir, nir_valid=nir_valid
            )
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
        {"boundary_iou": boundary},
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
            "baseline_run": pin["run_id"],
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "boundary_band": "2% of sheet diagonal, square neighbourhood",
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
