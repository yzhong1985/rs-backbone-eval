"""Locked test evaluation of a frozen four-band first layer plugin (whu_ft_nir4).

The trainer (t13_stem4_train) selects the NIR convolution on validation tiles and never
reads the test split. This step loads one selected ``stem4_plugin.pt``, checks that it
belongs to the pinned FT-B0 baseline and manifest, and scores the test split once. It
records mean IoU (primary) and Boundary IoU, and sets ``test_evaluated`` on its own run.
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
from .mit import mit_huge
from .stem4 import add_nir_stem
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
from .t13_fusion import FT_B0_BASELINES, find_full_dependency
from .t13_fusion_eval import PLUGIN_RUN_ENV, check_source_run
from .t13_fusion_train import REPO_ROOT, _require_finite
from .t13_stem4_train import STEM4_FORMAT, STEM4_PROTOCOL

PLUGIN_FILE = "stem4_plugin.pt"


def check_stem4_plugin(plugin: dict, *, pin: dict, manifest_sha: str) -> None:
    """Refuse a stem4 plugin that is not the validated result of this baseline."""
    if plugin.get("format") != STEM4_FORMAT:
        raise RuntimeError(f"plugin format is {plugin.get('format')!r}, not {STEM4_FORMAT!r}")
    if plugin.get("protocol") != STEM4_PROTOCOL:
        raise RuntimeError(f"plugin was trained for {plugin.get('protocol')}, not {STEM4_PROTOCOL}")
    if plugin.get("baseline_run") != pin["run_id"]:
        raise RuntimeError(
            f"plugin baseline is {plugin.get('baseline_run')}, the pin is {pin['run_id']}"
        )
    if plugin.get("baseline_checkpoint_sha256") != pin["checkpoint_sha256"]:
        raise RuntimeError("plugin baseline checkpoint SHA-256 differs from the pinned baseline")
    if plugin.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("plugin was trained on another manifest than this request's")
    config = plugin.get("config") or {}
    if not config.get("nir") or config.get("sar"):
        raise RuntimeError("plugin is not a NIR-only four-band first layer")
    if "nir.weight" not in (plugin.get("state_dict") or {}):
        raise RuntimeError("plugin state_dict has no nir.weight")


def evaluate_stem4(request, writer, emit, *, protocol, task_spec, model_id: str) -> int:
    if protocol.protocol_id != STEM4_PROTOCOL:
        raise ValueError(f"no stem4 locked evaluate for {protocol.protocol_id}")
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
    check_stem4_plugin(plugin, pin=pin, manifest_sha=manifest_sha)
    plugin_sha = check_source_run(
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
    stem = add_nir_stem(encoder)
    learned = plugin["state_dict"]["nir.weight"]
    if tuple(learned.shape) != tuple(stem.nir.weight.shape):
        raise RuntimeError(
            f"plugin nir.weight is {tuple(learned.shape)}, the stem expects "
            f"{tuple(stem.nir.weight.shape)}"
        )
    with torch.no_grad():
        stem.nir.weight.copy_(learned)
    encoder = encoder.to(device).eval()
    model = model.to(device).eval()
    inputs = Inputs(
        encoder,
        np.asarray(plugin["band_mean"], dtype=np.float32),
        np.asarray(plugin["band_std"], dtype=np.float32),
        device,
        finetune=True,
    )

    def logits_for(image_u8: torch.Tensor) -> torch.Tensor:
        """Four-band input through the frozen encoder and head (as in training)."""
        image = image_u8.to(device)
        rgb = (image[:, :3].float() - inputs.rgb_mean) / inputs.rgb_std
        nir = inputs.plugin_image(image)[:, 3:4]
        x = torch.cat([rgb, nir], dim=1)
        with inputs.autocast():
            levels = encoder(x)
            logits = model.head(levels, output_size=tuple(image.shape[-2:]))
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
            "plugin_sha256": plugin_sha,
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
