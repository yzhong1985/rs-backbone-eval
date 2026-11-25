"""Locked test evaluation of a trained T13 fusion plugin (plan v3 §6).

The trainer selects the plugin on validation tiles and never reads the test split.
This step loads one selected plugin, checks that it belongs to the pinned FT-B0
baseline and manifest, and scores the test split once. It records mean IoU (primary)
and Boundary IoU, and sets ``test_evaluated`` on its own run.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.evaluation import to_metric_records

from .arms import ArmModel
from .boundary import add_counts, boundary_counts, boundary_metrics, mean_boundary_iou
from .fusion_branches import GALILEO_LEVELS, FusionBranches, GalileoBranch
from .galileo_index import load_index, verify_token_index
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
    galileo: bool = False,
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
    if bool(config.get("nir")) != use_nir or bool(config.get("galileo", False)) != galileo:
        raise RuntimeError("plugin branches differ from the protocol's recipe")


def verify_plugin_bytes(runs_root: Path, plugin_run: str, *, plugin_file: str, raw: bytes) -> str:
    """Check the plugin's bytes against the SHA-256 its training run registered.

    Runs before ``torch.load``: a replaced file is refused without unpickling it. Returns
    the SHA-256 of ``raw`` for the evaluate ledger.
    """
    record_path = runs_root / plugin_run / "run.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"no run record for plugin run {plugin_run}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    registered = (record.get("tags") or {}).get("plugin_sha256")
    if not registered:
        raise RuntimeError(f"plugin run {plugin_run} registered no plugin SHA-256")
    current = hashlib.sha256(raw).hexdigest()
    if current != registered:
        raise RuntimeError(f"{plugin_file} differs from the SHA-256 its training run registered")
    return current


def check_source_run(
    runs_root: Path, plugin_run: str, *, plugin_file: str, epochs: int, plugin: dict
) -> None:
    """Refuse a plugin unless its source run is a completed, full-budget training run.

    The source ``run.json`` must be ``completed``, not superseded, must have run the
    protocol's epoch budget, and must register the plugin file as an artifact. The
    plugin must come from a non-smoke export; plugins without the smoke flag predate
    it and are refused.
    """
    record_path = runs_root / plugin_run / "run.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"no run record for plugin run {plugin_run}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError(f"plugin run {plugin_run} is {record.get('status')}, not completed")
    if (record.get("tags") or {}).get("superseded"):
        raise RuntimeError(f"plugin run {plugin_run} is superseded")
    done = int((record.get("training") or {}).get("epochs_completed") or 0)
    if done != epochs:
        raise RuntimeError(f"plugin run trained {done} epochs; the protocol budget is {epochs}")
    if plugin.get("smoke", True):
        raise RuntimeError("plugin comes from a smoke run, or predates the smoke flag")
    if not any(a.get("path") == plugin_file for a in record.get("artifacts") or []):
        raise RuntimeError(f"{plugin_file} is not a registered artifact of {plugin_run}")


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
    raw = plugin_path.read_bytes()
    plugin_sha = verify_plugin_bytes(runs_root, plugin_run, plugin_file=PLUGIN_FILE, raw=raw)
    plugin = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    manifest = Path(request.manifest_path)
    manifest_sha = manifest_sha256(manifest)
    pin = FT_B0_BASELINES[request.seed]
    check_plugin(
        plugin,
        pin=pin,
        manifest_sha=manifest_sha,
        protocol_id=protocol.protocol_id,
        use_nir=recipe["nir"],
        galileo=bool(recipe.get("galileo")),
    )
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
    inputs = Inputs(
        encoder,
        np.asarray(plugin["band_mean"], dtype=np.float32),
        np.asarray(plugin["band_std"], dtype=np.float32),
        device,
        finetune=True,
    )
    galileo = bool(recipe.get("galileo"))
    galileo_dir = None
    if galileo:
        token_dir = os.environ.get("LIM_WORKER_GALILEO_TOKENS", "")
        if not token_dir or not Path(token_dir).is_dir():
            raise ValueError("set LIM_WORKER_GALILEO_TOKENS to the precomputed Galileo token dir")
        galileo_dir = Path(token_dir)
        recorded = (plugin.get("config") or {}).get("galileo_token_digest") or {}
        if "train" not in recorded or "validation" not in recorded:
            raise RuntimeError("plugin records no Galileo token digests; retrain it")
        for split in ("train", "validation"):
            if load_index(galileo_dir, split)["digest"] != recorded[split]:
                raise RuntimeError(f"{split} token index differs from the cache the plugin used")
        galileo_test_digest = verify_token_index(galileo_dir, "test", test_rows)
        if load_index(galileo_dir, "test")["meta"] != load_index(galileo_dir, "train")["meta"]:
            raise RuntimeError("test token cache differs in encoder, band map or manifest")
        galileo_levels = tuple(recipe.get("galileo_levels", GALILEO_LEVELS))
        branches = GalileoBranch(CHANNELS, levels=galileo_levels).to(device).eval()
    else:
        branches = FusionBranches(CHANNELS, use_nir=recipe["nir"], use_sar=False).to(device).eval()
    branches.load_state_dict(plugin["state_dict"], strict=True)
    # A head-training plugin carries the UPerNet weights it was trained with.
    expects_head = protocol.protocol_id in {
        "whu_ft_nir_head",
        "whu_ft_galileo_head",
        "whu_ft_galileo_f23_head",
    }
    if bool((plugin.get("config") or {}).get("head")) != expects_head:
        raise RuntimeError("plugin head training differs from the protocol")
    if expects_head:
        model.head.load_state_dict(plugin["head_state"], strict=True)

    def logits_for(image_u8: torch.Tensor, tokens: torch.Tensor | None = None) -> torch.Tensor:
        image = image_u8.to(device)
        feats = inputs.features(image, grad=False)
        if galileo:
            tokens = tokens.to(device).float()
            valid = torch.ones_like(tokens[:, :1])
            with inputs.autocast():
                levels = branches(feats, tokens, valid)
                logits = model.head(levels, output_size=tuple(image.shape[-2:]))
            logits = logits.float()
            _require_finite("test logits", logits)
            return logits
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
            lambda image, extra: logits_for(image, extra),
            with_aef=False,
            batch=_int_env("LIM_WORKER_MICRO_BATCH", 2),
            on_sheet=on_sheet,
            galileo_dir=galileo_dir,
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
            "baseline_run": pin["run_id"],
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "plugin_sha256": plugin_sha,
            "galileo_token_digest_test": galileo_test_digest if galileo else "",
            "galileo_token_digest_train": recorded["train"] if galileo else "",
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
