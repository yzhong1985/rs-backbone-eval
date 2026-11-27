"""Locked test evaluate for the four-band full fine-tune (whu_ft_b0_4band).

Loads the validation-selected full checkpoint of one completed training run (seed 17),
builds the four-band encoder and head it was saved with, and scores the test split once.
The training run never reads the test split; this step does, and records its metrics on
the evaluate run. LIM_WORKER_TRAIN_RUN names the training run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.normalization import normalization_seed

from .arms import ArmModel
from .mit import mit_huge
from .t13 import CHANNELS, NUM_CLASSES, Inputs, load_full_checkpoint, score_sheets
from .t13_data import band_statistics, load_rows

PROTOCOL = "whu_ft_b0_4band"
TRAIN_RUN_ENV = "LIM_WORKER_TRAIN_RUN"


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value else default


def evaluate_ft4(request, writer, emit, *, protocol, task_spec, model_id: str) -> int:
    if protocol.protocol_id != PROTOCOL:
        raise ValueError(f"no four-band full-FT evaluate for {protocol.protocol_id}")
    train_run = os.environ.get(TRAIN_RUN_ENV, "")
    if not train_run:
        raise ValueError(f"set {TRAIN_RUN_ENV} to the completed four-band training run")
    runs_root = Path(request.runs_root)
    record_path = runs_root / train_run / "run.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"no run record for {train_run}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError(f"{train_run} is {record.get('status')}, not completed")
    if (record.get("tags") or {}).get("superseded"):
        raise RuntimeError(f"{train_run} is superseded")
    checkpoint = runs_root / train_run / "best_full_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no full checkpoint at {checkpoint}")

    manifest = Path(request.manifest_path)
    data_root = Path(request.data_root)
    manifest_sha = manifest_sha256(manifest)
    train_rows = load_rows(manifest, "train")
    test_rows = load_rows(manifest, "test")
    if not test_rows:
        raise ValueError("the manifest has no T13 test tiles")
    # Same statistics as the training run: a fixed sample of train tiles, seed-controlled.
    band_mean, band_std = band_statistics(train_rows, data_root, 256, normalization_seed())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = mit_huge(in_chans=4)
    model = ArmModel("B0", CHANNELS, NUM_CLASSES)
    load_full_checkpoint(checkpoint, encoder, model)
    encoder = encoder.to(device).eval()
    model = model.to(device).eval()
    inputs = Inputs(encoder, band_mean, band_std, device, finetune=True, in_bands=4)

    def predict(image: torch.Tensor) -> torch.Tensor:
        feats = inputs.features(image.to(device), grad=False)
        with inputs.autocast():
            logits = model.head(feats, output_size=tuple(image.shape[-2:]))
        return logits.float()

    limit = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    started = time.time()
    with torch.inference_mode():
        metrics, groups = score_sheets(
            test_rows,
            data_root,
            lambda image, extra: predict(image),
            with_aef=False,
            batch=_int_env("LIM_WORKER_MICRO_BATCH", 2),
            limit=limit,
        )
    records = to_metric_records(
        metrics, primary_name="mean_iou", direction="maximize", split="test"
    )
    writer.add_metrics(records)
    writer.add_confusion(groups, NUM_CLASSES)
    writer.attach_checkpoint(checkpoint)
    writer.update(
        tags={
            "test_evaluated": "yes",
            "train_run": train_run,
            "manifest_sha256": manifest_sha,
            "checkpoint_sha256": record.get("model_checkpoint_sha256") or "",
            "input_bands": "4",
            "evaluate_seconds": f"{time.time() - started:.1f}",
            "test_sheets": str(len({r["group_id"] for r in test_rows})),
            "evaluate_limit": str(limit),
        }
    )
    summary = f"test mean_iou={metrics['mean_iou']:.4f} ({PROTOCOL}, training {train_run})" + (
        " [SMOKE]" if limit else ""
    )
    writer.complete(summary)
    emit(request.run_id, "evaluate.metrics", summary)
    return 0
