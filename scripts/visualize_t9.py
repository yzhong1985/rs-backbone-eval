#!/usr/bin/env python3
"""Render qualitative iSAID (T9) predictions for a completed LIM run.

Usage (LIM worker environment has torch/rasterio/PIL):

    PYTHONPATH=src:workers/lim/src workers/lim/.venv/bin/python \
        scripts/visualize_t9.py --run-id lim_mit_huge-T9-upernet_frozen-train-17-v4 \
        --output-dir /home/yzhong/rs-eval/audits/t9-20261106 --images 6

Checks `best_checkpoint.pt`, then `head_only.pt`, then `latest.pt` (latest may
be pruned after a completed run). `head_only.pt` needs `LIM_CHECKPOINT` for the
frozen backbone. On `--device auto` the script only uses CUDA when the GPU is
idle, so it cannot steal the card from a running lane; it also takes an
exclusive output-dir lock. The script rebuilds the same MiT-Huge + UPerNet
model, runs whole-image sliding-window inference and writes per-image panels.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from lim_worker.mit import mit_huge
from lim_worker.replay import check_loaded_keys
from lim_worker.t9 import LimSegmentation, load_rows, tile_geometry
from PIL import Image, ImageDraw

from rs_backbone_eval.data.isaid_io import ISAID_CLASS_NAMES, PALETTE, read_label, read_rgb
from rs_backbone_eval.evaluation.windows import predict_image

# Inverse of the iSAID colour lookup, indexed by class id.
_INVERSE = {value: key for key, value in PALETTE.items()}
CLASS_COLORS = np.zeros((len(ISAID_CLASS_NAMES), 3), dtype=np.uint8)
for class_id, colour in _INVERSE.items():
    CLASS_COLORS[class_id] = colour


def _colorize(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[labels]


def _panel(rgb: np.ndarray, labels: np.ndarray, title: str, size: int = 384) -> Image.Image:
    image = Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill="black")
    return canvas


def _grid(panels: list[list[Image.Image]], titles: list[str], out: Path) -> None:
    if not panels:
        return
    width = sum(panel.width for panel in panels[0])
    height = sum(row[0].height for row in panels)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in panels:
        x = 0
        for panel in row:
            canvas.paste(panel, (x, y))
            x += panel.width
        y += row[0].height
    canvas.save(out)


def _pick_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        used = int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).splitlines()[0]
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return "cpu"
    return "cuda" if used <= 2000 else "cpu"


def _acquire_lock(output_dir: Path):
    """flock-based lock: the kernel releases it even after SIGKILL/OOM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".visualize.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_backbone(path: Path, record: dict) -> None:
    """Refuse a head_only render unless the encoder is the run's checkpoint.

    ``head_only.pt`` carries no encoder weights, so the backbone comes from
    ``LIM_CHECKPOINT``. Without this check a wrong or partially loaded encoder
    rendered plausible panels and the mistake was invisible (review
    2026-11-07, P2). The run record stores the checkpoint hash that the worker
    loaded.
    """
    expected = (record.get("model_checkpoint_sha256") or "").strip()
    if not expected:
        raise SystemExit(
            "head_only render needs model_checkpoint_sha256 in run.json to verify LIM_CHECKPOINT"
        )
    actual = _sha256(path)
    if actual != expected:
        raise SystemExit(f"LIM_CHECKPOINT hash mismatch: {actual[:16]} != run.json {expected[:16]}")


def _load_checkpoint(run_dir: Path, tags: dict) -> tuple[dict, str, Path]:
    """Prefer a full/selected checkpoint; fall back to the resume state."""
    for name, kind in (
        ("best_checkpoint.pt", "best"),
        ("head_only.pt", "head_only"),
        ("latest.pt", "latest"),
    ):
        path = run_dir / name
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True), kind, path
    raise SystemExit(f"{run_dir} has no best_checkpoint.pt / head_only.pt / latest.pt")


def _norm_from_tags(tags: dict) -> tuple[np.ndarray, np.ndarray]:
    text = tags.get("normalization", "")
    match = re.search(r"mean\s*(\[[^\]]*\])\s*std\s*(\[[^\]]*\])", text)
    if not match:
        raise ValueError(f"cannot parse normalization tag: {text!r}")
    mean = np.asarray(json.loads(match.group(1).replace(" ", "")), dtype=np.float32)
    std = np.asarray(json.loads(match.group(2).replace(" ", "")), dtype=np.float32)
    return mean.reshape(1, 3, 1, 1), std.reshape(1, 3, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("/geodata/rs-eval/runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    run_dir = args.runs_root / args.run_id
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    if record.get("model_id") != "lim_mit_huge":
        raise SystemExit("this renderer currently supports the LIM T9 worker only")
    test_value = next(
        metric["value"]
        for metric in record["metrics"]
        if metric["name"] == "mean_iou" and metric["split"] == "test"
    )
    tags = record.get("tags") or {}
    num_classes = 16
    payload, checkpoint_kind, checkpoint_path = _load_checkpoint(run_dir, tags)
    settings = (payload.get("identity") or {}).get("settings") or {}
    mean, std = _norm_from_tags(tags)
    crop = int(settings.get("crop") or tags.get("train_crop_px") or 1024)
    fpn_dim = int(settings.get("fpn_dim") or 256)
    data_root = Path(request["data_root"])
    rows = load_rows(Path(request["manifest_path"]), "test")
    _, stride = tile_geometry(rows)
    device = _pick_device(args.device)
    print(f"checkpoint {checkpoint_kind}: {checkpoint_path}; device {device}", flush=True)

    encoder = mit_huge(in_chans=3, drop_path_rate=0.0)
    model = LimSegmentation(encoder, num_classes=num_classes, fpn_dim=fpn_dim)
    if checkpoint_kind == "head_only":
        backbone = Path(os.environ.get("LIM_CHECKPOINT", ""))
        if not backbone.is_file():
            raise SystemExit("head_only.pt needs LIM_CHECKPOINT for the frozen backbone")
        from lim_worker.mit import load_encoder

        _verify_backbone(backbone, record)
        counts = load_encoder(backbone, encoder)
        if counts["missing"] or counts["unexpected"]:
            raise SystemExit(
                f"LIM_CHECKPOINT does not match the model: "
                f"missing={counts['missing']} unexpected={counts['unexpected']}"
            )
        result = model.load_state_dict(payload["head_state_dict"], strict=False)
        try:
            check_loaded_keys(
                result.missing_keys,
                result.unexpected_keys,
                "head_only",
                backbone_prefix="encoder.",
            )
        except ValueError as error:
            raise SystemExit(f"head_only checkpoint mismatch: {error}") from error
    else:
        state = payload.get("state_dict") or payload.get("model")
        if state is None:
            raise SystemExit(f"{checkpoint_path} has no state dict")
        model.load_state_dict(state)
    model.to(device).eval()

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    keys = sorted(groups)
    parse = max(1, len(keys) // args.images)
    chosen: list[str] = []
    for offset in range(0, len(keys), parse):
        key = keys[offset]
        label = read_label(data_root / groups[key][0]["label_path"])
        if len(np.unique(label)) >= 2:
            chosen.append(key)
        if len(chosen) >= args.images:
            break
    lock = _acquire_lock(args.output_dir)
    if lock is None:
        raise SystemExit(f"another visualization is using {args.output_dir}")
    # Keep `lock` alive for the whole process; flock is released on exit/kill.

    panels: list[list[Image.Image]] = []
    for index, key in enumerate(chosen, 1):
        group = groups[key]
        image_path = data_root / group[0]["image_paths"][0]
        label_path = data_root / group[0]["label_path"]
        rgb = read_rgb(image_path)
        label = read_label(label_path)
        height, width = label.shape

        def predict(window: tuple[int, int, int, int], image_path: Path = image_path) -> np.ndarray:
            pixels = read_rgb(image_path, window).astype(np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))[None]
            tensor = (tensor.to(device) - torch.from_numpy(mean).to(device)) / torch.from_numpy(
                std
            ).to(device)
            with torch.inference_mode():
                logits = model(tensor, (window[3], window[2]))
            return logits[0].float().cpu().numpy()

        prediction = predict_image(
            width,
            height,
            num_classes=num_classes,
            crop=crop,
            stride=stride,
            predict=predict,
        )
        gt_color = _colorize(label)
        pred_color = _colorize(prediction)
        error = np.zeros((height, width, 3), dtype=np.uint8)
        error[prediction == label] = (40, 40, 40)
        error[prediction != label] = (245, 70, 70)
        overlay = (
            (0.55 * rgb.astype(np.float32) + 0.45 * pred_color.astype(np.float32))
            .clip(0, 255)
            .astype(np.uint8)
        )
        row_panels = [
            _panel(rgb, label, f"{key} RGB"),
            _panel(gt_color, label, "ground truth"),
            _panel(pred_color, prediction, "prediction"),
            _panel(error, label, "correct dark / error red"),
            _panel(overlay, prediction, "RGB + prediction"),
        ]
        panels.append(row_panels)
        combined = Image.new(
            "RGB", (sum(panel.width for panel in row_panels), row_panels[0].height)
        )
        x = 0
        for panel in row_panels:
            combined.paste(panel, (x, 0))
            x += panel.width
        combined.save(args.output_dir / f"t9_{args.run_id}_{key}.png")
        print(f"[{index}/{len(chosen)}] {key} saved", flush=True)

    _grid(panels, chosen, args.output_dir / f"t9_{args.run_id}_grid.png")
    lines = [
        f"# T9 qualitative predictions: `{args.run_id}`",
        "",
        f"- test mean IoU: **{test_value:.4f}**",
        f"- checkpoint: `{run_dir / 'latest.pt'}`",
        f"- crop/stride: {crop}/{stride}",
        f"- images: {', '.join(chosen)}",
        "",
        "Per-image files contain RGB / ground truth / prediction / error / overlay.",
        "",
        "## iSAID classes",
        "",
    ]
    lines += [f"- {index}: {name}" for index, name in enumerate(ISAID_CLASS_NAMES)]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
