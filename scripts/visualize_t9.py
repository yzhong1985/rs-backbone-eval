#!/usr/bin/env python3
"""Render qualitative iSAID (T9) predictions for a completed LIM run.

Usage (LIM worker environment has torch/rasterio/PIL):

    PYTHONPATH=src:workers/lim/src workers/lim/.venv/bin/python \
        scripts/visualize_t9.py --run-id lim_mit_huge-T9-upernet_frozen-train-17-v4 \
        --output-dir /home/yzhong/rs-eval/audits/t9-20261106 --images 6

The run's `latest.pt` is the epoch-boundary model state (T9 B2U is frozen and
does not write a validation-selected best checkpoint). The script rebuilds the
same MiT-Huge + UPerNet model, runs the whole-image sliding-window inference on
a few test groups, and writes per-image four-panel figures plus a contact sheet.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from lim_worker.mit import mit_huge
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    payload = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    settings = payload["identity"]["settings"]
    mean, std = _norm_from_tags(tags)
    crop = int(settings["crop"])
    stride = crop // 2
    data_root = Path(request["data_root"])
    rows = load_rows(Path(request["manifest_path"]), "test")
    _, stride = tile_geometry(rows)

    encoder = mit_huge(in_chans=3, drop_path_rate=0.0)
    model = LimSegmentation(encoder, num_classes=num_classes, fpn_dim=int(settings["fpn_dim"]))
    model.load_state_dict(payload["model"])
    model.to(args.device).eval()

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
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
            tensor = (
                tensor.to(args.device) - torch.from_numpy(mean).to(args.device)
            ) / torch.from_numpy(std).to(args.device)
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
