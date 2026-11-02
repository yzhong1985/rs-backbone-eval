"""G1 cross-check: score the reference detector with our own merge + AP.

The G1 anchor is the MMRotate R50 Rotated Faster R-CNN checkpoint
(``rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth``, 1x, le90) on the
DOTA-v1.0 validation split at 1024 px with gap 200 (stride 824). MMRotate's own
DOTAMetric reports **mAP 0.5454** there (VOC-2010 AP, patch merge by rotated NMS
at IoU 0.1).

This script re-runs the same checkpoint over the same patches, shifts every
patch prediction into image coordinates, merges them with
``rs_backbone_eval.detection.nms.merge_tiles`` (IoU 0.1 to match the reference
merge) and scores the result with ``rs_backbone_eval.detection.ap.evaluate``.
If our merge and AP code are sound the two numbers should agree within a couple
of points: our AP interpolates over 101 COCO points and matches rotated
polygons directly, so exact equality is not expected.

Not part of the worker/control-plane environments: it needs the dedicated
MMRotate environment (torch 2.0.1+cu118, mmcv 2.0.1, mmdet 3.1.0,
mmrotate 1.0.0rc1; see ``docs/design/FAIR1M-2.0旋转目标检测评测方案`` §7.1.1).
Run it like::

    cd ~/rs-eval/tools/mmrotate-src && \
    PYTHONPATH=.:/path/to/rs-backbone-eval/src \
    ~/rs-eval/tools/mmrotate-venv/bin/python scripts/g1_reference_crosscheck.py

Beware: the pure-Python polygon matching makes the final evaluate() slow on the
full 458-scene validation set (tens of minutes); ``--limit`` subsets it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmrotate.utils import register_all_modules  # noqa: E402

from rs_backbone_eval.detection.anchors import polygons  # noqa: E402
from rs_backbone_eval.detection.ap import Predictions, Targets, evaluate  # noqa: E402
from rs_backbone_eval.detection.nms import Detection, merge_tiles  # noqa: E402

# MMRotate 1.x DOTA class order (mmrotate/datasets/dota.py METAINFO).
CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]
INDEX = {name: index for index, name in enumerate(CLASSES)}
PATCH = re.compile(r"^(?P<name>.+)__(?P<size>\d+)__(?P<x>-?\d+)___(?P<y>-?\d+)$")

# The reference merge (DOTAMetric.merge_results) NMSes patches at IoU 0.1.
MERGE_IOU = 0.1
REFERENCE_MAP = 0.5454


def read_gt(path: Path, drop_difficult: bool) -> Targets:
    target = Targets()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        coords = np.asarray([float(value) for value in parts[:8]], dtype=np.float64)
        name, difficult = parts[8], int(float(parts[9]))
        if name not in INDEX:
            continue
        if drop_difficult and difficult == 1:
            continue
        target.append(coords.reshape(4, 2), INDEX[name])
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("/geodata/rs-eval/data/raw/isaid/dota/split_val_1024_824"),
        help="MMRotate img_split.py output (images/ + annfiles/)",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/geodata/rs-eval/data/raw/isaid/dota/val/labelTxt-v1.0/labelTxt"),
        help="Full-image DOTA labelTxt directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/home/yzhong/rs-eval/tools/mmrotate-src/configs/rotated_faster_rcnn/"
            "rotated-faster-rcnn-le90_r50_fpn_1x_dota.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/yzhong/rs-eval/weights/mmrotate/"
            "rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth"
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="first N scenes only")
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args()

    register_all_modules()
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")
    model.eval()

    groups: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    for image in sorted((args.split / "images").glob("*.png")):
        match = PATCH.match(image.stem)
        if match is None:
            raise SystemExit(f"cannot parse patch id {image.stem}")
        groups[match.group("name")].append((int(match.group("x")), int(match.group("y")), image))
    names = sorted(groups)[: args.limit or None]

    predictions: dict[str, Predictions] = {}
    started = time.time()
    for count, name in enumerate(names, start=1):
        tile_outputs = []
        for x, y, image in sorted(groups[name]):
            result = inference_detector(model, str(image))
            instances = result.pred_instances.cpu()
            boxes = torch.from_numpy(np.asarray(instances.bboxes, dtype=np.float32)).reshape(-1, 5)
            detections = []
            if boxes.numel():
                corners = polygons(boxes).numpy() + np.array([x, y], dtype=np.float64)
                for polygon, score, label in zip(
                    corners,
                    np.asarray(instances.scores, dtype=np.float64),
                    np.asarray(instances.labels, dtype=np.int64),
                    strict=True,
                ):
                    detections.append(Detection(polygon, float(score), int(label)))
            tile_outputs.append((x, y, detections))
        scene = Predictions()
        for detection in merge_tiles(tile_outputs, iou_threshold=MERGE_IOU):
            scene.append(detection.polygon, detection.class_id, detection.score)
        predictions[name] = scene
        if count % 20 == 0:
            print(f"{count}/{len(names)} scenes, {time.time() - started:.0f}s", flush=True)

    summaries = []
    for label, drop in (("all", False), ("difficult-dropped", True)):
        ground_truth = {name: read_gt(args.gt_root / f"{name}.txt", drop) for name in names}
        metrics = evaluate(predictions, ground_truth, len(CLASSES))
        summaries.append(
            {
                "gt": label,
                "scenes": len(names),
                "merge_iou": MERGE_IOU,
                "map": round(float(metrics["map"]), 4),
                "map_50": round(float(metrics["map_50"]), 4),
                "reference_mmrotate_map": REFERENCE_MAP,
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
