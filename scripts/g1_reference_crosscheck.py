"""G1 cross-check: score the reference detector with our own AP and merge.

The G1 anchor is the MMRotate R50 Rotated Faster R-CNN checkpoint
(``rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth``, 1x, le90) on the
DOTA-v1.0 validation split at 1024 px with gap 200 (stride 824). MMRotate's own
DOTAMetric reports **mAP 0.5454** there. That number is *patch-level* AP at a
single IoU threshold 0.5 with VOC-07 11-point interpolation, difficult objects
included, and no patch merging (``merge_patches=False``) — so it is only
comparable with our patch-level ``map_50``, not with the COCO-style ``map``
over 0.5:0.95, and not with whole-scene numbers.

Three stages, so each difference has one cause:

``--stage inference``
    GPU. Runs the checkpoint over every split patch and dumps the predictions
    (patch coordinates) to JSON. No offset is applied here: ``merge_tiles``
    shifts tiles into image coordinates itself, and adding the origin here as
    well double-shifted every detection and scored a perfect prediction as a
    miss (review 2026-11-02).

``--stage patch-ap``
    CPU, reads the dump. Scores each patch against its own annfile with our
    ``evaluate`` at IoU 0.5 only. The remaining difference to 0.5454 is the
    interpolation (COCO 101-point vs VOC-07 11-point), typically around one
    point.

``--stage scene-ap``
    CPU, reads the dump. Merges each scene's patches with our ``merge_tiles``
    (rotated NMS at IoU 0.1, the reference merge setting) and scores against
    the full-image labelTxt. This checks the merge; its number is not expected
    to equal 0.5454, because split objects and cross-patch duplicates change
    the problem.

Needs the dedicated MMRotate environment (torch 2.0.1+cu118, mmcv 2.0.1,
mmdet 3.1.0, mmrotate 1.0.0rc1); see
``docs/design/FAIR1M-2.0旋转目标检测评测方案`` §7.1.1. Example::

    cd ~/rs-eval/tools/mmrotate-src && \
    PYTHONPATH=.:/path/to/rs-backbone-eval/src \
    ~/rs-eval/tools/mmrotate-venv/bin/python scripts/g1_reference_crosscheck.py \
        --stage inference --dump /tmp/opencode/g1_predictions.json
    uv run python scripts/g1_reference_crosscheck.py --stage patch-ap \
        --dump /tmp/opencode/g1_predictions.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

# MMRotate's DOTA metric: patch-level, IoU 0.5, VOC-07 11-point, difficult kept.
REFERENCE_MAP50 = 0.5454
# The reference patch merge (DOTAMetric.merge_results) NMSes at IoU 0.1.
MERGE_IOU = 0.1


def read_dota_annotations(path: Path, drop_difficult: bool) -> Targets:
    """Targets from one DOTA labelTxt file (8 coords, category, difficult)."""
    target = Targets()
    if not path.is_file():
        return target
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        name, difficult = parts[8], int(float(parts[9]))
        if name not in INDEX:
            continue
        if drop_difficult and difficult == 1:
            continue
        coords = np.asarray([float(value) for value in parts[:8]], dtype=np.float64)
        target.append(coords.reshape(4, 2), INDEX[name])
    return target


def run_inference(args: argparse.Namespace) -> None:
    import torch
    from mmdet.apis import inference_detector, init_detector
    from mmrotate.utils import register_all_modules

    from rs_backbone_eval.detection.anchors import polygons

    register_all_modules()
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")
    model.eval()

    patches = sorted((args.split / "images").glob("*.png"))
    dump: list[dict] = []
    started = time.time()
    for count, image in enumerate(patches, start=1):
        result = inference_detector(model, str(image))
        instances = result.pred_instances.cpu()
        boxes = torch.from_numpy(np.asarray(instances.bboxes, dtype=np.float32)).reshape(-1, 5)
        entry = {
            "patch": image.stem,
            # patch coordinates on purpose: merge_tiles applies the tile origin
            "polygons": [],
            "scores": [float(value) for value in np.asarray(instances.scores)],
            "labels": [int(value) for value in np.asarray(instances.labels)],
        }
        if boxes.numel():
            entry["polygons"] = [
                polygon.tolist() for polygon in polygons(boxes).numpy().astype(float)
            ]
        dump.append(entry)
        if count % 500 == 0:
            print(f"{count}/{len(patches)} patches, {time.time() - started:.0f}s", flush=True)
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    args.dump.write_text(json.dumps(dump), encoding="utf-8")
    print(f"wrote {args.dump} ({len(dump)} patches)", flush=True)


def load_dump(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_patch_ap(args: argparse.Namespace) -> None:
    predictions: dict[str, Predictions] = {}
    targets: dict[str, Targets] = {}
    for entry in load_dump(args.dump):
        patch = entry["patch"]
        scene = Predictions()
        for polygon, score, label in zip(
            entry["polygons"], entry["scores"], entry["labels"], strict=True
        ):
            scene.append(np.asarray(polygon, dtype=np.float64), int(label), float(score))
        predictions[patch] = scene
        targets[patch] = read_dota_annotations(args.split / "annfiles" / f"{patch}.txt", False)
    for drop in (False, True):
        ground = {
            patch: read_dota_annotations(args.split / "annfiles" / f"{patch}.txt", drop)
            for patch in predictions
        }
        metrics = evaluate(predictions, ground, len(CLASSES), iou_thresholds=(0.5,))
        print(
            json.dumps(
                {
                    "stage": "patch-ap",
                    "difficult": "dropped" if drop else "kept",
                    "patches": len(predictions),
                    "map_50": round(float(metrics["map_50"]), 4),
                    "reference_mmrotate_map50_11point": REFERENCE_MAP50,
                    "note": "same unit (patch-level, IoU .5); only the interpolation differs",
                }
            ),
            flush=True,
        )


def run_scene_ap(args: argparse.Namespace) -> None:
    groups: dict[str, list[tuple[int, int, list[Detection]]]] = defaultdict(list)
    for entry in load_dump(args.dump):
        match = PATCH.match(entry["patch"])
        if match is None:
            raise SystemExit(f"cannot parse patch id {entry['patch']}")
        detections = [
            Detection(np.asarray(polygon, dtype=np.float64), float(score), int(label))
            for polygon, score, label in zip(
                entry["polygons"], entry["scores"], entry["labels"], strict=True
            )
        ]
        groups[match.group("name")].append(
            (int(match.group("x")), int(match.group("y")), detections)
        )
    names = sorted(groups)[: args.limit or None]
    predictions: dict[str, Predictions] = {}
    for name in names:
        scene = Predictions()
        for detection in merge_tiles(groups[name], iou_threshold=MERGE_IOU):
            scene.append(detection.polygon, detection.class_id, detection.score)
        predictions[name] = scene
    targets = {name: read_dota_annotations(args.gt_root / f"{name}.txt", False) for name in names}
    metrics = evaluate(predictions, targets, len(CLASSES), iou_thresholds=(0.5,))
    print(
        json.dumps(
            {
                "stage": "scene-ap",
                "scenes": len(names),
                "merge": f"merge_tiles rotated NMS at IoU {MERGE_IOU}",
                "map_50": round(float(metrics["map_50"]), 4),
                "reference_mmrotate_map50_patchlevel": REFERENCE_MAP50,
                "note": "whole-scene after merge; not comparable with the patch-level reference",
            }
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("inference", "patch-ap", "scene-ap"),
        required=True,
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("/geodata/rs-eval/data/raw/isaid/dota/split_val_1024_824"),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/geodata/rs-eval/data/raw/isaid/dota/val/labelTxt-v1.0/labelTxt"),
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
    parser.add_argument("--dump", type=Path, default=Path("/tmp/opencode/g1_predictions.json"))
    parser.add_argument("--limit", type=int, default=0, help="scene-ap: first N scenes only")
    args = parser.parse_args()

    if args.stage == "inference":
        run_inference(args)
    elif args.stage == "patch-ap":
        run_patch_ap(args)
    else:
        run_scene_ap(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
