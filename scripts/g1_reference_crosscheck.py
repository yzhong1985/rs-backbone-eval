"""G1 cross-check: score the reference detector with our own AP and merge.

The G1 anchor is the MMRotate R50 Rotated Faster R-CNN checkpoint
(``rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth``, 1x, le90) on the
DOTA-v1.0 validation split at 1024 px with gap 200 (stride 824). MMRotate's own
DOTAMetric reports **mAP 0.5454** there: patch level, IoU 0.5 only, VOC-07
11-point interpolation, difficult objects kept, no patch merging.

That number is *not* comparable with our formal COCO evaluator: besides the
interpolation, the matching rules differ. The reference takes the highest-IoU
ground truth over all of them and records a false positive when that one is
taken, without trying the second-best GT; the formal evaluator matches among
still-unmatched targets (2026-11-02: on the same input the reference reads
0.545 and the formal metric 1.0). So ``patch-ap`` reports both numbers: the
reference-matching one is what compares with 0.5454, the COCO one is context
only.

Stages:

``--stage inference``
    GPU. Runs the checkpoint over every split patch and dumps the predictions
    (patch coordinates) to JSON. No offset is applied here: ``merge_tiles``
    shifts tiles into image coordinates itself, and adding the origin here as
    well double-shifted every detection (review 2026-11-02).

``--stage patch-ap``
    CPU, reads the dump. Validates that predictions and annotations cover the
    same patch set with no duplicates — a missing annfile is an error, never an
    empty ground truth — and reports per-class GT counts plus the hashes of the
    dump and the annotations.

``--stage scene-ap``
    CPU, reads the dump. Merges each scene's patches with our ``merge_tiles``
    (rotated NMS at IoU 0.1, the reference merge setting) and scores against the
    full-image labelTxt. This checks the merge; its number is not expected to
    equal 0.5454, because split objects and cross-patch duplicates change the
    problem.

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
import hashlib
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
from rs_backbone_eval.detection.reference_ap import evaluate_reference  # noqa: E402

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
    """Targets from one DOTA labelTxt file (8 coords, category, difficult).

    A missing file is an error, not an empty ground truth: silently scoring a
    patch without its labels inflates AP (review 2026-11-02, supplementary
    round). An existing but empty file is a legitimate "no objects here".
    """
    if not path.is_file():
        raise SystemExit(f"missing annotation file: {path}")
    target = Targets()
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


def _pairs_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


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
    dump = load_dump(args.dump)
    patches = [entry["patch"] for entry in dump]
    duplicates = sorted({patch for patch in patches if patches.count(patch) > 1})
    if duplicates:
        raise SystemExit(f"duplicate patches in the dump: {duplicates[:5]}")
    annfiles = sorted((args.split / "annfiles").glob("*.txt"))
    declared = {path.stem for path in annfiles}
    if set(patches) != declared:
        missing = sorted(declared - set(patches))[:5]
        extra = sorted(set(patches) - declared)[:5]
        raise SystemExit(
            "predictions and annotations do not cover the same patches "
            f"(missing predictions {missing}, predictions without annotation {extra})"
        )

    predictions: dict[str, Predictions] = {}
    targets: dict[str, Targets] = {}
    for entry in dump:
        patch = entry["patch"]
        scene = Predictions()
        for polygon, score, label in zip(
            entry["polygons"], entry["scores"], entry["labels"], strict=True
        ):
            scene.append(np.asarray(polygon, dtype=np.float64), int(label), float(score))
        predictions[patch] = scene
        targets[patch] = read_dota_annotations(args.split / "annfiles" / f"{patch}.txt", False)

    per_class: dict[str, int] = {}
    for target in targets.values():
        for class_id in target.class_ids:
            name = CLASSES[class_id]
            per_class[name] = per_class.get(name, 0) + 1

    reference = evaluate_reference(predictions, targets, len(CLASSES), iou_threshold=0.5)
    formal = evaluate(predictions, targets, len(CLASSES), iou_thresholds=(0.5,))
    print(
        json.dumps(
            {
                "stage": "patch-ap",
                "patches": len(predictions),
                # the number to compare with the published reference
                "map50_reference_matching_11point": round(float(reference["map"]), 4),
                # the formal project protocol, for context only
                "map50_coco_matching_101point": round(float(formal["map_50"]), 4),
                "reference_mmrotate_map50": REFERENCE_MAP50,
                "classes_with_targets": reference["classes_with_targets"],
                "gt_per_class": dict(sorted(per_class.items())),
                "dump_sha256": hashlib.sha256(args.dump.read_bytes()).hexdigest(),
                "annfiles_sha256": _pairs_hash(annfiles),
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
    reference = evaluate_reference(predictions, targets, len(CLASSES), iou_threshold=0.5)
    formal = evaluate(predictions, targets, len(CLASSES), iou_thresholds=(0.5,))
    print(
        json.dumps(
            {
                "stage": "scene-ap",
                "scenes": len(names),
                "merge": f"merge_tiles rotated NMS at IoU {MERGE_IOU}",
                "map50_reference_matching_11point": round(float(reference["map"]), 4),
                "map50_coco_matching_101point": round(float(formal["map_50"]), 4),
                "reference_mmrotate_map50_patchlevel": REFERENCE_MAP50,
                "note": "whole-scene after merge; not comparable with the patch-level reference",
            }
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inference", "patch-ap", "scene-ap"), required=True)
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
