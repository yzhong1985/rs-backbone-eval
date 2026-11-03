"""Read-only replay adapter for completed LIM runs (T10 detection, then T9).

The visualisation generator must show the *same* predictions the score came
from, so this module reuses ``t10.score``'s building blocks instead of writing a
second inference path: the same tile reader, normalisation, detector call,
padding clip and rotated-NMS merge. It only adds what evaluation discards: the
per-detection TP/FP/FN status and a PNG of a window.

Nothing here writes to ``run.json``; callers pass an output directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from rs_backbone_eval.data.isaid_io import read_rgb
from rs_backbone_eval.detection.ap import _match_class
from rs_backbone_eval.detection.classes import FAIR1M_ID_TO_CLASS, class_id
from rs_backbone_eval.detection.geometry import poly2obb
from rs_backbone_eval.detection.nms import Detection, clip_polygon, merge_tiles

from .mit import mit_huge
from .t10 import (
    MIN_BOX_SIDE_PX,
    AnnotationCache,
    LimDetector,
    tile_geometry,
)

T10_CLASS_NMS_IOU = 0.5  # detector.py:505, a locked constant rather than a setting
T10_MERGE_NMS_IOU = 0.5  # t10.py:555, the scene-level merge


@dataclass
class ReplayContext:
    """Everything a replay needs, loaded once."""

    run_id: str
    task_id: str
    model_id: str
    protocol_id: str
    seed: int
    model: LimDetector
    detector: object
    device: str
    num_classes: int
    crop: int
    stride: int
    means: torch.Tensor
    stds: torch.Tensor
    keep_ratio: float
    checkpoint_path: Path
    checkpoint_kind: str
    checkpoint_sha256: str
    anchors: tuple[int, ...]
    ratios: tuple[float, ...]
    data_root: Path
    manifest_path: Path
    run_commit: str
    tags: dict

    def params(self) -> dict:
        """Inference parameters as actually used, for the manifest fingerprint."""
        return {
            "crop": int(self.crop),
            "stride": int(self.stride),
            "score_threshold": float(self.detector.score_threshold),
            "rpn_nms_iou": float(self.detector.nms_threshold),
            "class_nms_iou": T10_CLASS_NMS_IOU,
            "merge_nms_iou": T10_MERGE_NMS_IOU,
            "rpn_post_nms": int(self.detector.rpn_post_nms),
            "detections_per_image": int(self.detector.detections_per_image),
            "anchor_sizes": list(self.anchors),
            "anchor_ratios": [float(value) for value in self.ratios],
            "keep_ratio": float(self.keep_ratio),
            "min_box_side_px": float(MIN_BOX_SIDE_PX),
            "background_class": None,
            "ignore_values": [],
            "merge": "rotated-nms",
            "box_convention": "le90",
        }


@dataclass
class SceneMatch:
    detections: list[Detection]
    statuses: list[str]  # tp / fp, aligned with detections
    targets: list[tuple[np.ndarray, int]]  # polygon, class id
    matched_targets: set[int]
    counts: dict[str, int]

    @property
    def fn_indices(self) -> list[int]:
        return [i for i in range(len(self.targets)) if i not in self.matched_targets]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_pairs(tag: str, key: str) -> tuple[float, ...]:
    """Read the ``normalization`` tag: ``train-tile mean [..] std [..]``."""
    match = re.search(rf"{key}\s*(\[[^\]]*\])", tag)
    if not match:
        raise ValueError(f"run tag normalization has no {key} list: {tag!r}")
    return tuple(float(value) for value in ast.literal_eval(match.group(1)))


def _checkpoint_kind(run_dir: Path) -> tuple[Path, str]:
    for name, kind in (
        ("best_checkpoint.pt", "best"),
        ("head_only.pt", "head_only"),
        ("latest.pt", "latest"),
    ):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate, kind
    raise FileNotFoundError(f"{run_dir} has no best_checkpoint.pt / head_only.pt / latest.pt")


def _load_state(context: ReplayContext, payload: dict) -> dict[str, torch.Tensor]:
    if context.checkpoint_kind in {"best", "latest"}:
        state = payload.get("state_dict") or payload.get("model")
        if state is None:
            raise ValueError(f"{context.checkpoint_path.name} has no state dict")
        return state
    # head_only keeps only the trainable part. The frozen backbone is rebuilt
    # from the run's recorded pretrained checkpoint, whose sha256 must match.
    identity = payload.get("identity") or {}
    expected = identity.get("pretrained_checkpoint")
    backbone = Path(
        context.tags.get("pretrained_checkpoint_path") or os.environ.get("LIM_CHECKPOINT", "")
    )
    if not expected or not backbone.is_file():
        raise ValueError(
            "head_only replay needs the frozen backbone: set LIM_CHECKPOINT to the "
            "MixMIM checkpoint recorded in the run"
        )
    if _sha256(backbone) != expected:
        raise ValueError(
            f"LIM_CHECKPOINT does not match the run's pretrained sha256 ({expected[:12]}…)"
        )
    from .mit import load_encoder

    # Load into the encoder the model actually uses. Building a throwaway
    # ``mit_huge()`` here left the real encoder randomly initialised, so the
    # subsequent strict state-dict load failed with missing encoder keys and a
    # pruned run (latest.pt deleted, head_only.pt kept) could never be replayed
    # (review 2026-11-03).
    stats = load_encoder(backbone, context.model.encoder)
    if stats["missing"] or stats["unexpected"]:
        raise ValueError(f"encoder load mismatch: {stats}")
    return payload["head_state_dict"]


def load_for_eval(run_dir: Path, device: str | None = None) -> ReplayContext:
    """Build the model a completed run was scored with."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    if record["task_id"] != "T10" or record["protocol_id"] != "rotated_frozen":
        raise ValueError(
            f"{record['run_id']}: replay supports T10 rotated_frozen, "
            f"got {record['task_id']}/{record['protocol_id']}"
        )
    if record["status"] != "completed":
        raise ValueError(f"{record['run_id']}: run is {record['status']}, not completed")
    tags = record.get("tags") or {}
    task_config = yaml.safe_load(Path(request["task_config_path"]).read_text(encoding="utf-8"))
    num_classes = int(task_config["num_outputs"])

    data_root = Path(request["data_root"])
    manifest_path = Path(request["manifest_path"])
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    crop, stride = tile_geometry(train_rows)

    anchors = tuple(int(value) for value in tags["anchor_sizes"].split(","))
    ratios = tuple(float(value) for value in tags["anchor_ratios"].split(","))
    keep_ratio = float(tags.get("keep_ratio", 0.7))
    roi_positive = float(tags.get("roi_positive_fraction", 0.0)) or None

    normalisation = tags["normalization"]
    means_np = np.asarray(_parse_pairs(normalisation, "mean"), dtype=np.float32)
    stds_np = np.asarray(_parse_pairs(normalisation, "std"), dtype=np.float32)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = mit_huge(in_chans=3, drop_path_rate=0.0)
    model = LimDetector(
        encoder,
        num_classes,
        fpn_dim=int(tags.get("fpn_dim", 256)),
        anchor_sizes=anchors,
        anchor_ratios=ratios,
        roi_positive_fraction=roi_positive,
    ).to(device)
    detector = model.detector
    for parameter in detector.parameters():
        parameter.requires_grad = False

    checkpoint_path, kind = _checkpoint_kind(run_dir)
    context = ReplayContext(
        run_id=record["run_id"],
        task_id=record["task_id"],
        model_id=record["model_id"],
        protocol_id=record["protocol_id"],
        seed=int(record["seed"]),
        model=model,
        detector=detector,
        device=device,
        num_classes=num_classes,
        crop=crop,
        stride=stride,
        means=torch.tensor(means_np, device=device).view(1, 3, 1, 1),
        stds=torch.tensor(stds_np, device=device).view(1, 3, 1, 1),
        keep_ratio=keep_ratio,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=kind,
        checkpoint_sha256=_sha256(checkpoint_path),
        anchors=anchors,
        ratios=ratios,
        data_root=data_root,
        manifest_path=manifest_path,
        run_commit=str(record.get("git_commit") or ""),
        tags=tags,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = _load_state(context, payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    check_loaded_keys(missing, unexpected, kind)
    model.eval()
    return context


#: Prefix of the frozen encoder inside ``LimDetector.state_dict()``.
BACKBONE_PREFIX = "detector.backbone."


def check_loaded_keys(missing: list[str], unexpected: list[str], kind: str) -> None:
    """Strict key check after ``load_state_dict(strict=False)``.

    ``head_only.pt`` carries no encoder: ``_load_state`` has already loaded the
    hash-checked pretrained encoder into ``model.encoder``, so the encoder keys
    are *expected* to be missing from the head state, and only those. A missing
    head key, any unexpected key, or a missing encoder key for a full checkpoint
    is still an error (shape mismatches raise inside ``load_state_dict``).
    Treating every missing key as an error made head_only replay impossible
    (review 1b3ceae, 2026-11-03).
    """
    if kind == "head_only":
        missing = [key for key in missing if not key.startswith(BACKBONE_PREFIX)]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")


def predict_scene(
    context: ReplayContext,
    rows: list[dict],
    annotations: AnnotationCache | None = None,
) -> list[Detection]:
    """Whole-scene inference, mirroring ``t10.score`` (tile -> clip -> merge)."""
    annotations = annotations or AnnotationCache(context.data_root, context.keep_ratio)
    ordered = sorted(rows, key=lambda row: (row["window"][1], row["window"][0]))
    tile_outputs: list[tuple[int, int, list[Detection]]] = []
    with torch.inference_mode():
        for row in ordered:
            window = [int(value) for value in row["window"]]
            payload = annotations.get(row["annotation_path"])
            valid_w = min(context.crop, int(payload["width"]) - window[0])
            valid_h = min(context.crop, int(payload["height"]) - window[1])
            pixels = read_rgb(context.data_root / row["image_paths"][0], tuple(window)).astype(
                np.float32
            )
            tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            tensor = (tensor[None].to(context.device) - context.means) / context.stds
            output = context.detector.detections(
                context.detector.pyramid(tensor), (context.crop, context.crop)
            )[0]
            detections: list[Detection] = []
            for polygon, value, label in zip(
                output["polygons"].cpu().numpy(),
                output["scores"].cpu().tolist(),
                output["labels"].cpu().tolist(),
                strict=True,
            ):
                clipped = clip_polygon(polygon, valid_w, valid_h)
                if clipped is None:
                    continue
                detections.append(
                    Detection(polygon=clipped, score=float(value), class_id=int(label))
                )
            tile_outputs.append((window[0], window[1], detections))
    return merge_tiles(tile_outputs, iou_threshold=T10_MERGE_NMS_IOU)


def scene_targets(
    context: ReplayContext,
    row: dict,
    annotations: AnnotationCache | None = None,
) -> list[tuple[np.ndarray, int]]:
    annotations = annotations or AnnotationCache(context.data_root, context.keep_ratio)
    targets: list[tuple[np.ndarray, int]] = []
    for instance in annotations.instances(row["annotation_path"]):
        corners = np.asarray(instance.polygon, dtype=np.float64)
        _, _, box_w, box_h, _ = poly2obb(corners)
        if min(box_w, box_h) < MIN_BOX_SIDE_PX:
            continue
        targets.append((corners, class_id(instance.class_name)))
    return targets


def match_scene(
    detections: list[Detection],
    targets: list[tuple[np.ndarray, int]],
    iou_threshold: float = 0.5,
) -> SceneMatch:
    """Class-aware greedy matching, identical to ``ap._match_class``."""
    by_class: dict[int, list[int]] = {}
    for index, detection in enumerate(detections):
        by_class.setdefault(detection.class_id, []).append(index)
    targets_by_class: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(targets):
        targets_by_class.setdefault(label, []).append(index)

    statuses = ["fp"] * len(detections)
    matched: set[int] = set()
    for label, detection_indices in by_class.items():
        target_indices = targets_by_class.get(label, [])
        detections_for_class = [
            (detections[index].score, detections[index].polygon) for index in detection_indices
        ]
        targets_for_class = [(targets[index][0], False) for index in target_indices]
        tp, matched_local = _match_class(detections_for_class, targets_for_class, iou_threshold)
        for position, is_true_positive in enumerate(tp):
            if is_true_positive:
                statuses[detection_indices[position]] = "tp"
        for target_position in matched_local:
            if target_position >= 0:
                matched.add(target_indices[int(target_position)])

    tp_count = statuses.count("tp")
    counts = {
        "tp": tp_count,
        "fp": len(detections) - tp_count,
        "fn": len(targets) - len(matched),
        "gt": len(targets),
        "pred": len(detections),
    }
    return SceneMatch(
        detections=detections,
        statuses=statuses,
        targets=targets,
        matched_targets=matched,
        counts=counts,
    )


# --------------------------------------------------------------------- drawing


def _font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:  # pragma: no cover - depends on the host fonts
        return ImageFont.load_default(size=size)


def _points(polygon: np.ndarray, offset: tuple[int, int]) -> list[tuple[float, float]]:
    dx, dy = offset
    return [(float(x) - dx, float(y) - dy) for x, y in polygon]


def _draw_dashed(draw: ImageDraw.ImageDraw, points, colour, width: int, dash: int = 8) -> None:
    closed = list(points)
    if closed and closed[0] != closed[-1]:
        closed.append(closed[0])
    for start, end in zip(closed, closed[1:], strict=False):
        x0, y0 = start
        x1, y1 = end
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length <= 0:
            continue
        steps = max(1, int(length // dash))
        for step in range(steps):
            if step % 2:
                continue
            t0 = step / steps
            t1 = min(1.0, (step + 1) / steps)
            draw.line(
                [
                    (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                    (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
                ],
                fill=colour,
                width=width,
            )


#: Height of the caption strip above each panel, in pixels.
PANEL_HEADER_PX = 30


def _with_header(layer: Image.Image, caption: str) -> Image.Image:
    """Add the caption strip *after* drawing.

    Boxes are drawn on ``layer`` in window pixel coordinates; adding the strip
    first and drawing on the taller canvas shifted every box, label and FN
    outline up by the strip height (review 1b3ceae, 2026-11-03).
    """
    canvas = Image.new("RGB", (layer.width, layer.height + PANEL_HEADER_PX), (12, 22, 20))
    canvas.paste(layer, (0, PANEL_HEADER_PX))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), caption, fill=(236, 245, 240), font=_font(18))
    return canvas


def _overlaps_window(polygon: np.ndarray, window: tuple[int, int, int, int]) -> bool:
    x, y, width, height = window
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    return bool(
        xs.max() >= x and xs.min() <= x + width and ys.max() >= y and ys.min() <= y + height
    )


def choose_focus(
    match: SceneMatch,
    window: tuple[int, int, int, int],
    digest: str,
    *,
    display_score: float,
) -> dict | None:
    """Pick one object to highlight: a correct box, a false alarm or a miss.

    ``FP=FN=0`` is unreachable at the protocol threshold for this model (0 of
    1732 test scenes at 0.05, 31 at 0.50), so the bucket stays scene-level and
    every sample names the object the reader should look at. Alternating TP/FP by the
    digest stops the gallery from showing only false alarms.
    """
    inside_tp = [
        (index, detection)
        for index, (detection, status) in enumerate(
            zip(match.detections, match.statuses, strict=True)
        )
        if status == "tp"
        and detection.score >= display_score
        and _overlaps_window(detection.polygon, window)
    ]
    inside_fp = [
        (index, detection)
        for index, (detection, status) in enumerate(
            zip(match.detections, match.statuses, strict=True)
        )
        if status == "fp"
        and detection.score >= display_score
        and _overlaps_window(detection.polygon, window)
    ]
    inside_fn = [
        index for index in match.fn_indices if _overlaps_window(match.targets[index][0], window)
    ]
    if not (inside_tp or inside_fp or inside_fn):
        return None
    if match.counts["tp"] == 0 and inside_fn:
        kind = "fn"
    elif inside_tp and inside_fp:
        kind = "tp" if int(digest[:8], 16) % 2 == 0 else "fp"
    elif inside_tp:
        kind = "tp"
    elif inside_fp:
        kind = "fp"
    else:
        kind = "fn"

    if kind == "fn":
        index = max(
            inside_fn,
            key=lambda i: float(
                np.ptp(match.targets[i][0][:, 0]) * np.ptp(match.targets[i][0][:, 1])
            ),
        )
        label = match.targets[index][1]
        return {
            "kind": "fn",
            "index": int(index),
            "class": FAIR1M_ID_TO_CLASS.get(label, str(label)),
            "score": None,
        }
    pool = inside_tp if kind == "tp" else inside_fp
    index, detection = max(pool, key=lambda item: (item[1].score, -item[0]))
    return {
        "kind": kind,
        "index": int(index),
        "class": FAIR1M_ID_TO_CLASS.get(detection.class_id, str(detection.class_id)),
        "score": round(float(detection.score), 4),
    }


def render_window(
    context: ReplayContext,
    row: dict,
    match: SceneMatch,
    output: Path,
    *,
    display_score: float = 0.3,
    focus: dict | None = None,
    valid_size: tuple[int, int] | None = None,
) -> dict:
    """Two-panel PNG: GT boxes, then predictions coloured by TP/FP/FN.

    The panel is cropped for display only; statuses come from the whole-scene
    match, so a detection whose box leaves the window keeps its real status.
    Only boxes at or above ``display_score`` are drawn (the card still reports
    the protocol threshold); the focused object is always drawn.
    """
    window = tuple(int(value) for value in row["window"])
    pixels = read_rgb(context.data_root / row["image_paths"][0], window)
    if valid_size is not None:
        # FAIR1M images are smaller than the 1024 tile; the rest is protocol
        # padding. Crop the figure to the real pixels so the panel is not 40%
        # black (boxes keep their tile coordinates, offset removes them).
        pixels = pixels[: valid_size[1], : valid_size[0]]
    offset = (window[0], window[1])
    left, right = pixels.shape[1], pixels.shape[0]

    gt_layer = Image.fromarray(pixels.copy())
    gt_draw = ImageDraw.Draw(gt_layer)
    for polygon, label in match.targets:
        points = _points(polygon, offset)
        gt_draw.polygon(points, outline=(61, 220, 151), width=3)
        gt_draw.text(
            (points[0][0] + 3, points[0][1] + 3),
            FAIR1M_ID_TO_CLASS.get(label, str(label)),
            fill=(61, 220, 151),
            font=_font(14),
        )

    pred_layer = Image.fromarray(pixels.copy())
    pred_draw = ImageDraw.Draw(pred_layer)
    for polygon, _ in match.targets:
        pred_draw.polygon(_points(polygon, offset), outline=(110, 120, 116), width=1)
    colours = {"tp": (61, 220, 151), "fp": (255, 107, 107)}
    focus_index = focus["index"] if focus and focus["kind"] in {"tp", "fp"} else None
    visible = [
        (index, detection)
        for index, (detection, status) in enumerate(
            zip(match.detections, match.statuses, strict=True)
        )
        if (detection.score >= display_score or index == focus_index)
        and min(detection.polygon[:, 0]) <= window[0] + left
        and max(detection.polygon[:, 0]) >= window[0]
        and min(detection.polygon[:, 1]) <= window[1] + right
        and max(detection.polygon[:, 1]) >= window[1]
    ]
    # 170 boxes on one tile make every label overlap; keep the focus and the
    # five highest-scoring boxes labelled, draw the rest as outlines.
    labelled = {index for index, _ in sorted(visible, key=lambda item: -item[1].score)[:5]}
    if focus_index is not None:
        labelled.add(focus_index)
    for index, detection in visible:
        points = _points(detection.polygon, offset)
        status = match.statuses[index]
        pred_draw.polygon(points, outline=colours[status], width=3)
        if index == focus_index:
            pred_draw.polygon(points, outline=(255, 255, 255), width=7)
        if index not in labelled:
            continue
        name = FAIR1M_ID_TO_CLASS.get(detection.class_id, str(detection.class_id))
        pred_draw.text(
            (points[0][0] + 3, points[0][1] + 3),
            f"{name} {detection.score:.2f}",
            fill=colours[status],
            font=_font(14),
        )
    focus_fn = focus["index"] if focus and focus["kind"] == "fn" else None
    for index in match.fn_indices:
        polygon, label = match.targets[index]
        _draw_dashed(pred_draw, _points(polygon, offset), (255, 180, 84), 3)
        if index == focus_fn:
            pred_draw.polygon(_points(polygon, offset), outline=(255, 255, 255), width=5)
        pred_draw.text(
            (_points(polygon, offset)[0][0] + 3, _points(polygon, offset)[0][1] + 3),
            f"FN {FAIR1M_ID_TO_CLASS.get(label, str(label))}",
            fill=(255, 180, 84),
            font=_font(14),
        )

    gt_panel = _with_header(gt_layer, "GT")
    pred_panel = _with_header(pred_layer, "PRED  green=TP  red=FP  yellow-dashed=FN")
    combined = Image.new("RGB", (gt_panel.width, gt_panel.height + pred_panel.height), (12, 22, 20))
    combined.paste(gt_panel, (0, 0))
    combined.paste(pred_panel, (0, gt_panel.height))
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output, format="PNG", optimize=True)

    # How many of the scene's boxes live outside this window, so the figure does
    # not look better/worse than the card next to it.
    outside = {"tp": 0, "fp": 0, "fn": 0}
    for detection, status in zip(match.detections, match.statuses, strict=True):
        if detection.score < display_score:
            continue
        points = _points(detection.polygon, offset)
        inside = (
            max(x for x, _ in points) >= 0
            and min(x for x, _ in points) <= left
            and max(y for _, y in points) >= 0
            and min(y for _, y in points) <= right
        )
        if not inside and status in outside:
            outside[status] += 1
    for index in match.fn_indices:
        points = _points(match.targets[index][0], offset)
        inside = (
            max(x for x, _ in points) >= 0
            and min(x for x, _ in points) <= left
            and max(y for _, y in points) >= 0
            and min(y for _, y in points) <= right
        )
        if not inside:
            outside["fn"] += 1
    return outside


def class_names() -> dict[int, str]:
    return dict(FAIR1M_ID_TO_CLASS)
