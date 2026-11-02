"""Detector assembly: anchor/encode round-trip, one training step, inference path.

Skipped where torch/torchvision are absent (the control-plane environment has
neither); run it in a worker environment, e.g.

    workers/prithvi/.venv/bin/python -m pytest tests/test_detection_model.py -q
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn  # noqa: E402
from torchvision.ops import box_iou  # noqa: E402

from rs_backbone_eval.detection.anchors import (  # noqa: E402
    anchors_for_level,
    decode,
    encode,
    hull,
    polygons,
)
from rs_backbone_eval.detection.detector import GroundTruth, RotatedFasterRCNN  # noqa: E402


class TinyBackbone(nn.Module):
    """Four levels at /4, /8, /16, /32 from one 128 px input."""

    def __init__(self, channels=(16, 32, 64, 128)) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels[0], 3, stride=4, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(channels[index], channels[index + 1], 3, stride=2, padding=1)
                for index in range(len(channels) - 1)
            ]
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(images)]
        for block in self.blocks:
            features.append(block(features[-1]))
        return features


def test_anchor_encode_decode_round_trip() -> None:
    anchors = anchors_for_level(8, 8, 4, sizes=(32,), ratios=(1.0,))
    box = torch.tensor([[40.0, 40.0, 20.0, 10.0, 0.3]])
    matched_anchor = anchors[:1]
    deltas = encode(matched_anchor, box)
    recovered = decode(matched_anchor, deltas)
    assert torch.allclose(recovered, box, atol=1e-4)


def test_hull_and_polygons_agree_on_a_rotated_box() -> None:
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.7853982]])
    corner = polygons(box).abs().max().item()
    assert corner == pytest.approx(7.0710678, rel=1e-5)
    bounds = hull(box)[0]
    assert bounds[2] == pytest.approx(7.0710678, rel=1e-5)
    assert bounds[0] == pytest.approx(-7.0710678, rel=1e-5)


def test_one_training_step_produces_finite_losses_and_gradients() -> None:
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    # freezing is the worker's job (prepare_encoder), not the detector's: the
    # detector only guarantees it never adds trainable backbone parameters
    for parameter in detector.backbone.parameters():
        parameter.requires_grad = False
    images = torch.randn(1, 3, 128, 128)
    targets = [
        GroundTruth(
            boxes=torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.0], [90.0, 70.0, 24.0, 12.0, 0.5]]),
            labels=torch.tensor([0, 2]),
        )
    ]
    losses = detector.losses(detector.pyramid(images), targets)
    total = losses["loss"]
    assert torch.isfinite(total), losses
    total.backward()
    graded = [
        name
        for name, parameter in detector.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    ]
    assert any(name.startswith("rpn_head") for name in graded)
    assert any(name.startswith("cls_score") for name in graded)
    # the backbone is registered for checkpointing but frozen: no gradient flows
    # into it, and the trainable count is the detector's alone
    backbone_grads = [
        name
        for name, parameter in detector.backbone.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    ]
    assert backbone_grads == []
    trainable = [
        parameter for parameter in detector.backbone.parameters() if parameter.requires_grad
    ]
    assert trainable == []


def test_detections_return_polygons_and_scores() -> None:
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    detector.eval()
    with torch.no_grad():
        output = detector.detections(detector.pyramid(torch.randn(1, 3, 128, 128)), (128, 128))
    assert len(output) == 1
    assert output[0]["polygons"].shape[-2:] == (4, 2)
    if output[0]["scores"].numel():
        assert float(output[0]["scores"].max()) <= 1.0
        assert int(output[0]["labels"].max()) < 3


def test_batched_forward_is_refused_with_a_reason() -> None:
    detector = RotatedFasterRCNN(TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32)
    with pytest.raises(ValueError, match="one image per forward"):
        detector.losses(detector.pyramid(torch.randn(2, 3, 128, 128)), [])


def test_ignored_objects_absorb_proposals_instead_of_teaching_background() -> None:
    targets = GroundTruth(
        boxes=torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.0]]),
        labels=torch.tensor([0]),
        ignored_boxes=torch.tensor([[90.0, 90.0, 20.0, 20.0, 0.0]]),
    )
    from rs_backbone_eval.detection.anchors import hull

    proposals = torch.tensor([[90.0, 90.0, 10.0, 10.0, 0.0], [10.0, 10.0, 10.0, 10.0, 0.0]])
    # the ignored object is 20x20 and the first proposal 10x10 at the same
    # centre: IoU 0.25, enough to be dropped at the detector's 0.2 default
    mask = targets.ignore_mask(hull(proposals), threshold=0.2)
    assert mask.tolist() == [True, False]


def test_training_with_ignored_objects_stays_finite() -> None:
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    for parameter in detector.backbone.parameters():
        parameter.requires_grad = False
    targets = [
        GroundTruth(
            boxes=torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.0]]),
            labels=torch.tensor([0]),
            ignored_boxes=torch.tensor([[90.0, 70.0, 24.0, 12.0, 0.5]]),
        )
    ]
    losses = detector.losses(detector.pyramid(torch.randn(1, 3, 128, 128)), targets)
    assert torch.isfinite(losses["loss"])


def test_padded_band_anchors_are_not_sampled_as_background() -> None:
    """43% of FAIR1M tiles are less than half real pixels; the zeros are not negatives."""
    ground_truth = GroundTruth(
        boxes=torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.0]]),
        labels=torch.tensor([0]),
        valid_region=(0.0, 0.0, 64.0, 128.0),  # left half only
    )
    hulls = torch.tensor([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 120.0, 120.0]])
    assert ground_truth.valid_mask(hulls).tolist() == [True, False]

    # the default has no valid region and keeps everything usable
    plain = GroundTruth(boxes=torch.zeros(0, 5), labels=torch.zeros(0, dtype=torch.long))
    assert plain.valid_mask(hulls).tolist() == [True, True]


def test_training_inside_a_padded_tile_stays_finite() -> None:
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    for parameter in detector.backbone.parameters():
        parameter.requires_grad = False
    targets = [
        GroundTruth(
            boxes=torch.tensor([[30.0, 30.0, 16.0, 8.0, 0.0]]),
            labels=torch.tensor([1]),
            valid_region=(0.0, 0.0, 64.0, 128.0),
        )
    ]
    losses = detector.losses(detector.pyramid(torch.randn(1, 3, 128, 128)), targets)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_encoding_against_its_own_anchor_is_zero() -> None:
    """`encode` regresses from an x1,y1,x2,y2 anchor to a rotated box.

    For an axis-aligned box the hull *is* the box, so the deltas vanish. Passing
    the rotated box itself as the anchor (the bug fixed on 2026-10-29) would make
    the implied width `w - cx`, i.e. negative whenever cx > w.
    """
    box = torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.0]])
    assert encode(hull(box), box).abs().max() < 1e-5
    # a rotated box has a larger hull, so its own encoding is not zero: the
    # anchor and the target are different rectangles
    rotated = torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.3]])
    assert encode(hull(rotated), rotated).abs().max() > 1e-3


def test_empty_and_ignored_only_tiles_train_the_rpn_on_negatives() -> None:
    """11.7% of the FAIR1M manifest has no object; it must not crash or be skipped."""
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    cases = {
        "empty": GroundTruth(boxes=torch.zeros(0, 5), labels=torch.zeros(0, dtype=torch.long)),
        "ignored-only": GroundTruth(
            boxes=torch.zeros(0, 5),
            labels=torch.zeros(0, dtype=torch.long),
            ignored_boxes=torch.tensor([[60.0, 60.0, 20.0, 10.0, 0.0]]),
            valid_region=(0.0, 0.0, 128.0, 128.0),
        ),
    }
    for name, target in cases.items():
        losses = detector.losses(detector.pyramid(torch.randn(1, 3, 128, 128)), [target])
        assert torch.isfinite(losses["loss"]), name
        # negatives still teach the RPN something; there is nothing for the RoI head
        assert float(losses["loss_rpn_cls"].detach()) > 0, name
        assert float(losses["loss_cls"]) == 0, name
        losses["loss"].backward()
        detector.zero_grad(set_to_none=True)


def test_ground_truth_in_the_proposals_gives_a_finite_regression_loss() -> None:
    """Regression: the RoI ridge used the rotated box where an anchor was expected.

    With `cx > w` (almost always) the "anchor width" went negative, so `log`
    produced NaN once the ground truth joined the proposals.
    """
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    target = GroundTruth(
        boxes=torch.tensor([[40.0, 40.0, 16.0, 8.0, 0.0], [90.0, 90.0, 24.0, 12.0, 0.5]]),
        labels=torch.tensor([0, 2]),
        valid_region=(0.0, 0.0, 128.0, 128.0),
    )
    losses = detector.losses(detector.pyramid(torch.randn(1, 3, 128, 128)), [target])
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["loss_reg"])
    assert float(losses["loss_reg"].detach()) > 0


def test_encode_decode_recovers_the_ground_truth_from_a_proposal() -> None:
    """Inference must decode from the same anchor representation training used.

    Regression 2026-10-29: training regressed `encode(hull(proposal), gt)` but
    inference called `decode(proposal, ...)` with cx, cy, w, h where x1, y1, x2,
    y2 was expected, so the "anchor width" was `w - cx` -- negative whenever
    cx > w, which is almost always. Every predicted box came out with negative
    width and the test mAP would have been ~0.
    """
    proposal = torch.tensor([[100.0, 100.0, 62.0, 20.0, 0.0]])  # cx, cy, w, h, theta
    ground_truth = torch.tensor([[305.0, 398.0, 64.0, 22.0, 0.35]])

    deltas = encode(hull(proposal), ground_truth)
    recovered = decode(hull(proposal), deltas)
    assert torch.allclose(recovered, ground_truth, atol=1e-3)

    # and the misuse is visibly wrong, so the shape of the bug stays documented
    wrong = decode(proposal, deltas)
    assert float(wrong[0, 2]) < 0  # negative width


def test_inference_returns_positive_boxes_inside_the_image() -> None:
    torch.manual_seed(0)
    detector = RotatedFasterRCNN(
        TinyBackbone(), [16, 32, 64, 128], num_classes=3, fpn_dim=32, rpn_post_nms=50
    )
    detector.eval()
    with torch.no_grad():
        output = detector.detections(detector.pyramid(torch.randn(1, 3, 128, 128)), (128, 128))[0]
    polygons = output["polygons"]
    if polygons.numel():
        extent = polygons.max(dim=1).values - polygons.min(dim=1).values
        assert (extent > 0).all(), "a predicted box has zero or negative extent"
        # decode clamps the centre to the image; the worker then clips the
        # polygon (see clip_polygon) so nothing outside the scene reaches the AP
        centres = polygons.mean(dim=1)
        assert float(centres.min()) >= -1.0
        assert float(centres.max()) <= 129.0


def test_chunked_iou_matches_the_full_matrix() -> None:
    """The RPN labels must not change when the anchor matrix is chunked.

    A [1.3 M anchors, N_gt] IoU matrix is multi-GB on a dense FAIR1M tile and
    OOM'd the 24 GB card; chunking is only acceptable if it is bit-identical to
    the full computation, including the argmax tie-break.
    """
    from rs_backbone_eval.detection import detector as module

    generator = torch.Generator().manual_seed(7)
    boxes = torch.cat(
        [
            torch.rand(90, 2, generator=generator) * 100,
            torch.rand(90, 2, generator=generator) * 20 + 4,
        ],
        dim=1,
    )
    # Two identical ground-truth rows force a tie across a chunk boundary.
    others = torch.cat(
        [
            torch.cat(
                [
                    torch.rand(5, 2, generator=generator) * 90,
                    torch.rand(5, 2, generator=generator) * 30 + 5,
                ],
                dim=1,
            ),
            boxes[3:4].clone(),
            boxes[3:4].clone(),
        ]
    )
    full = box_iou(boxes, others)

    rows, row_index = module._iou_rows(boxes, others)
    torch.testing.assert_close(rows, full.max(dim=1).values)
    assert torch.equal(row_index, full.max(dim=1).indices)

    columns, column_index = module._iou_columns(boxes, others)
    torch.testing.assert_close(columns, full.max(dim=0).values)
    assert torch.equal(column_index, full.max(dim=0).indices)

    # Chunk size must not matter: 90 boxes with a chunk of 7 crosses the
    # boundary several times.
    original = module.IOU_CHUNK
    module.IOU_CHUNK = 7
    try:
        chunked_rows, chunked_row_index = module._iou_rows(boxes, others)
        chunked_columns, chunked_column_index = module._iou_columns(boxes, others)
    finally:
        module.IOU_CHUNK = original
    torch.testing.assert_close(chunked_rows, rows)
    assert torch.equal(chunked_row_index, row_index)
    torch.testing.assert_close(chunked_columns, columns)
    assert torch.equal(chunked_column_index, column_index)


def test_ignored_objects_are_excluded_with_the_chunked_iou() -> None:
    from rs_backbone_eval.detection import detector as module

    generator = torch.Generator().manual_seed(11)
    hulls = torch.cat(
        [
            torch.rand(40, 2, generator=generator) * 100,
            torch.rand(40, 2, generator=generator) * 20 + 2,
        ],
        dim=1,
    )
    # An axis-aligned le90 box whose hull is exactly hulls[5]: IoU 1 with it.
    hulls[5] = torch.tensor([10.0, 10.0, 60.0, 40.0])
    target = GroundTruth(
        boxes=torch.tensor([[200.0, 200.0, 20.0, 10.0, 0.0]]),
        labels=torch.zeros(1, dtype=torch.long),
        ignored_boxes=torch.tensor([[35.0, 25.0, 50.0, 30.0, 0.0]]),
    )
    original = module.IOU_CHUNK
    module.IOU_CHUNK = 3
    try:
        mask = target.ignore_mask(hulls)
    finally:
        module.IOU_CHUNK = original
    assert bool(mask[5])
    assert int(mask.sum()) == 1


def test_encode_never_returns_an_infinite_target_for_a_degenerate_box() -> None:
    """A zero-width annotation made the RPN regression loss inf at step 8,552.

    115 of 356,140 FAIR1M instances are degenerate polygons. The loader drops
    them, but `encode` must not be able to produce +/-inf from one either.
    """
    from rs_backbone_eval.detection.anchors import encode

    anchors = torch.tensor([[0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 32.0, 32.0]])
    boxes = torch.tensor(
        [
            [16.0, 16.0, 0.0, 0.0, 0.0],  # duplicate vertices
            [16.0, 16.0, 12.0, 8.0, 0.3],  # a normal object
        ]
    )
    targets = encode(anchors, boxes)
    assert torch.isfinite(targets).all()
    assert float(targets[0, 2]) < -5.0  # clamped, not -inf
    torch.testing.assert_close(targets[1, 2], torch.log(torch.tensor(12.0 / 32.0)) * 5.0)


def test_roi_features_keep_the_callers_row_order() -> None:
    """Rows must not be regrouped by FPN level.

    Grouping them permuted features across boxes, so the RoI classifier was
    trained on another box's label: every T10 run reported mAP 0.0000 while a
    linear probe on the same features fits the tile perfectly (2026-10-31).
    """
    from rs_backbone_eval.detection.detector import ROI_FINEST_SCALE, RotatedFasterRCNN

    strides = (4, 8, 16, 32)

    class Dummy:
        pass

    dummy = Dummy()
    dummy.strides = strides
    # Each level is a constant map, so the pooled value identifies its level.
    features = [
        torch.full((1, 1, 1024 // stride, 1024 // stride), float(index + 1))
        for index, stride in enumerate(strides)
    ]
    # Areas deliberately interleave the levels (2, 0, 3, 1): grouping the rows
    # by level would return them as 1, 2, 3, 4 and mislabel every box.
    level_order = [2, 0, 3, 1]
    # level k covers [56*2^k, 56*2^(k+1)) px; 1.1 keeps the side inside the band
    areas = [(ROI_FINEST_SCALE * 2**k * 1.1) ** 2 for k in level_order]
    boxes = torch.tensor([[512.0, 512.0, math.sqrt(area), math.sqrt(area), 0.0] for area in areas])
    pooled = RotatedFasterRCNN._roi_features(dummy, features, boxes)
    assert pooled.shape[0] == boxes.shape[0]
    observed = [round(float(row.mean())) for row in pooled]
    assert observed == [k + 1 for k in level_order], observed


def test_aef_context_is_added_to_one_level_only() -> None:
    """The context head must not disturb the other pyramid levels."""
    from rs_backbone_eval.detection.aef_context import AefContext

    channels = (256, 256, 256, 256)  # FPN output width is fpn_dim on every level
    strides = (4, 8, 16, 32)
    features = [
        torch.zeros(1, width, 128 // stride, 128 // stride)
        for width, stride in zip(channels, strides, strict=True)
    ]
    context = torch.randn(1, 64, 125, 125)
    head = AefContext(channels=256, level=1)
    out = head(features, context)
    assert out[0] is features[0] and out[2] is features[2] and out[3] is features[3]
    assert out[1].shape == features[1].shape
    assert not torch.equal(out[1], features[1])


def test_shuffled_context_keeps_values_and_loses_layout() -> None:
    from rs_backbone_eval.detection.aef_context import shuffle_within_tile

    context = torch.arange(2 * 64 * 4 * 4, dtype=torch.float32).reshape(2, 64, 4, 4)
    shuffled = shuffle_within_tile(context, torch.Generator().manual_seed(0))
    assert torch.equal(
        shuffled.flatten(2).sort(dim=2).values, context.flatten(2).sort(dim=2).values
    )
    assert not torch.equal(shuffled, context)


def test_anchor_rows_follow_the_rpn_flatten_order() -> None:
    # ``_rpn_outputs`` reshapes [B, A, H, W] to [B, H*W*A], so row
    # (h*W + w)*A + a must be the a-th anchor of cell (h, w). Anchor-major rows
    # paired every score with another cell (review 2026-11-02).
    anchors = anchors_for_level(1, 2, 4, sizes=(4, 8), ratios=(1.0,))
    assert anchors.shape == (4, 4)
    # (h, w, a): (0,0,0) (0,0,1) (0,1,0) (0,1,1) at centres 2 and 6
    expected = [(2.0, 4.0), (2.0, 8.0), (6.0, 4.0), (6.0, 8.0)]
    for row, (centre_x, side) in enumerate(expected):
        x1, y1, x2, y2 = anchors[row].tolist()
        assert (x1 + x2) / 2 == pytest.approx(centre_x)
        assert (y1 + y2) / 2 == pytest.approx(2.0)
        assert x2 - x1 == pytest.approx(side)


def test_class_detections_keep_scores_with_their_boxes() -> None:
    from rs_backbone_eval.detection.detector import class_detections

    boxes = torch.tensor(
        [
            [20.0, 20.0, 8.0, 8.0, 0.0],
            [60.0, 60.0, 8.0, 8.0, 0.0],
            [100.0, 100.0, 8.0, 8.0, 0.0],
        ]
    )
    # Thresholding keeps proposals 1 and 2, a non-prefix subset; NMS then
    # returns local indices 0 and 1. The old code looked those up in the full
    # score array and reported [0.1, 0.9] for boxes 1 and 2.
    scores = torch.tensor([0.1, 0.9, 0.8])
    deltas = torch.zeros(3, 5)
    selected = class_detections(
        boxes,
        scores,
        deltas,
        score_threshold=0.5,
        nms_threshold=0.5,
        detections_per_image=100,
        width=128,
        height=128,
    )
    assert selected is not None
    polygons_out, kept_scores = selected
    assert kept_scores.tolist() == pytest.approx([0.9, 0.8])
    centres_x = polygons_out[:, :, 0].mean(dim=1).tolist()
    assert centres_x == pytest.approx([60.0, 100.0])


def test_roi_levels_follow_the_reference_mapping() -> None:
    # MMDetection SingleRoIExtractor default finest_scale=56: level 0 below
    # 112 px, level 2 at 224 px. The old formula read a 20 px FAIR1M object on
    # /16 instead of /4 (review 2026-11-02).
    from rs_backbone_eval.detection.detector import roi_levels

    sides = [20.0, 100.0, 224.0, 448.0, 1024.0]
    boxes = torch.tensor([[0.0, 0.0, side, side, 0.0] for side in sides])
    assert roi_levels(boxes, 4).tolist() == [0, 0, 2, 3, 3]


def test_rpn_sampling_caps_the_batch_at_rpn_batch() -> None:
    from rs_backbone_eval.detection.detector import RPN_BATCH, RPN_POSITIVE_FRACTION, sample_rpn

    # 400 positives, 500 negatives: the old formula capped num_pos by
    # max(RPN_BATCH - negative, 0) = 0, so positives escaped subsampling and
    # 400 + 256 = 656 anchors entered the loss (review 2026-11-02).
    labels = torch.zeros(900, dtype=torch.long)
    labels[:400] = 1
    sampled = sample_rpn(labels)
    assert sampled.numel() == RPN_BATCH
    assert int((labels[sampled] == 1).sum()) == int(RPN_BATCH * RPN_POSITIVE_FRACTION)

    # scarce negatives: positives keep their cap, negatives cannot fill the rest
    labels = torch.full((900,), -1, dtype=torch.long)
    labels[:400] = 1
    labels[400:410] = 0
    sampled = sample_rpn(labels)
    assert sampled.numel() == 128 + 10
    assert int((labels[sampled] == 1).sum()) == 128

    # plentiful negatives, few positives: batch filled by negatives
    labels = torch.zeros(900, dtype=torch.long)
    labels[:10] = 1
    sampled = sample_rpn(labels)
    assert sampled.numel() == RPN_BATCH
    assert int((labels[sampled] == 1).sum()) == 10


def test_class_detections_keeps_boxes_that_only_overlap_as_hulls() -> None:
    # Two 80x8 boxes at +45/-45 degrees, same centre: their axis-aligned hulls
    # are nearly the same square (hull IoU ~ 1), so the old horizontal NMS kept
    # only the first, while the rotated IoU is ~0.05 and both survive. The
    # whole-image merge cannot bring a suppressed box back (review 2026-11-02;
    # the earlier 0 vs pi/2 shape of this test also passed on the old code and
    # protected nothing).
    from rs_backbone_eval.detection.anchors import encode, hull
    from rs_backbone_eval.detection.detector import class_detections

    long_box = [80.0, 8.0]
    boxes = torch.tensor(
        [
            [200.0, 200.0, *long_box, math.pi / 4],
            [200.0, 200.0, *long_box, -math.pi / 4],
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    # class_detections decodes from the hull anchor, so zero deltas would fold
    # both boxes into the same square; encode the rotated targets instead
    deltas = encode(hull(boxes), boxes)
    selected = class_detections(
        boxes,
        scores,
        deltas,
        score_threshold=0.5,
        nms_threshold=0.5,
        detections_per_image=100,
        width=1024,
        height=1024,
    )
    assert selected is not None
    polygons_out, kept_scores = selected
    assert kept_scores.tolist() == pytest.approx([0.9, 0.8])
    assert polygons_out.shape[0] == 2


def test_class_detections_passes_numpy_polygons_to_the_nms(monkeypatch) -> None:
    # GPU inference crashed at exactly this edge: rows of polygons(decoded) are
    # tensors on the detector's device, and np.asarray refuses a non-CPU tensor
    # inside Tensor.__array__ (review 2026-11-02, third round). The spy pins
    # the contract without needing a GPU: the NMS always sees numpy arrays.
    import rs_backbone_eval.detection.detector as detector_module

    seen: list[str] = []
    real = detector_module.rotated_nms

    def spy(polygons, scores, iou_threshold):
        seen.extend(type(polygon).__name__ for polygon in polygons)
        return real(polygons, scores, iou_threshold)

    monkeypatch.setattr(detector_module, "rotated_nms", spy)
    boxes = torch.tensor([[100.0, 100.0, 40.0, 20.0, 0.3], [300.0, 300.0, 30.0, 30.0, 0.0]])
    scores = torch.tensor([0.9, 0.7])
    deltas = torch.zeros(2, 5)
    selected = detector_module.class_detections(
        boxes,
        scores,
        deltas,
        score_threshold=0.5,
        nms_threshold=0.5,
        detections_per_image=100,
        width=512,
        height=512,
    )
    assert seen and set(seen) == {"ndarray"}
    assert selected is not None
