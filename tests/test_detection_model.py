"""Detector assembly: anchor/encode round-trip, one training step, inference path.

Skipped where torch/torchvision are absent (the control-plane environment has
neither); run it in a worker environment, e.g.

    workers/prithvi/.venv/bin/python -m pytest tests/test_detection_model.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import nn  # noqa: E402

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
    assert float(losses["loss_reg"]) > 0
