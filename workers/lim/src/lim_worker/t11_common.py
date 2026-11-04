"""Shared T11 helpers: device selection and norm/input provenance tags.

Both LIM T11 variants (t11.py, t11_ltae.py) call these so the run record
always carries the same provenance: the device actually used (review
2026-11-04: workers must not silently train on CPU and must not hardcode a
host name in the notes), the runtime torch version and the true input
modality (review 2026-11-04: the manifest still lists sensor=aef / bands
AEF 64 because the AEF grid is the row payload; the LIM variants read the
aux S2 stack instead, and that must be visible in the tags).
"""

from __future__ import annotations

from rs_backbone_eval.matrix import OFFICIAL_TEST_SAMPLES, OFFICIAL_VALIDATION_SAMPLES


def training_device() -> tuple[str, dict[str, str]]:
    """Pick cuda when available and describe the device + host in tags.

    ``RS_EVAL_REQUIRE_DEVICE=cuda`` turns the silent CPU fallback into a hard
    error: the Colab notebook pins it so an A100 session either trains on the
    GPU or fails loudly (review 2026-11-04 round 4).
    """
    import os

    import torch

    required = os.environ.get("RS_EVAL_REQUIRE_DEVICE", "")
    if required == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "RS_EVAL_REQUIRE_DEVICE=cuda but torch.cuda.is_available() is False; "
            "refusing to train on CPU in a GPU-declared cell"
        )

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0).lower().replace(" ", "-")
        tags = {"device": device, "host": name}
    else:
        device = "cpu"
        tags = {"device": device, "host": "cpu"}
    tags["torch"] = torch.__version__
    return device, tags


# The official T11 test fold is 496 patches (full Fold 1-3 contract); the
# authoritative number lives in the control layer so every worker (AEF too)
# anchors on the same value instead of the manifest's own length, which a
# shrunk manifest would self-attest (review 2026-11-04 round 2-5).
EXPECTED_TEST_SAMPLES = OFFICIAL_TEST_SAMPLES["T11"]
EXPECTED_VALIDATION_SAMPLES = OFFICIAL_VALIDATION_SAMPLES["T11"]


def norm_tags(mean, std) -> dict[str, str]:
    """Record the final normalisation honestly.

    The effective input is the DN/10000*255 RGB value z-scored by the
    train-fold tile-balanced statistics (the ImageNet affine pass in
    _decode_* cancels exactly under z-scoring, so it does not influence the
    numbers; NORM_DECISION a808302 is NOT effective for T11 -- declared
    exception, review 2026-11-04). The raw fold statistics go into the tags.
    """
    mean_flat = ", ".join(f"{value:.2f}" for value in mean)
    std_flat = ", ".join(f"{value:.2f}" for value in std)
    return {
        "norm_source": "train_fold_tile_stats",
        "norm_mean_rgb": mean_flat,
        "norm_std_rgb": std_flat,
        "true_input": (
            "aux s2_path stack (T,10,128,128) int16 -> RGB B04/B03/B02, "
            "DN/10000*255, z-scored by train-fold tile stats"
        ),
    }
