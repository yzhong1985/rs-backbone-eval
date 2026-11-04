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


def training_device() -> tuple[str, dict[str, str]]:
    """Pick cuda when available and describe the device + host in tags."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0).lower().replace(" ", "-")
        tags = {"device": device, "host": name}
    else:
        device = "cpu"
        tags = {"device": device, "host": "cpu"}
    tags["torch"] = torch.__version__
    return device, tags


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
