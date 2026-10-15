"""Probe the OlmoEarth v1.2 Tiny encoder contract without downloading weights.

Builds the encoder, expands a six-band chip through the band adapter, runs one
forward pass and prints the output shapes. Run with::

    HF_ENDPOINT=https://hf-mirror.com workers/olmoearth/.venv/bin/python scripts/probe_olmoearth.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "olmoearth" / "src"))

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402
from olmoearth_worker.bands import (  # noqa: E402
    bandset_coverage,
    bandset_usable,
    to_olmoearth_s2l2a,
)

PROJECT_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def main() -> int:
    model = load_model_from_id(ModelID.OLMOEARTH_V1_2_TINY, load_weights=False)
    encoder = model.encoder.eval()
    print("encoder params:", round(sum(p.numel() for p in encoder.parameters()) / 1e6, 3), "M")

    height, width, steps = 64, 64, 1
    chip = np.random.randn(len(PROJECT_BANDS), height, width).astype(np.float32)
    expanded = to_olmoearth_s2l2a(chip, PROJECT_BANDS)
    pixels = torch.from_numpy(expanded).permute(1, 2, 0).unsqueeze(0)  # [H,W,12] -> [1,H,W,12]
    pixels = pixels.unsqueeze(3).expand(1, height, width, steps, 12).contiguous()

    present = bandset_coverage(PROJECT_BANDS)
    usable = bandset_usable(PROJECT_BANDS)
    # A partly-present band-set is sent online with filled values (only a set with
    # no band present is masked MISSING), so the SWIR set is not dropped.
    values = [MaskValue.ONLINE_ENCODER.value if ok else MaskValue.MISSING.value for ok in usable]
    mask = (
        torch.tensor(values, dtype=torch.long)
        .view(1, 1, 1, 1, 3)
        .expand(1, height, width, steps, 3)
    )
    timestamps = torch.tensor([[[1, 0, 2020]]], dtype=torch.long)  # [B,T,3] = day, month, year
    sample = MaskedOlmoEarthSample(
        timestamps=timestamps, sentinel2_l2a=pixels, sentinel2_l2a_mask=mask
    )

    print("bandset fully covered:", present)
    print("bandset sent online:", usable)
    with torch.no_grad():
        out = encoder(sample, patch_size=4)
    tokens = out["tokens_and_masks"]
    print("forward ->", type(tokens).__name__)
    for field in tokens._fields:
        tensor = getattr(tokens, field)
        if tensor is None:
            continue
        print(f"  {field}: {tuple(tensor.shape)}")
    pooled = out.get("project_aggregated")
    print(f"  project_aggregated: {tuple(pooled.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
