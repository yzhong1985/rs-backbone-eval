"""Probe the OlmoEarth v1.2 Tiny encoder contract without downloading weights.

Builds the encoder, constructs a minimal unmasked sample, runs one forward pass
and prints the token shapes. Run with::

    HF_ENDPOINT=https://hf-mirror.com workers/olmoearth/.venv/bin/python scripts/probe_olmoearth.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402


def main() -> int:
    model = load_model_from_id(ModelID.OLMOEARTH_V1_2_TINY, load_weights=False)
    encoder = model.encoder.eval()
    print("encoder params:", round(sum(p.numel() for p in encoder.parameters()) / 1e6, 3), "M")

    batch, height, width, steps, bands = 1, 64, 64, 1, 6
    pixels = torch.randn(batch, height, width, steps, bands)
    mask = torch.full(
        (batch, height, width, steps), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
    )
    timestamps = torch.tensor([[[1, 0, 2020]]], dtype=torch.long)  # [B, T, 3] = day, month, year
    sample = MaskedOlmoEarthSample(
        timestamps=timestamps, sentinel2_l2a=pixels, sentinel2_l2a_mask=mask
    )

    with torch.no_grad():
        out = encoder(sample, patch_size=4)
    print("forward ->", type(out).__name__)
    for key in list(out)[:6]:
        value = out[key]
        shape = getattr(value, "shape", None)
        print(f"  {key}: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
