"""NIR and SAR residual branches on the frozen LIM pyramid (T13 plan §3).

``RGB -> LIM -> [F1..F4] -> UPerNet``. The branches correct selected LIM levels
before UPerNet:

* ``NirBranch`` reads the optical 4th band (NIR) at full resolution and adds
  residuals at stride 8 (F2) and stride 16 (F3).
* ``SarBranch`` reads the paired SAR sheet and adds gated residuals at stride
  16 (F3) and stride 32 (F4).

Each residual is ``LIM_level + mask * projection(branch_feature)``. The last
1x1 projection starts at zero, so with every branch present the output equals
the baseline pyramid at step 0. A branch whose modality is absent is skipped
explicitly; the validity mask multiplies the residual, so convolution biases
and normalisation cannot turn invalid pixels into non-zero features.

The SAR gate is one scalar per pixel, computed from the SAR feature, the
masked mean of the SAR feature, and the channel mean of the RGB level. A
scalar gate keeps the parameter count small: with the 1x1 projection, the
whole fusion module stays well below the 1 M parameter design target.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .arms import _groups

NIR_LEVELS = (1, 2)  # F2 (stride 8), F3 (stride 16)
SAR_LEVELS = (2, 3)  # F3 (stride 16), F4 (stride 32)
EPS = 1e-6


def zero_invalid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Replace invalid input pixels with 0 before any arithmetic.

    Multiplying by the mask would keep NaN (NaN * 0 is NaN), so the selection
    is explicit: where the mask is 0 the value is 0 whatever the input held.
    """
    return torch.where(valid > 0, x, torch.zeros_like(x))


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Valid-pixel coverage of each feature cell, in ``[0, 1]``.

    Area averaging, not nearest sampling: a cell with one valid pixel out of
    64 gets coverage 1/64. Nearest sampling flipped a whole cell between valid
    and invalid when a single pixel moved by one step.
    """
    if mask.shape[-2:] == size:
        return mask
    return F.adaptive_avg_pool2d(mask, size)


def masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``sum(mask * feat) / sum(mask)`` per image; zero where the mask is empty."""
    num = (feat * mask).sum(dim=(2, 3), keepdim=True)
    den = mask.sum(dim=(2, 3), keepdim=True)
    return torch.where(den > 0, num / den.clamp_min(EPS), torch.zeros_like(num))


def _conv_block(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(_groups(out_ch, 8), out_ch),
        nn.GELU(),
    )


def _zero_projection(in_ch: int, out_ch: int) -> nn.Conv2d:
    proj = nn.Conv2d(in_ch, out_ch, 1)
    nn.init.zeros_(proj.weight)
    nn.init.zeros_(proj.bias)
    return proj


class NirBranch(nn.Module):
    """Single-band CNN: stride 8 and stride 16 features from the NIR band."""

    def __init__(self, in_channels: Sequence[int], levels: Sequence[int] = NIR_LEVELS):
        super().__init__()
        self.levels = tuple(levels)
        self.stem = nn.Sequential(
            _conv_block(1, 16),  # stride 2
            _conv_block(16, 32),  # stride 4
            _conv_block(32, 32),  # stride 8  -> f8, 32 ch
        )
        self.to_16 = _conv_block(32, 64)  # stride 16 -> f16, 64 ch
        feat_ch = {1: 32, 2: 64}
        self.proj = nn.ModuleDict(
            {str(i): _zero_projection(feat_ch[i], in_channels[i]) for i in self.levels}
        )

    def forward(
        self, levels: list[torch.Tensor], nir: torch.Tensor, valid: torch.Tensor
    ) -> list[torch.Tensor]:
        """``nir``/``valid``: ``[B, 1, H, W]`` at image resolution; returns corrected levels."""
        f8 = self.stem(zero_invalid(nir, valid))
        feats = {1: f8, 2: self.to_16(f8)}
        out = list(levels)
        for i in self.levels:
            size = tuple(levels[i].shape[-2:])
            residual = self.proj[str(i)](_match(feats[i], size))
            out[i] = levels[i] + residual * resize_mask(valid, size)
        return out


class SarBranch(nn.Module):
    """Single-band SAR CNN with a per-pixel gate from SAR and RGB context."""

    def __init__(self, in_channels: Sequence[int], levels: Sequence[int] = SAR_LEVELS):
        super().__init__()
        self.levels = tuple(levels)
        self.stem = nn.Sequential(
            _conv_block(1, 16),  # stride 2
            _conv_block(16, 32),  # stride 4
            _conv_block(32, 64),  # stride 8
        )
        self.to_16 = _conv_block(64, 64)  # stride 16 -> f16, 64 ch
        self.to_32 = _conv_block(64, 128)  # stride 32 -> f32, 128 ch
        feat_ch = {2: 64, 3: 128}
        self.proj = nn.ModuleDict(
            {str(i): _zero_projection(feat_ch[i], in_channels[i]) for i in self.levels}
        )
        # Gate input: SAR feature, SAR masked mean, RGB channel mean (one map).
        self.gate = nn.ModuleDict(
            {str(i): nn.Conv2d(2 * feat_ch[i] + 1, 1, 1) for i in self.levels}
        )
        for gate in self.gate.values():
            nn.init.zeros_(gate.bias)  # sigmoid(0) = 0.5 at start: the projection gets gradient

    def forward(
        self,
        levels: list[torch.Tensor],
        sar: torch.Tensor,
        valid: torch.Tensor,
    ) -> list[torch.Tensor]:
        """``sar``/``valid``: ``[B, 1, H, W]`` at image resolution; returns corrected levels."""
        f16 = self.to_16(self.stem(zero_invalid(sar, valid)))
        feats = {2: f16, 3: self.to_32(f16)}
        out = list(levels)
        for i in self.levels:
            size = tuple(levels[i].shape[-2:])
            feat = _match(feats[i], size)
            mask = resize_mask(valid, size)
            ctx = masked_mean(feat, mask).expand_as(feat)
            rgb = levels[i].mean(dim=1, keepdim=True)
            gate = torch.sigmoid(self.gate[str(i)](torch.cat([feat, ctx, rgb], dim=1)))
            out[i] = levels[i] + self.proj[str(i)](feat) * gate * mask
        return out


def _match(feat: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if feat.shape[-2:] == size:
        return feat
    return F.interpolate(feat, size=size, mode="bilinear", align_corners=False)


class FusionBranches(nn.Module):
    """Optional NIR and SAR branches applied to one LIM pyramid.

    A branch runs only when it was built (``use_nir`` / ``use_sar``) and its
    modality tensor is passed. Passing nothing leaves the pyramid unchanged.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        use_nir: bool = False,
        use_sar: bool = False,
    ):
        super().__init__()
        self.nir = NirBranch(in_channels) if use_nir else None
        self.sar = SarBranch(in_channels) if use_sar else None

    def forward(
        self,
        levels: list[torch.Tensor],
        *,
        nir: torch.Tensor | None = None,
        nir_valid: torch.Tensor | None = None,
        sar: torch.Tensor | None = None,
        sar_valid: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        # Both branches read the baseline pyramid, not each other's output, and
        # their residuals are summed. Chaining them would let the NIR residual
        # change the SAR gate, which adds an interaction the joint readout does
        # not mean to measure.
        out = list(levels)
        if self.nir is not None and nir is not None:
            if nir_valid is None:
                nir_valid = torch.ones_like(nir[:, :1])
            nir_out = self.nir(levels, nir, nir_valid)
            for i in self.nir.levels:
                out[i] = out[i] + (nir_out[i] - levels[i])
        if self.sar is not None and sar is not None:
            if sar_valid is None:
                sar_valid = torch.ones_like(sar[:, :1])
            sar_out = self.sar(levels, sar, sar_valid)
            for i in self.sar.levels:
                out[i] = out[i] + (sar_out[i] - levels[i])
        return out
