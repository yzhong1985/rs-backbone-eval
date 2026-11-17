"""Image-only deployment arms for T13 (WHU-OPT-SAR): B0 / R / D / Bcap / P.

Port of ``docs/design/实验执行方案_v3.4.md`` §5.1 with the fixes listed in
``docs/design/实验执行方案_v3.4_复核意见_2026-11-17.md`` §2:

* real LIM pyramid (MiT-Huge 256/512/1024/2048 at strides 4/8/16/32), so every
  parameter count is derived from the module, not copied from the plan;
* the shared ``rs_backbone_eval.heads.UPerNetHead`` (GroupNorm, fpn 256, the
  same head as LIM T9) instead of the plan's BatchNorm ``TinyUPerNet``: a frozen
  head then has no running statistics that could drift;
* ``configure_trainable`` sets ``requires_grad`` for the whole model from the
  arm/stage and never silently re-enables something the caller froze;
* a loaded regional plugin inherits the model's train/eval mode (the plan left
  it in train mode, so NIR dropout kept firing at inference after a hot swap).

Arms (all operate on the frozen LIM pyramid ``feats``):

========  ======================================  ==============================
arm       training inputs                         deployment inputs
========  ======================================  ==============================
B0        feats                                   image
R         feats + AEF (R_null: zero / shuffled)   image + AEF (reference only)
D         feats + frozen R logits (KL)            image
Bcap      feats (+ NIR), plugin, lambda = 0        image (+ NIR)
P         feats (+ NIR), plugin, AEF alignment     image (+ NIR)
========  ======================================  ==============================

Bcap and P are the same module; they differ only in the codebook
initialisation (random vs. train-fold AEF k-means) and the alignment weight.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rs_backbone_eval.heads import UPerNetHead

ARMS = ("B0", "R", "D", "Bcap", "P")
PLUGIN_ARMS = ("Bcap", "P")
AEF_DIM = 64
AEF_NODATA = -128
NIR_STEM_DIM = 32
PLUGIN_FORMAT = "t13-regional-plugin-v1"


# ---------------------------------------------------------------------------
# Teacher decoding, pooling and losses
# ---------------------------------------------------------------------------


def decode_aef(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 AEF codes ``[B,64,H,W]`` -> (float features, ``[B,1,H,W]`` validity).

    AEF stores ``sign(x) * sqrt(|x|) * 127.5``; the inverse is
    ``sign(q) * (|q| / 127.5) ** 2``. A cell is valid only if no band is the
    ``-128`` NoData code. Decoded float input is rejected: decoding twice would
    square the values again.
    """
    if codes.is_floating_point():
        raise TypeError("decode_aef expects the raw int8 codes, got a float tensor")
    valid = (codes != AEF_NODATA).all(dim=1, keepdim=True)
    q = codes.float()
    features = torch.sign(q) * (q.abs() / 127.5) ** 2
    return features * valid, valid.float()


def pool_teacher(
    teacher: torch.Tensor, valid: torch.Tensor, size: tuple[int, int], min_cover: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked average of the teacher onto a ``size`` grid.

    Non-finite values are zeroed *before* multiplying by the mask (``nan * 0``
    is still ``nan``). A target cell is valid when at least ``min_cover`` of its
    footprint is valid teacher data.
    """
    clean = torch.nan_to_num(teacher.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = (torch.nan_to_num(valid.float(), nan=0.0) > 0.5).float()
    total = F.adaptive_avg_pool2d(clean * mask, size)
    cover = F.adaptive_avg_pool2d(mask, size)
    pooled_valid = (cover >= min_cover).float()
    return total / cover.clamp_min(1e-6) * pooled_valid, pooled_valid


def safe_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """CE that returns a graph-connected zero when every pixel is ignored."""
    if not (target != ignore_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def alignment_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, l1_weight: float = 0.5
) -> torch.Tensor:
    """``(1 - cos) + l1_weight * SmoothL1`` on valid cells; zero if none are valid."""
    keep = valid.squeeze(1) > 0.5
    if not keep.any():
        return pred.sum() * 0.0
    cosine = 1.0 - (F.normalize(pred, dim=1) * F.normalize(target, dim=1)).sum(dim=1)
    l1 = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    return cosine[keep].mean() + l1_weight * l1[keep].mean()


def distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    temperature: float = 4.0,
) -> torch.Tensor:
    """``tau^2 * KL(teacher || student)`` averaged over labelled pixels."""
    keep = target != ignore_index
    if not keep.any():
        return student.sum() * 0.0
    log_p = F.log_softmax(student / temperature, dim=1)
    q = F.softmax(teacher.detach() / temperature, dim=1)
    kl = F.kl_div(log_p, q, reduction="none").sum(dim=1)
    return kl[keep].mean() * temperature**2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _groups(channels: int, preferred: int = 16) -> int:
    for groups in (preferred, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ZeroInjector(nn.Module):
    """Residual injector whose last layer starts at zero (output 0 at step 0).

    ``mode="mlp"``: Conv1x1(in->hidden) -> GN -> GELU -> Conv1x1(hidden->out);
    ``mode="linear"``: one Conv1x1(in->out). Only the last conv is zeroed, so
    the first layer still has a random basis for the gradient to use.
    """

    def __init__(self, in_dim: int, out_dim: int, mode: str = "mlp", hidden: int = 128):
        super().__init__()
        if mode == "linear":
            self.net = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
        elif mode == "mlp":
            self.net = nn.Sequential(
                nn.Conv2d(in_dim, hidden, 1, bias=False),
                nn.GroupNorm(_groups(hidden, 8), hidden),
                nn.GELU(),
                nn.Conv2d(hidden, out_dim, 1),
            )
        else:
            raise ValueError(f"unknown injector mode {mode!r}")
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.net(x)


class QueryHead(nn.Module):
    """Bottleneck depthwise-separable head: feature level -> 64-d AEF query."""

    def __init__(self, in_dim: int, out_dim: int = AEF_DIM, mid: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid, 1, bias=False),
            nn.GroupNorm(16, mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.GroupNorm(16, mid),
            nn.GELU(),
            nn.Conv2d(mid, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegionalPlugin(nn.Module):
    """Scale-decoupled regional plugin (plan v3.4 §2.2), shared by Bcap and P.

    * shallow levels: never receive AEF; with a 4th (NIR) band, a zero-init
      linear injector adds a native-resolution NIR edge residual;
    * deep levels: a query head maps the feature to a 64-d query, attention over
      the K-prototype codebook synthesises an AEF-like vector, and a zero-init
      MLP injector adds it back. With NIR, a zero-init residual corrects the
      query.

    ``nir_input_mode="rgb_mean_placeholder"`` feeds the RGB mean through the NIR
    path: same capacity, no NIR information (the A-03 attribution control).
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_prototypes: int = 512,
        shallow_levels: Sequence[int] = (0, 1),
        deep_levels: Sequence[int] = (2, 3),
        temperature: float = 0.07,
        nir_drop_prob: float = 0.5,
        nir_input_mode: str = "real_nir",
    ):
        super().__init__()
        if nir_input_mode not in ("real_nir", "rgb_mean_placeholder"):
            raise ValueError(f"unknown nir_input_mode {nir_input_mode!r}")
        if set(shallow_levels) & set(deep_levels):
            raise ValueError("a level cannot be both shallow and deep")
        self.in_channels = [int(c) for c in in_channels]
        self.num_prototypes = int(num_prototypes)
        self.shallow_levels = tuple(int(i) for i in shallow_levels)
        self.deep_levels = tuple(int(i) for i in deep_levels)
        self.temperature = float(temperature)
        self.nir_drop_prob = float(nir_drop_prob)
        self.nir_input_mode = nir_input_mode
        self.codebook_metadata: dict[str, Any] = {"init": "random"}

        self.codebook = nn.Parameter(F.normalize(torch.randn(num_prototypes, AEF_DIM), dim=-1))
        levels = len(self.in_channels)
        self.nir_stem = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    1 if i == 0 else NIR_STEM_DIM,
                    NIR_STEM_DIM,
                    3,
                    stride=4 if i == 0 else 2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, NIR_STEM_DIM),
                nn.GELU(),
            )
            for i in range(levels)
        )
        self.shallow_nir = nn.ModuleDict(
            {
                str(i): ZeroInjector(NIR_STEM_DIM, self.in_channels[i], "linear")
                for i in self.shallow_levels
            }
        )
        self.query = nn.ModuleDict(
            {str(i): QueryHead(self.in_channels[i]) for i in self.deep_levels}
        )
        self.query_nir = nn.ModuleDict()
        for i in self.deep_levels:
            head = nn.Sequential(
                nn.Conv2d(self.in_channels[i] + NIR_STEM_DIM, 64, 1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, AEF_DIM, 1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.query_nir[str(i)] = head
        self.inject = nn.ModuleDict(
            {str(i): ZeroInjector(AEF_DIM, self.in_channels[i], "mlp") for i in self.deep_levels}
        )

    def config(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "num_prototypes": self.num_prototypes,
            "shallow_levels": list(self.shallow_levels),
            "deep_levels": list(self.deep_levels),
            "temperature": self.temperature,
            "nir_drop_prob": self.nir_drop_prob,
            "nir_input_mode": self.nir_input_mode,
        }

    @torch.no_grad()
    def init_codebook_kmeans(
        self, vectors: torch.Tensor, seed: int, iterations: int = 15, **metadata: Any
    ) -> None:
        """Cosine k-means over train-fold AEF vectors ``[N, 64]`` (deterministic)."""
        if vectors.ndim != 2 or vectors.shape[1] != AEF_DIM:
            raise ValueError(f"expected [N, {AEF_DIM}], got {tuple(vectors.shape)}")
        data = torch.nan_to_num(vectors.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        data = F.normalize(data[data.abs().sum(dim=1) > 1e-8], dim=-1)
        if data.shape[0] < self.num_prototypes:
            raise ValueError(f"{data.shape[0]} valid vectors < {self.num_prototypes} prototypes")
        gen = torch.Generator().manual_seed(seed)
        centers = data[torch.randperm(data.shape[0], generator=gen)[: self.num_prototypes]].clone()
        for _ in range(iterations):
            assign = (data @ centers.T).argmax(dim=1)
            sums = torch.zeros_like(centers).index_add_(0, assign, data)
            counts = torch.bincount(assign, minlength=self.num_prototypes)
            empty = counts == 0
            centers = F.normalize(sums, dim=-1)
            if empty.any():
                refill = torch.randperm(data.shape[0], generator=gen)[: int(empty.sum())]
                centers[empty] = data[refill]
        self.codebook.copy_(centers.to(self.codebook))
        self.codebook_metadata = {
            "init": "kmeans",
            "seed": seed,
            "train_vectors": int(data.shape[0]),
            **metadata,
        }

    def _nir_pyramid(
        self, image: torch.Tensor, feats: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if self.nir_input_mode == "real_nir":
            x = image[:, 3:4]
        else:
            x = image[:, :3].mean(dim=1, keepdim=True)
        out = []
        for stem, level in zip(self.nir_stem, feats, strict=True):
            x = stem(x)
            out.append(
                x
                if x.shape[-2:] == level.shape[-2:]
                else F.interpolate(x, size=level.shape[-2:], mode="bilinear", align_corners=False)
            )
            x = out[-1]
        return out

    def _retrieve(self, query: torch.Tensor) -> torch.Tensor:
        b, d, h, w = query.shape
        codebook = F.normalize(self.codebook, dim=-1)
        q = F.normalize(query, dim=1).permute(0, 2, 3, 1).reshape(b, h * w, d)
        attention = torch.softmax(q @ codebook.T / self.temperature, dim=-1)
        return (attention @ codebook).reshape(b, h, w, d).permute(0, 3, 1, 2)

    def forward(
        self, feats: Sequence[torch.Tensor], image: torch.Tensor
    ) -> dict[str, list[torch.Tensor]]:
        if image.shape[1] not in (3, 4):
            raise ValueError(f"plugin expects a 3- or 4-band image, got {image.shape[1]}")
        has_nir = image.shape[1] == 4
        use_nir = has_nir and not (self.training and torch.rand(()).item() < self.nir_drop_prob)
        nir = self._nir_pyramid(image, feats) if has_nir else None
        fused: list[torch.Tensor] = []
        proxy_rgb: list[torch.Tensor] = []
        proxy_full: list[torch.Tensor] = []
        for i, level in enumerate(feats):
            key = str(i)
            if i in self.shallow_levels:
                fused.append(
                    level + self.shallow_nir[key](nir[i], level.shape[-2:]) if use_nir else level
                )
            elif i in self.deep_levels:
                query = self.query[key](level)
                synth = self._retrieve(query)
                proxy_rgb.append(synth)
                if nir is not None:
                    full = self._retrieve(
                        query + self.query_nir[key](torch.cat([level, nir[i]], dim=1))
                    )
                    proxy_full.append(full)
                active = proxy_full[-1] if use_nir else synth
                fused.append(level + self.inject[key](active, level.shape[-2:]))
            else:
                fused.append(level)
        return {"fused": fused, "proxy_rgb": proxy_rgb, "proxy_full": proxy_full}


# ---------------------------------------------------------------------------
# Arm wrapper
# ---------------------------------------------------------------------------


def head_hash(head: nn.Module, backbone_id: str) -> str:
    """16-hex digest of the shared head (structure + every tensor) and backbone id.

    The LIM encoder is fixed by its checkpoint, so it is identified by
    ``backbone_id`` (checkpoint sha256) instead of hashing 524 M parameters.
    """
    digest = hashlib.sha256(backbone_id.encode())
    digest.update(repr(head).encode())
    for name, tensor in sorted(head.state_dict().items()):
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


class ArmModel(nn.Module):
    """Shared head + optional arm module on top of a frozen LIM pyramid.

    The encoder is not part of this module: features come from the cache (or a
    live frozen encoder) so the same object trains on cached pyramids and
    deploys behind a real encoder. ``D`` is structurally ``B0``; its difference
    is the training loss (``distillation_loss`` against a frozen ``R``).
    """

    def __init__(
        self,
        arm: str,
        in_channels: Sequence[int],
        num_classes: int,
        *,
        fpn_dim: int = 256,
        deep_levels: Sequence[int] = (2, 3),
        plugin: dict[str, Any] | None = None,
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        plugin = dict(plugin or {})
        self.in_channels = [int(c) for c in in_channels]
        # a plugin config may narrow the deep levels (tests, ablations); R uses the same set
        self.deep_levels = tuple(int(i) for i in plugin.pop("deep_levels", deep_levels))
        self.head = UPerNetHead(self.in_channels, num_classes=num_classes, fpn_dim=fpn_dim)
        self.injectors: nn.ModuleDict | None = None
        self.plugin: RegionalPlugin | None = None
        self.plugin_enabled = True
        if arm == "R":
            # Same levels and injector as the plugin's deep path, so R and P
            # differ only in where the 64-d vector comes from.
            self.injectors = nn.ModuleDict(
                {
                    str(i): ZeroInjector(AEF_DIM, self.in_channels[i], "mlp")
                    for i in self.deep_levels
                }
            )
        elif arm in PLUGIN_ARMS:
            self.plugin = RegionalPlugin(self.in_channels, deep_levels=self.deep_levels, **plugin)

    # -- trainability ------------------------------------------------------

    def configure_trainable(
        self, *, train_head: bool, freeze_codebook: bool = True
    ) -> list[nn.Parameter]:
        """Set ``requires_grad`` for every parameter and return the trainable ones.

        Plugin arms train the plugin on a *locked* head (``train_head=False``):
        the head comes from B0 and stays bit-identical. The codebook is frozen
        unless ``freeze_codebook=False``.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        if train_head:
            for p in self.head.parameters():
                p.requires_grad_(True)
        if self.injectors is not None:
            for p in self.injectors.parameters():
                p.requires_grad_(True)
        if self.plugin is not None:
            for name, p in self.plugin.named_parameters():
                p.requires_grad_(not (freeze_codebook and name == "codebook"))
        return [p for p in self.parameters() if p.requires_grad]

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        image: torch.Tensor,
        aef: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """``feats``: LIM pyramid; ``image``: normalised bands; ``aef``: decoded, R only."""
        if aef is not None and self.arm != "R":
            raise RuntimeError(f"arm {self.arm} is image-only and must not receive AEF")
        if len(feats) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} levels, got {len(feats)}")
        out: dict[str, Any] = {"proxy_rgb": [], "proxy_full": []}
        levels = list(feats)
        if self.arm == "R":
            if aef is None:
                raise RuntimeError("arm R needs the AEF input (zeros or shuffled for R_null)")
            for i in self.deep_levels:
                levels[i] = levels[i] + self.injectors[str(i)](aef, levels[i].shape[-2:])
        elif self.plugin is not None and self.plugin_enabled:
            plug = self.plugin(levels, image)
            levels = plug["fused"]
            out["proxy_rgb"], out["proxy_full"] = plug["proxy_rgb"], plug["proxy_full"]
        out["logits"] = self.head(levels, output_size=tuple(image.shape[-2:]))
        return out

    # -- regional plugin files ---------------------------------------------

    def save_plugin(self, path, backbone_id: str, **metadata: Any) -> None:
        if self.plugin is None:
            raise RuntimeError(f"arm {self.arm} has no regional plugin")
        torch.save(
            {
                "format": PLUGIN_FORMAT,
                "head_hash": head_hash(self.head, backbone_id),
                "config": self.plugin.config(),
                "codebook_metadata": self.plugin.codebook_metadata,
                "metadata": metadata,
                "state_dict": {k: v.detach().cpu() for k, v in self.plugin.state_dict().items()},
            },
            path,
        )

    def load_plugin(self, path, backbone_id: str) -> dict[str, Any]:
        """Swap in a regional plugin; refuses one trained on a different head."""
        if self.arm not in PLUGIN_ARMS:
            raise RuntimeError(f"arm {self.arm} has no regional plugin")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != PLUGIN_FORMAT:
            raise ValueError(f"not a {PLUGIN_FORMAT} file: {payload.get('format')!r}")
        expected = head_hash(self.head, backbone_id)
        if payload["head_hash"] != expected:
            raise ValueError(f"plugin head hash {payload['head_hash']} != model {expected}")
        config = dict(payload["config"])
        in_channels = config.pop("in_channels")
        if list(in_channels) != self.in_channels:
            raise ValueError(f"plugin channels {in_channels} != model {self.in_channels}")
        reference = next(self.head.parameters())
        plugin = RegionalPlugin(in_channels, **config).to(reference.device, reference.dtype)
        plugin.load_state_dict(payload["state_dict"], strict=True)
        plugin.codebook_metadata = dict(payload.get("codebook_metadata", {}))
        plugin.train(self.training)  # a hot-swapped plugin must not stay in train mode
        for p in plugin.parameters():
            p.requires_grad_(False)
        self.plugin = plugin
        self.plugin_enabled = True
        return dict(payload.get("metadata", {}))
