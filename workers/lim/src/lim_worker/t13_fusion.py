"""T13 NIR/SAR fusion diagnostics on a frozen full-fine-tuned LIM (plan v3, §2–§3).

The start point is the FT-B0 full checkpoint of the same seed (``whu_ft_b0``, run
``best_full_checkpoint.pt``). Its encoder and UPerNet head stay frozen and in eval
mode; only the NIR/SAR residual branches train. Outputs are logits of the fused
pyramid, so with every branch at zero the logits equal FT-B0's logits.

Training and the test-set evaluation are separate steps here: ``fused_logits`` is
the forward used by both, and the trainer (not in this module yet) must only
score validation tiles while it trains.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .fusion_branches import FusionBranches
from .t13 import FULL_CHECKPOINT

# Protocol recipes for the diagnostic panel. Each one starts from whu_ft_b0.
FUSION_RECIPES: dict[str, dict[str, bool]] = {
    "whu_ft_nir": {"nir": True, "sar": False},
    "whu_ft_sar": {"nir": False, "sar": True},
    "whu_ft_nir_sar": {"nir": True, "sar": True},  # joint readout, only after the singles pass
}
DEPENDENCY_PROTOCOL = "whu_ft_b0"


def find_full_dependency(
    runs_root: Path,
    *,
    seed: int,
    model_id: str,
    manifest_sha: str,
    backbone_sha: str,
    epochs: int,
    batch: int,
    smoke: bool = False,
) -> Path:
    """The one completed FT-B0 run this fusion run may build on; returns its full checkpoint.

    Matches the rules of ``t13.find_dependency`` (seed, model, completed, not
    superseded, same manifest, backbone, epoch budget and effective batch, and the
    same smoke/formal split), but reads ``best_full_checkpoint.pt``, the file that
    holds the encoder and head together.
    """
    hits = []
    for record in Path(runs_root).glob("*/run.json"):
        try:
            run = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tags = run.get("tags") or {}
        if not (
            run.get("task_id") == "T13"
            and run.get("protocol_id") == DEPENDENCY_PROTOCOL
            and run.get("seed") == seed
            and run.get("model_id") == model_id
            and run.get("status") == "completed"
            and not tags.get("superseded")
            and ("SMOKE" in (run.get("notes") or "")) == smoke
        ):
            continue
        if (
            (run.get("dataset") or {}).get("manifest_sha256") == manifest_sha
            and run.get("model_checkpoint_sha256") == backbone_sha
            and tags.get("epochs_budget") == str(epochs)
            and (run.get("training") or {}).get("epochs_completed") == epochs
            and tags.get("effective_batch") == str(batch)
        ):
            hits.append(record.parent)
    if len(hits) != 1:
        raise RuntimeError(
            f"need exactly one completed {DEPENDENCY_PROTOCOL} run for seed {seed} / {model_id}; "
            f"found {len(hits)}: {sorted(h.name for h in hits)}"
        )
    checkpoint = hits[0] / FULL_CHECKPOINT
    if not checkpoint.is_file():
        raise RuntimeError(f"{hits[0].name} has no {FULL_CHECKPOINT}")
    return checkpoint


def fused_logits(
    head: nn.Module,
    levels: Sequence[torch.Tensor],
    branches: FusionBranches,
    output_size: tuple[int, int],
    *,
    nir: torch.Tensor | None = None,
    nir_valid: torch.Tensor | None = None,
    sar: torch.Tensor | None = None,
    sar_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Logits of the frozen head on the residual-corrected LIM pyramid.

    ``levels`` come from the frozen encoder (no grad needed); gradients reach only
    the branches' parameters through the residuals.
    """
    fused = branches(
        list(levels),
        nir=nir,
        nir_valid=nir_valid,
        sar=sar,
        sar_valid=sar_valid,
    )
    return head(fused, output_size=output_size)


def branch_parameters(branches: FusionBranches) -> list[nn.Parameter]:
    """The only parameters the fusion run optimises. The caller freezes the encoder and head."""
    params = list(branches.parameters())
    for param in params:
        param.requires_grad_(True)
    return params
