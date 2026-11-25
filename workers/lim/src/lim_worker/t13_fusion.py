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

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn

from .fusion_branches import FusionBranches
from .t13 import FULL_CHECKPOINT

# Protocol recipes for the diagnostic panel. Each one starts from whu_ft_b0.
FUSION_RECIPES: dict[str, dict[str, bool]] = {
    "whu_ft_nir": {"nir": True, "sar": False},
    "whu_ft_nir_head": {"nir": True, "sar": False},  # injector + trainable UPerNet head
    "whu_ft_sar": {"nir": False, "sar": True},
    "whu_ft_nir_sar": {"nir": True, "sar": True},  # joint readout, only after the singles pass
}
DEPENDENCY_PROTOCOL = "whu_ft_b0"
# Pinned FT-B0 baselines (plan v3 §2): run ID, training commit and SHA-256 of
# best_full_checkpoint.pt. Seed 73 is absent until its run finishes and is hashed.
FT_B0_BASELINES: dict[int, dict[str, str]] = {
    17: {
        "run_id": "lim_mit_huge-T13-whu_ft_b0-train-17",
        "git_commit": "b431d11a91168b08dd7377129545a1021d213344",
        "checkpoint_sha256": "d7a6e843011b5e8b9f2270a3fad78b755cb2d6a3e19e34d422b3ee2ac467813e",
        "manifest_sha256": "93483483b9a7f8b29ad8896010eebe3b7963a78dd240c4512389a1b7fdfce91e",
        "backbone_sha256": "3c2e1b2e9257d8a74330a507145b82d4e6dd5bea65705aec73c8ae0caa8a6985",
        "model_id": "lim_mit_huge",
        "effective_batch": "8",
    },
    42: {
        "run_id": "lim_mit_huge-T13-whu_ft_b0-train-42",
        "git_commit": "b431d11a91168b08dd7377129545a1021d213344",
        "checkpoint_sha256": "3c59a10f8148ceac5a4e345bfee67bd0390a66967acf4bc763fca34d67be68f8",
        "manifest_sha256": "93483483b9a7f8b29ad8896010eebe3b7963a78dd240c4512389a1b7fdfce91e",
        "backbone_sha256": "3c2e1b2e9257d8a74330a507145b82d4e6dd5bea65705aec73c8ae0caa8a6985",
        "model_id": "lim_mit_huge",
        "effective_batch": "8",
    },
    73: {
        "run_id": "lim_mit_huge-T13-whu_ft_b0-train-73",
        "git_commit": "b431d11a91168b08dd7377129545a1021d213344",
        "checkpoint_sha256": "931869809ba6ece78fed4188dc06aead4bea8da65af72ab82ad102146268db1b",
        "manifest_sha256": "93483483b9a7f8b29ad8896010eebe3b7963a78dd240c4512389a1b7fdfce91e",
        "backbone_sha256": "3c2e1b2e9257d8a74330a507145b82d4e6dd5bea65705aec73c8ae0caa8a6985",
        "model_id": "lim_mit_huge",
        "effective_batch": "8",
    },
}


def _read_record(run_dir: Path) -> dict:
    """The run record as a dict; anything else is refused, not skipped."""
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{run_dir.name}: run.json unreadable: {exc}") from exc
    if not isinstance(run, dict):
        raise RuntimeError(f"{run_dir.name}: run.json is not an object")
    return run


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def admit_formal(run: dict, repo: Path) -> None:
    """Raise unless the record is a formal result under the ledger's own rules.

    The record must validate as a ``RunRecord`` and must not be in
    ``audit.excluded_run_ids`` (a commit that predates a metric fix, or no commit).

    The ancestry check needs the Git history of ``repo``. The container image has
    no ``.git``, so this admission fails there until a read-only history is mounted.
    The host deploy checkout is the supported path for now.
    """
    from rs_backbone_eval.audit import excluded_run_ids
    from rs_backbone_eval.contracts import RunRecord
    from rs_backbone_eval.matrix import is_reduced_run

    try:
        record = RunRecord.model_validate(run)
    except ValueError as exc:
        raise RuntimeError(f"{run.get('run_id')}: record does not validate: {exc}") from exc
    # Reuses the ledger's own rule: notes, budget tags and the epoch/step/batch
    # counters, so a batch- or epoch-override run is refused without a notes mark.
    if is_reduced_run(record):
        raise RuntimeError(f"{record.run_id} is a reduced-budget run")
    if record.run_id in excluded_run_ids([record], repo):
        raise RuntimeError(f"{record.run_id} is excluded from formal results")


def find_full_dependency(
    runs_root: Path,
    *,
    seed: int,
    manifest_sha: str,
    model_id: str,
    batch: int,
    repo: Path,
    pins: dict[int, dict[str, str]] = FT_B0_BASELINES,
    verify_sha: bool = True,
    admit: Callable[[dict, Path], None] | None = None,
) -> Path:
    """The pinned FT-B0 full checkpoint for ``seed``, after identity and admission checks.

    The run is read by its pinned ID, not found by scanning the ledger. Identity:
    run ID, task, protocol, seed, training commit, the 20-epoch budget, the
    manifest, model, backbone checkpoint and effective batch of the current request,
    and the checkpoint SHA-256. Admission: the record must validate as a ``RunRecord``,
    be completed, not superseded or smoke, carry no REDUCED-BUDGET mark, and not be in
    ``audit.excluded_run_ids`` (stale commit or no commit). Smoke runs use the same
    formal baseline; only the plugin's own budget is shortened.
    """
    pin = pins.get(seed)
    if pin is None:
        raise RuntimeError(
            f"no pinned FT-B0 baseline for seed {seed}; hash its best_full_checkpoint.pt "
            "and add it to FT_B0_BASELINES first"
        )
    run_dir = Path(runs_root) / pin["run_id"]
    run = _read_record(run_dir)
    tags = run.get("tags") if isinstance(run.get("tags"), dict) else {}
    training = run.get("training") if isinstance(run.get("training"), dict) else {}
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    notes = run.get("notes") or ""
    checks = {
        "run_id": run.get("run_id") == pin["run_id"],
        "task": run.get("task_id") == "T13",
        "protocol": run.get("protocol_id") == DEPENDENCY_PROTOCOL,
        "seed": run.get("seed") == seed,
        "status completed": run.get("status") == "completed",
        "not superseded": not tags.get("superseded"),
        "not smoke": "SMOKE" not in notes,
        "not reduced budget": "REDUCED-BUDGET" not in notes
        and "REDUCED-BUDGET" not in json.dumps(tags),
        "training commit": run.get("git_commit") == pin["git_commit"],
        "20-epoch budget": tags.get("epochs_budget") == "20"
        and training.get("epochs_completed") == 20,
        "manifest": dataset.get("manifest_sha256") == manifest_sha == pin["manifest_sha256"],
        "model": run.get("model_id") == model_id == pin["model_id"],
        "backbone": run.get("model_checkpoint_sha256") == pin["backbone_sha256"],
        "effective batch": tags.get("effective_batch") == str(batch) == pin["effective_batch"],
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise RuntimeError(f"{pin['run_id']} is not the pinned baseline: failed {failed}")
    (admit or admit_formal)(run, repo)
    checkpoint = run_dir / FULL_CHECKPOINT
    if not checkpoint.is_file():
        raise RuntimeError(f"{pin['run_id']} has no {FULL_CHECKPOINT}")
    if verify_sha and _sha256(checkpoint) != pin["checkpoint_sha256"]:
        raise RuntimeError(f"{checkpoint}: SHA-256 differs from the pinned baseline")
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
