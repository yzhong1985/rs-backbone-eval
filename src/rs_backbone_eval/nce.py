"""Symmetric InfoNCE helpers shared by the distillation workers."""

from __future__ import annotations

INFONCE_TEMPERATURE = 0.07
INFONCE_TOKENS = 1024
INFONCE_PATCHES = 8


def patches_per_step(loss_kind: str) -> int:
    """InfoNCE needs more negatives per step; the other losses stay at 2."""
    return INFONCE_PATCHES if loss_kind == "infonce" else 2


def info_nce(student, teacher, temperature: float = INFONCE_TEMPERATURE):
    """Symmetric InfoNCE over matched (student, teacher) token pairs."""
    import torch
    import torch.nn.functional as F

    z = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.float(), dim=1)
    logits = z @ t.transpose(0, 1) / temperature
    labels = torch.arange(z.shape[0], device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def sample_token_pairs(prediction, target, mask, count: int, generator):
    """Flatten observed tokens and sample at most ``count`` matched pairs."""
    import torch

    _, dims, _, _ = prediction.shape
    flat_prediction = prediction.permute(0, 2, 3, 1).reshape(-1, dims)
    flat_target = target.permute(0, 2, 3, 1).reshape(-1, dims)
    indices = mask.bool().reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if indices.numel() > count:
        pick = torch.randperm(indices.numel(), generator=generator)[:count].to(indices.device)
        indices = indices[pick]
    return flat_prediction[indices], flat_target[indices]


def retrieval_top1(student, teacher) -> tuple[int, int]:
    """Count sampled student tokens whose nearest teacher is the matched one."""
    import torch

    unit_student = torch.nn.functional.normalize(student.float(), dim=1)
    unit_teacher = torch.nn.functional.normalize(teacher.float(), dim=1)
    matches = (unit_student @ unit_teacher.transpose(0, 1)).argmax(dim=1)
    labels = torch.arange(student.shape[0], device=student.device)
    return int((matches == labels).sum()), int(student.shape[0])
