"""Evaluation-split resolution shared by the workers.

A run normally reports its headline metric on ``test`` (or ``official_validation``
for datasets that publish only a validation split). Holdout experiments reuse
the same trained model but report on a derived split, so the split is taken from
``RS_EVAL_EVAL_SPLIT`` instead of being hard-coded. The override only changes
where the *final* metric is measured; checkpoint selection still uses the
task's validation split so the holdout never influences the model choice.
"""

from __future__ import annotations

import os

HOLDOUT_SPLITS = ("geographic_holdout", "temporal_holdout")


def evaluation_split(default: str = "test") -> str:
    """Split the worker evaluates on; ``RS_EVAL_EVAL_SPLIT`` overrides ``default``."""
    value = os.environ.get("RS_EVAL_EVAL_SPLIT", "").strip()
    return value or default


def evaluation_splits(default: str = "test") -> list[str]:
    """Comma-separated evaluation splits (defaults to a single ``default``)."""
    raw = os.environ.get("RS_EVAL_EVAL_SPLIT", "").strip()
    if not raw:
        return [default]
    return [part.strip() for part in raw.split(",") if part.strip()]
