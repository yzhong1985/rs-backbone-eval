"""Per-group confusion accumulation for the segmentation workers.

Workers already accumulate one global confusion matrix per evaluation; feeding
the same predictions here as well keeps a matrix per group (region/tile/event).
The report then bootstraps an IoU interval by resampling groups and
re-accumulating, which answers "how stable is this over regions?" rather than
"how stable is this over pixels?".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .confusion import ConfusionMatrix, GroupConfusion


class GroupedSegmentation:
    """One global matrix plus one matrix per group for the same predictions."""

    def __init__(self, num_classes: int, *, ignore_index: int | None = None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.global_matrix = ConfusionMatrix(num_classes)
        self.group_matrix = GroupConfusion(num_classes)

    def update(self, group_ids: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Add one batch; ``group_ids`` is aligned with the batch's first axis."""
        truth = np.asarray(y_true)
        pred = np.asarray(y_pred)
        if len(group_ids) != truth.shape[0]:
            raise ValueError(f"got {len(group_ids)} group ids for {truth.shape[0]} samples")
        self.global_matrix.update(truth, pred, ignore_index=self.ignore_index)
        for index, group_id in enumerate(group_ids):
            self.group_matrix.update(
                str(group_id), truth[index], pred[index], ignore_index=self.ignore_index
            )

    def group_confusion(self) -> dict[str, list[list[int]]]:
        return self.group_matrix.to_json()
