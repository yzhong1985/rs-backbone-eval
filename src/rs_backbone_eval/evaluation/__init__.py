"""Common evaluation metrics shared by every model worker.

Metrics are derived from globally accumulated counts (confusion matrices or
error sums) so that per-pixel tasks first aggregate over the whole collection
and then produce one global number, as required by the evaluation protocol.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..contracts import MetricDirection, MetricRecord


def to_metric_records(
    values: Mapping[str, float],
    *,
    primary_name: str,
    direction: MetricDirection = "maximize",
    split: str = "test",
    unit: str | None = None,
) -> list[MetricRecord]:
    """Convert computed metrics into run-record entries.

    Keys of the form ``name/class_label`` become per-class records with the
    class name attached; the key matching ``primary_name`` is flagged primary.
    NaN values are preserved so consumers can see undefined metrics explicitly.
    """
    records: list[MetricRecord] = []
    for name, value in values.items():
        class_name: str | None = None
        if "/" in name:
            name, class_name = name.split("/", 1)
        records.append(
            MetricRecord(
                name=name,
                value=float(value),
                split=split,
                direction=direction,
                unit=unit,
                class_name=class_name,
                is_primary=name == primary_name,
            )
        )
    return records
