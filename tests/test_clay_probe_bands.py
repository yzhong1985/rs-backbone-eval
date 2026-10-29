"""The Clay probe must follow the task's real bands, not assume Sentinel-2.

Regression 2026-10-29: the T9 (iSAID) probe died with
`index 3 is out of bounds for axis 0 with size 3` because the probe always took
Clay's ten S2 band indices from a three-band RGB tile, so the Clay iSAID smoke
could never have run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rs_backbone_eval.registry import TaskSpec

# The worker package is not installed in the control-plane environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers" / "clay" / "src"))

S2_TASK = {
    "task_id": "T1",
    "application": "flood",
    "dataset_id": "Sen1Floods11",
    "family": "segmentation",
    "sensor": "sentinel-2",
    "time_steps": 1,
    "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    "num_outputs": 2,
    "primary_metric": {"name": "water_iou", "direction": "maximize"},
    "official_config_url": "https://example.invalid/t1",
}
RGB_TASK = {
    **S2_TASK,
    "task_id": "T9",
    "dataset_id": "iSAID",
    "sensor": "aerial-rgb",
    "bands": ["RED", "GREEN", "BLUE"],
    "num_outputs": 16,
}


def test_rgb_tasks_use_clays_aerial_entry_point() -> None:
    from clay_worker.clay_probe import probe_bands

    sensor, order, indices = probe_bands(TaskSpec.model_validate(RGB_TASK))
    assert sensor == "naip"
    assert order == ["red", "green", "blue"]
    assert indices == [0, 1, 2]


def test_multispectral_tasks_keep_the_sentinel2_mapping() -> None:
    from clay_worker.clay_probe import probe_bands

    sensor, order, indices = probe_bands(TaskSpec.model_validate(S2_TASK))
    assert sensor == "sentinel-2-l2a"
    assert len(order) == len(indices) == 10
    assert order[0] == "blue" and indices[0] == 1
