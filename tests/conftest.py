"""Shared builders for T3-shaped manifests and task specs used across tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rs_backbone_eval.data.sequence import GeoSequenceSample
from rs_backbone_eval.registry import TaskSpec

TASK_PAYLOAD: dict = {
    "task_id": "T3",
    "application": "Burn Scar Intensity",
    "dataset_id": "Burn Intensity",
    "family": "segmentation",
    "sensor": "hls",
    "time_steps": 3,
    "bands": ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"],
    "num_outputs": 5,
    "ignore_index": -1,
    "primary_metric": {"name": "mean_iou", "direction": "maximize"},
    "official_config_url": "https://example.invalid/t3",
}

SAMPLE_PAYLOAD: dict = {
    "task_id": "T3",
    "sample_id": "one",
    "group_id": "fire-a",
    "split": "train",
    "image_paths": ["pre.tif", "during.tif", "post.tif"],
    "label_path": "label.tif",
    "sensor": "hls",
    "bands": ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"],
    "timestamps": ["pre", "during", "post"],
    "temporal_mask": [True, True, True],
    "crs": "EPSG:32610",
    "transform": [30, 0, 0, 0, -30, 0],
    "gsd_m": 30,
    "checksum": "a" * 64,
}


@pytest.fixture(autouse=True)
def _drop_portal_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer .env may carry RS_EVAL_PORTAL_TOKEN; tests start unguarded.

    paths.py calls load_dotenv() on every RuntimePaths lookup, and dotenv only
    fills variables that are unset -- hence an empty string, not delenv(), or
    the developer's token would be pulled back in mid-test (2026-10-28).
    """
    monkeypatch.setenv("RS_EVAL_PORTAL_TOKEN", "")


@pytest.fixture
def make_task() -> Callable[..., TaskSpec]:
    def _make_task(**overrides) -> TaskSpec:
        payload = {**TASK_PAYLOAD, **overrides}
        return TaskSpec.model_validate(payload)

    return _make_task


@pytest.fixture
def make_row() -> Callable[..., GeoSequenceSample]:
    def _make_row(
        sample_id: str = "one",
        group_id: str = "fire-a",
        split: str = "train",
        **overrides,
    ) -> GeoSequenceSample:
        payload = {
            **SAMPLE_PAYLOAD,
            "sample_id": sample_id,
            "group_id": group_id,
            "split": split,
            **overrides,
        }
        return GeoSequenceSample.model_validate(payload)

    return _make_row
