"""Shard-set selection for the OlmoEarth mirror (no network, no zarr)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "olmoearth_shard_list.py"
spec = importlib.util.spec_from_file_location("olmoearth_shard_list", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TRANSFORM = (10.0, 0.0, 0.0, 0.0, -10.0, 9502720.0)


def test_a_patch_inside_one_shard_maps_to_one_cell() -> None:
    # col0 = xmin / 10 = 0, row0 = (9502720 - ymax) / 10 = 0
    assert module.shards_for_bbox((0.0, 9502719.0, 1280.0, 9502720.0), TRANSFORM) == {(0, 0)}


def test_a_patch_straddling_the_shard_edge_maps_to_four_cells() -> None:
    # 100 px before the 2048 boundary, 128 px wide -> two rows and two columns.
    xmin = (2048 - 100) * 10.0
    ymax = 9502720.0 - (2048 - 100) * 10.0
    cells = module.shards_for_bbox((xmin, ymax - 1280.0, xmin + 1280.0, ymax), TRANSFORM)
    assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_required_shards_groups_by_utm_zone() -> None:
    sidecars = [
        {"tile": "t30uxv", "grid_bbox": (0.0, 9502719.0, 1280.0, 9502720.0)},
        {"tile": "t31uxv", "grid_bbox": (0.0, 9502719.0, 1280.0, 9502720.0)},
        {"tile": "t30uxv", "grid_bbox": (20490.0, 9502719.0, 21770.0, 9502720.0)},
    ]
    by_zone = module.required_shards(sidecars, {30: TRANSFORM, 31: TRANSFORM})
    assert by_zone[30] == {(0, 0), (0, 1)}
    assert by_zone[31] == {(0, 0)}
