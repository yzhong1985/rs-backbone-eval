"""FBP extractor: cache validation and window placement (no data, no GDAL)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fbp_extract_windows.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fbp_extract_windows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Scale:
    """Stand-in for a rasterio Affine: enough for ``transform * (col, row)``."""

    def __mul__(self, point):
        col, row = point
        return (col * 10.0, -row * 10.0)


def test_windows_stay_inside_their_quadrants_and_do_not_overlap() -> None:
    module = load_module()
    labels = np.ones((1200, 1200), dtype=np.uint8)

    windows = module.pick_windows(labels, Scale())

    assert len(windows) == 4
    quadrants = {(0, 0), (0, 600), (600, 0), (600, 600)}
    seen = set()
    for top, left, _, _, _ in windows:
        assert 0 <= top <= 1200 - module.WINDOW
        assert 0 <= left <= 1200 - module.WINDOW
        quadrant = (top // 600 * 600, left // 600 * 600)
        assert quadrant in quadrants
        assert quadrant not in seen, "two windows landed in the same quadrant"
        seen.add(quadrant)
    for i, first in enumerate(windows):
        for second in windows[i + 1 :]:
            row_overlap = min(first[0] + module.WINDOW, second[0] + module.WINDOW) - max(
                first[0], second[0]
            )
            col_overlap = min(first[1] + module.WINDOW, second[1] + module.WINDOW) - max(
                first[1], second[1]
            )
            assert row_overlap <= 0 or col_overlap <= 0, "windows overlap"


def test_save_npz_leaves_no_temporary_file(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "scene.npz"

    module._save_npz(target, version=np.array("v1"), windows=np.zeros((0, 5)))

    assert target.exists()
    assert not list(tmp_path.glob("*.tmp.npz"))
    assert str(np.load(target)["version"]) == "v1"


def test_is_current_rejects_a_stale_version(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "scene.npz"
    module._save_npz(target, version=np.array("v1"))

    assert module._is_current(target, {"version": "v1"})
    assert not module._is_current(target, {"version": "v2"})


def test_is_current_rejects_moved_windows(tmp_path: Path) -> None:
    """Extraction is skipped only when the window coordinates still match."""
    module = load_module()
    target = tmp_path / "scene.npz"
    old = np.array([[0.0, 0.0, 0.9, 100.0, 200.0]])
    new = old.copy()
    new[0, 3] += 10.0

    module._save_npz(target, version=np.array("v1"), windows=old, source=np.array("aef"))

    assert module._is_current(target, {"version": "v1", "windows": old, "source": "aef"})
    assert not module._is_current(target, {"version": "v1", "windows": new, "source": "aef"})


def test_is_current_rejects_a_ten_metre_utm_shift(tmp_path: Path) -> None:
    """A 10 m northing change at UTM magnitude must fail despite default rtol."""
    module = load_module()
    target = tmp_path / "scene.npz"
    old = np.array([[0.0, 0.0, 0.9, 500000.0, 5_500_000.0]])
    new = old.copy()
    new[0, 4] += 10.0

    module._save_npz(target, version=np.array("v1"), windows=old, source=np.array("aef"))

    assert module._is_current(target, {"version": "v1", "windows": old, "source": "aef"})
    assert not module._is_current(target, {"version": "v1", "windows": new, "source": "aef"})


def test_is_current_rejects_a_truncated_file(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "scene.npz"
    target.write_bytes(b"PK\x03\x04truncated")

    assert not module._is_current(target, {"version": "v1"})


def test_is_current_requires_a_version_field(tmp_path: Path) -> None:
    """Files written before the version tag existed must be recomputed."""
    module = load_module()
    target = tmp_path / "scene.npz"
    np.savez_compressed(target, features=np.zeros((1, 2)))

    assert not module._is_current(target, {"version": "v1"})


def test_window_selection_skips_windows_below_the_valid_threshold() -> None:
    module = load_module()
    labels = np.ones((1200, 1200), dtype=np.uint8)
    labels[:, 600:] = 0  # right half unlabeled

    windows = module.pick_windows(labels, Scale())

    assert len(windows) <= 2
    for top, left, valid, _, _ in windows:
        assert valid >= module.MIN_VALID
        block = labels[top : top + module.WINDOW, left : left + module.WINDOW]
        assert (block != 0).mean() >= module.MIN_VALID
