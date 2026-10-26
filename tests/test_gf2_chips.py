"""T8 chip protocol: grid coverage and reproducible per-epoch draws (no data)."""

from __future__ import annotations

from rs_backbone_eval.gf2_chips import CHIP, Gf2Chips, epoch_seed, grid


def chips(sizes: list[tuple[int, int]]) -> Gf2Chips:
    rows = [
        {"group_id": f"g{index}", "image_paths": ["x"], "label_path": "y"}
        for index in range(len(sizes))
    ]
    return Gf2Chips(root=".", rows=rows, sizes=sizes)


def test_grid_tiles_without_overlap_and_drops_the_partial_edge() -> None:
    corners = grid(CHIP * 3 + 10, CHIP * 2 + 5)
    assert len(corners) == 6
    assert len(set(corners)) == 6
    assert max(col for col, _ in corners) == CHIP * 2


def test_full_grid_covers_every_scene() -> None:
    dataset = chips([(CHIP * 2, CHIP * 2), (CHIP * 3, CHIP)])
    full = dataset.full_grid()
    assert len(full) == 4 + 3
    assert {chip.scene for chip in full} == {0, 1}


def test_draw_is_reproducible_and_changes_with_the_epoch() -> None:
    dataset = chips([(CHIP * 10, CHIP * 10)] * 3)
    first = dataset.draw(8, epoch_seed(42, 0))
    assert first == dataset.draw(8, epoch_seed(42, 0))
    assert first != dataset.draw(8, epoch_seed(42, 1))
    assert len(first) == 24
    # Without replacement within a scene.
    assert len(set(first)) == 24


def test_draw_takes_all_chips_of_a_small_scene() -> None:
    dataset = chips([(CHIP * 2, CHIP)])
    assert len(dataset.draw(32, 0)) == 2
