"""iSAID I/O helpers shared by the segmentation workers.

The palette and the band handling used to live in the LIM worker; a 4-band and
then a 1-band DOTA tile crashed two call sites in a row, so they are now one
module with tests (review 2026-10-29). rasterio is imported lazily: the control
plane and these tests never need it.
"""

from __future__ import annotations

import numpy as np

from rs_backbone_eval.data.isaid_io import PALETTE, colour_lookup, rgb_only


def test_rgb_only_accepts_every_dota_band_count() -> None:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    assert rgb_only(rgb).shape == (4, 5, 3)

    rgba = np.dstack([np.full((4, 5, 3), 7, np.uint8), np.full((4, 5, 1), 255, np.uint8)])
    converted = rgb_only(rgba)
    assert converted.shape == (4, 5, 3)
    assert converted[0, 0].tolist() == [7, 7, 7]

    gray = np.full((4, 5, 1), 9, dtype=np.uint8)
    assert rgb_only(gray)[0, 0].tolist() == [9, 9, 9]

    gray_alpha = np.dstack([np.full((4, 5, 1), 4, np.uint8), np.full((4, 5, 1), 0, np.uint8)])
    assert rgb_only(gray_alpha)[0, 0].tolist() == [4, 4, 4]

    with np.testing.assert_raises(ValueError):
        rgb_only(np.zeros((4, 5, 5), dtype=np.uint8))


def test_palette_maps_every_class_exactly_once() -> None:
    lut = colour_lookup()
    assert lut.shape == (256, 256, 256)
    for (red, green, blue), klass in PALETTE.items():
        assert lut[red, green, blue] == klass
    assert sorted(PALETTE.values()) == list(range(16))
    assert len(set(PALETTE)) == 16  # no colour claimed by two classes


def test_background_is_black_and_class_ids_follow_the_categories() -> None:
    assert PALETTE[(0, 0, 0)] == 0
    # iSAID_train.json: 1 storage_tank ... 15 Helicopter
    assert PALETTE[(0x00, 0x3F, 0x3F)] == 1
    assert PALETTE[(0x00, 0x00, 0xBF)] == 15
