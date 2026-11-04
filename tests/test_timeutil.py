"""Tests for the calendar helper used by the multi-temporal workers."""

from __future__ import annotations

import pytest

from rs_backbone_eval.timeutil import day_of_year, days_since_reference


def test_day_of_year_is_calendar_based() -> None:
    assert day_of_year("2018-01-01") == 1
    assert day_of_year("2018-12-31") == 365
    assert day_of_year("2020-12-31") == 366  # leap year
    assert day_of_year("2020-03-01") == 61


def test_day_of_year_passes_integers_through() -> None:
    assert day_of_year(7) == 7


def test_day_of_year_is_not_epoch_days_mod_365() -> None:
    # the previous implementation drifted: this date is not 4 modulo 365
    value = day_of_year("2018-04-27")
    assert value == 117
    with pytest.raises(Exception):  # noqa: B017 - sanity that the input must be a date
        day_of_year("not-a-date")


def test_days_since_reference_is_monotone_across_the_pastis_series() -> None:
    # PASTIS-HD spans 2018-09 .. 2019-10; day-of-year repeats after New Year.
    dates = [20180924, 20181019, 20190117, 20190610, 20190924, 20191019]
    offsets = [days_since_reference(day) for day in dates]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)
    assert offsets[0] == 23  # 2018-09-24 is 23 days after the reference
    # the two 24 September frames are a year apart, not the same day
    assert offsets[4] - offsets[0] == 365
    assert all(value >= 0 for value in offsets)
    # positions above 366 (the old period assumption) must be usable
    assert offsets[-1] > 366


def test_days_since_reference_rejects_bad_dates() -> None:
    with pytest.raises(ValueError):
        days_since_reference("2019-9")
