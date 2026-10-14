"""Tests for the calendar helper used by the multi-temporal workers."""

from __future__ import annotations

import pytest

from rs_backbone_eval.timeutil import day_of_year


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
