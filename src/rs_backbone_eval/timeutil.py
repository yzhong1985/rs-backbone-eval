"""Calendar helpers shared by the multi-temporal workers."""

from __future__ import annotations

import numpy as np

# PASTIS-HD time series span 2018-09 to 2019-10. Both AnySat's PASTIS loader
# and U-TAE's PASTIS setup position frames by days since this reference date,
# *not* by calendar day-of-year: a 13-month series has two 24 September frames
# and two 19 October frames, and day-of-year also jumps 322 -> 17 on 1 January.
PASTIS_REFERENCE_DATE = "2018-09-01"


def _as_date(stamp: str | int) -> np.datetime64:
    """Parse ``20190924`` / ``"2019-09-24"`` / ``"20190924"`` as a date."""
    text = str(stamp).strip()
    digits = text.replace("-", "")[:8]
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"not an ISO/compact date: {stamp!r}")
    return np.datetime64(f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}", "D")


def day_of_year(stamp: str | int) -> int:
    """Calendar day-of-year (1..366) for an ISO date; integers pass through.

    AnySat and Prithvi's temporal encodings expect the day *within the calendar
    year*. ``epoch_days % 365`` (the previous implementation) is not that: it
    drifts across leap years and is off by one for most dates.
    """
    if isinstance(stamp, int):
        return stamp
    date = _as_date(stamp)
    year = int(date.astype("datetime64[Y]").astype(int)) + 1970
    start = np.datetime64(f"{year}-01-01", "D")
    return int((date - start).astype(int)) + 1


def days_since_reference(stamp: str | int, reference: str = PASTIS_REFERENCE_DATE) -> int:
    """Whole days between ``stamp`` and ``reference`` (can be negative).

    A 13-month PASTIS series needs a monotone, non-repeating time axis. The
    positional encoders of AnySat's S2 LTAE (``T=1000``) and the vendored
    U-TAE LTAE2d are sinusoidal over the position value, so offsets larger than
    366 are fine; duplicate day-of-year values are what break them.
    """
    return int((_as_date(stamp) - _as_date(reference)).astype(int))
