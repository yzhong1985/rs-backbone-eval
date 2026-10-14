"""Calendar helpers shared by the multi-temporal workers."""

from __future__ import annotations

import numpy as np


def day_of_year(stamp: str | int) -> int:
    """Calendar day-of-year (1..366) for an ISO date; integers pass through.

    AnySat and Prithvi's temporal encodings expect the day *within the calendar
    year*. ``epoch_days % 365`` (the previous implementation) is not that: it
    drifts across leap years and is off by one for most dates.
    """
    if isinstance(stamp, int):
        return stamp
    date = np.datetime64(str(stamp)[:10], "D")
    year = int(date.astype("datetime64[Y]").astype(int)) + 1970
    start = np.datetime64(f"{year}-01-01", "D")
    return int((date - start).astype(int)) + 1
