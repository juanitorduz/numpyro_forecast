"""Build ``numpyro_forecast/data/us_inflation.csv`` from four FRED price indexes.

Run once, with network access, from the repository root:

```bash
uv run python scripts/build_us_inflation_csv.py
```

The script downloads the monthly, seasonally adjusted index levels of
``CPIAUCSL`` (CPI, all items), ``CPILFESL`` (CPI less food and energy),
``PCEPI`` (PCE price index) and ``PCEPILFE`` (PCE less food and energy), fills
a month that a series skips inside its published range with the geometric mean
of the two neighboring months (see `fill_interior_gaps`), averages each series
over its complete quarters, keeps the quarters from 1959Q1 on that all four
series share, and writes the quarterly average levels with six decimals. The
last quarter of the file is the last quarter whose three months are published
for every series. `numpyro_forecast.datasets.load_us_inflation()` reads that
file and turns the levels into annualized quarterly log rates, so the
transformation stays in tested package code. After a rebuild, update the
vintage date and the list of filled months in the docstring of
``load_us_inflation``.
"""

import csv
import datetime as dt
import io
import urllib.request
from pathlib import Path

import numpy as np

SERIES = {"cpi": "CPIAUCSL", "core_cpi": "CPILFESL", "pce": "PCEPI", "core_pce": "PCEPILFE"}
"""CSV column name to FRED series id, in the column order of the file."""

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
"""Download URL of one FRED series as CSV (``observation_date,<series_id>``)."""

FIRST_QUARTER = dt.date(1959, 1, 1)
"""First quarter kept: the PCE indexes start in January 1959."""

MONTHS_PER_QUARTER = 3
"""Published (or filled) months a quarter needs to be written."""

OUTPUT = Path(__file__).resolve().parent.parent / "numpyro_forecast" / "data" / "us_inflation.csv"
"""Destination of the bundled CSV."""


def fetch_monthly(series_id: str) -> dict[dt.date, float]:
    """Download one FRED series and return ``{month: level}`` without missing values."""
    with urllib.request.urlopen(URL.format(series_id=series_id), timeout=60) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    date_column, value_column = rows[0].keys()
    levels: dict[dt.date, float] = {}

    for row in rows:
        if row[value_column] in {"", "."}:
            continue
        levels[dt.date.fromisoformat(row[date_column])] = float(row[value_column])

    return levels


def next_month(month: dt.date) -> dt.date:
    """First day of the month after ``month``."""
    return dt.date(month.year + month.month // 12, month.month % 12 + 1, 1)


def fill_interior_gaps(levels: dict[dt.date, float]) -> dict[dt.date, float]:
    """Fill each single missing month inside the published range of a series.

    The BLS published no CPI for October 2025 (the federal government shutdown
    of 2025), so ``CPIAUCSL`` and ``CPILFESL`` skip that month. The filled value
    is the geometric mean of the previous and the next published month, which
    splits the two-month log change evenly, as the BLS did in its November 2025
    release. A gap of two or more consecutive months raises, because a straight
    line through a longer gap is not a published statistic any more.

    Parameters
    ----------
    levels
        ``{month: level}`` of one series, with missing months absent.

    Returns
    -------
    dict[datetime.date, float]
        ``levels`` plus the filled months.

    Raises
    ------
    ValueError
        If two or more consecutive months are missing inside the range.
    """
    filled = dict(levels)
    month = min(levels)

    while month < max(levels):
        following = next_month(month)
        if following not in levels:
            after = next_month(following)
            if after not in levels:
                msg = f"two or more consecutive months missing from {following}"
                raise ValueError(msg)
            filled[following] = float(np.sqrt(levels[month] * levels[after]))
        month = following

    return filled


def quarterly_average(levels: dict[dt.date, float]) -> dict[dt.date, float]:
    """Average monthly levels over each complete quarter, keyed by its first day."""
    months: dict[dt.date, list[float]] = {}

    for month, level in levels.items():
        quarter = dt.date(month.year, 3 * ((month.month - 1) // 3) + 1, 1)
        months.setdefault(quarter, []).append(level)

    return {
        quarter: float(np.mean(values))
        for quarter, values in months.items()
        if len(values) == MONTHS_PER_QUARTER
    }


def main() -> None:
    """Download the four series, fill the gaps, align the quarters and write the CSV."""
    tables = {
        name: quarterly_average(fill_interior_gaps(fetch_monthly(series_id)))
        for name, series_id in SERIES.items()
    }
    quarters = sorted(set.intersection(*(set(table) for table in tables.values())))
    quarters = [quarter for quarter in quarters if quarter >= FIRST_QUARTER]

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quarter", *SERIES])

        for quarter in quarters:
            writer.writerow(
                [quarter.isoformat(), *(f"{tables[name][quarter]:.6f}" for name in SERIES)]
            )


if __name__ == "__main__":
    main()
