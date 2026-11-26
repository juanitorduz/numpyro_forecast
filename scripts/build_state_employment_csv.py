"""Build ``numpyro_forecast/data/state_employment.csv`` from 51 FRED employment series.

Run once, with network access, from the repository root:

```bash
uv run python scripts/build_state_employment_csv.py
```

The script downloads the monthly, seasonally adjusted level of total nonfarm
employment (thousands of persons, BLS Current Employment Statistics, state and
area) for the 50 US states and the District of Columbia from FRED, where the
series id is the two-letter postal code followed by ``NA`` (``CANA`` for
California, ``DCNA`` for the District of Columbia), keeps the months from
1990-01 on that every series has published, and writes the levels with one
decimal, one column per state in alphabetical order of the postal code. The
script stops with an error when a series skips a month inside that range.
`numpyro_forecast.datasets.load_state_employment()` reads the file and turns
the levels into annualized monthly log growth rates, so the transformation
stays in tested package code. After a rebuild, update the vintage date and
the last month in the docstring of ``load_state_employment``.
"""

import concurrent.futures
import csv
import datetime as dt
import io
import urllib.request
from pathlib import Path

STATES = (
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA",
    "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS",
    "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
)  # fmt: skip
"""Postal codes of the 50 states and the District of Columbia, the column order."""

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
"""Download URL of one FRED series as CSV (``observation_date,<series_id>``)."""

FIRST_MONTH = dt.date(1990, 1, 1)
"""First month kept: the state series start in January 1990."""

OUTPUT = (
    Path(__file__).resolve().parent.parent / "numpyro_forecast" / "data" / "state_employment.csv"
)
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


def common_months(series: dict[str, dict[dt.date, float]]) -> list[dt.date]:
    """Consecutive months from `FIRST_MONTH` to the last month every series has published."""
    last = min(max(levels) for levels in series.values())
    months = []
    month = FIRST_MONTH

    while month <= last:
        months.append(month)
        month = next_month(month)

    for state, levels in series.items():
        missing = [month for month in months if month not in levels]
        if missing:
            msg = f"{state}NA skips {len(missing)} month(s) inside the range, first {missing[0]}"
            raise ValueError(msg)

    return months


def main() -> None:
    """Download the 51 series, align the months and write the CSV."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        levels = pool.map(lambda state: fetch_monthly(f"{state}NA"), STATES)
        series = dict(zip(STATES, levels, strict=True))
    months = common_months(series)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_date", *STATES])

        for month in months:
            writer.writerow([month.isoformat(), *(f"{series[s][month]:.1f}" for s in STATES)])


if __name__ == "__main__":
    main()
