"""Dataset helpers for the example notebooks.

The BART loaders are thin wrappers around
`numpyro.examples.datasets.load_bart_od()`; `load_victoria_electricity()`
and `load_us_inflation()` read small bundled CSVs; `load_m5()` downloads the
M5 competition files once into a cache directory and reads them into dense
arrays. All return arrays in the package convention (time at axis ``-2``).
"""

import contextlib
import hashlib
import importlib.resources
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array

if TYPE_CHECKING:
    import polars

HOURS_PER_WEEK = 24 * 7


def bart_available() -> bool:
    """Return whether the BART dataset can be loaded (download succeeds).

    Returns
    -------
    bool
        ``True`` if `load_bart_od()` loads without error.
    """
    try:
        _load_counts()
    except Exception:
        return False
    return True


def _load_counts() -> tuple[Array, list[str]]:
    """Load raw hourly origin-destination counts ``(time, origin, destin)``."""
    from numpyro.examples.datasets import load_bart_od

    dataset = load_bart_od()
    counts = jnp.asarray(dataset["counts"])
    stations = [str(name) for name in dataset["stations"]]
    return counts, stations


def load_bart_weekly() -> Float[Array, " weeks 1"]:
    """Load total weekly BART ridership (log scale) for the univariate example.

    Hourly counts are summed over all origin-destination pairs, aggregated into
    non-overlapping weeks, and log-transformed.

    Returns
    -------
    Float[Array, " weeks 1"]
        Log weekly totals with time at axis ``-2`` and a single observation dim.
    """
    counts, _ = _load_counts()
    hourly_total = counts.sum(axis=(1, 2))
    num_weeks = hourly_total.shape[0] // HOURS_PER_WEEK
    weekly = hourly_total[: num_weeks * HOURS_PER_WEEK]
    weekly = weekly.reshape(num_weeks, HOURS_PER_WEEK).sum(axis=1)
    return jnp.log(weekly)[:, None]


def load_bart_hierarchical(
    train_days: int = 90,
    test_weeks: int = 2,
) -> tuple[Float[Array, " origin time destin"], int, list[str]]:
    """Load the windowed hierarchical BART panel for the hierarchical example.

    The counts are ``log1p``-transformed and transposed to the
    ``(origin, time, destin)`` convention, then restricted to a ``train_days``
    training window followed by a ``test_weeks`` test window.

    Parameters
    ----------
    train_days
        Number of training days (24 hours each). Must be at least 1.
    test_weeks
        Number of test weeks (``24 * 7`` hours each). Must be non-negative;
        ``0`` returns the training window only, with ``split == y.shape[1]``.

    Returns
    -------
    y : Float[Array, " origin time destin"]
        Log counts over the train+test window with time at axis ``-2``.
    split : int
        Index along the time axis separating train from test.
    stations : list[str]
        Station names.

    Raises
    ------
    ValueError
        If ``train_days`` is less than 1 or ``test_weeks`` is negative (checked
        before the download), or if the requested ``train_days`` + ``test_weeks``
        window exceeds the available history (which would otherwise wrap a
        negative slice index).
    """
    if train_days < 1:
        msg = f"train_days must be at least 1, got {train_days}; pass a positive number of days"
        raise ValueError(msg)
    if test_weeks < 0:
        msg = (
            f"test_weeks must be non-negative, got {test_weeks}; "
            "pass 0 for no test window or a positive number of weeks"
        )
        raise ValueError(msg)
    counts, stations = _load_counts()
    log_counts = jnp.log1p(jnp.transpose(counts, (1, 0, 2)))
    t_total = log_counts.shape[1]
    t1 = t_total - test_weeks * HOURS_PER_WEEK
    t0 = t1 - train_days * 24
    if t0 < 0:
        msg = (
            f"requested window (train_days={train_days}, test_weeks={test_weeks}) "
            f"exceeds available history of {t_total} hours"
        )
        raise ValueError(msg)
    y = log_counts[:, t0:t_total, :]
    split = t1 - t0
    return y, split, stations


def load_victoria_electricity() -> tuple[Float[Array, " time 1"], Float[Array, " time"]]:
    """Load hourly Victoria (Australia) electricity demand and temperature.

    The series covers the first eight weeks of 2014, sampled hourly, from the
    Victoria electricity demand dataset used in the TensorFlow Probability
    structural-time-series case study and in Hyndman and Athanasopoulos'
    *Forecasting: Principles and Practice*. The original half-hourly data is
    downsampled to hourly by taking every other step. The values are bundled as a
    small CSV next to this module.

    Returns
    -------
    demand : Float[Array, " time 1"]
        Hourly electricity demand (GW) with time at axis ``-2`` and a single
        observation dimension.
    temperature : Float[Array, " time"]
        Hourly temperature (degrees Celsius), aligned with ``demand``.
    """
    source = importlib.resources.files("numpyro_forecast").joinpath(
        "data", "victoria_electricity.csv"
    )
    with source.open("r", encoding="utf-8") as handle:
        table = np.loadtxt(handle, delimiter=",", skiprows=1, dtype=np.float32)
    demand = jnp.asarray(table[:, 0])[:, None]
    temperature = jnp.asarray(table[:, 1])
    return demand, temperature


class USInflation(NamedTuple):
    r"""Quarterly US inflation for four price measures.

    Attributes
    ----------
    rates
        Annualized quarter-on-quarter log inflation in percent,
        $400 \, (\log P_t - \log P_{t-1})$, shape ``(quarters, measures)`` with time
        at axis ``-2``.
    dates
        First day of each quarter of ``rates`` as ``datetime64[D]``, starting at
        1959-04-01.
    names
        Column names of ``rates`` in order: ``("cpi", "core_cpi", "pce", "core_pce")``.
        Core PCE is last.
    """

    rates: Float[Array, " quarters measures"]
    dates: np.ndarray
    names: tuple[str, ...]


def load_us_inflation() -> USInflation:
    r"""Load quarterly US inflation for CPI, core CPI, PCE and core PCE.

    The bundled CSV holds quarterly averages of the monthly, seasonally adjusted
    price indexes ``CPIAUCSL`` (CPI, all items), ``CPILFESL`` (CPI less food and
    energy), ``PCEPI`` (PCE price index) and ``PCEPILFE`` (PCE less food and
    energy) from FRED, downloaded on 2026-11-08 by
    ``scripts/build_us_inflation_csv.py``. The levels run from 1959Q1 to 2026Q3,
    the last quarter with three published months for every series. They are
    revised data as published at that date, not real-time vintages. The BLS
    published no CPI for October 2025 (the federal government shutdown of 2025),
    so that month of the two CPI series is the geometric mean of September and
    November 2025 before the quarterly average, the approximation the BLS uses for
    that month in its seasonal adjustment. The BLS also carried the October 2025
    rents forward, so the rent and owners' equivalent rent indexes count no rent
    change for that month until the April 2026 index catches up (questions 2, 8
    and 12 of the
    [BLS shutdown FAQ](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)).
    The loader turns the levels into annualized quarterly log rates in percent,
    $400 \, (\log P_t - \log P_{t-1})$, so the rates start in 1959Q2. All four
    series are public domain (FRED tag *Public Domain: Citation Requested*; sources
    BLS and BEA).

    Returns
    -------
    USInflation
        The rates ``(quarters, measures)`` as ``float32``, the quarter dates and
        the measure names, core PCE last.
    """
    source = importlib.resources.files("numpyro_forecast").joinpath("data", "us_inflation.csv")
    with source.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        table = np.loadtxt(handle, delimiter=",", dtype=str)
    levels = table[:, 1:].astype(np.float64)
    rates = 400.0 * np.diff(np.log(levels), axis=0)
    dates = table[1:, 0].astype("datetime64[D]")
    return USInflation(
        rates=jnp.asarray(rates, dtype=jnp.float32), dates=dates, names=tuple(header[1:])
    )


M5_URL = (
    "https://github.com/Nixtla/m5-forecasts/raw/"
    "72b8e7fd3b565b3c538adcb1d1a05117d8562d7e/datasets/m5.zip"
)
"""Nixtla's mirror of the competition files, pinned to a commit so the digest below holds."""
M5_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
M5_FILES = (
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sales_test_evaluation.csv",
    "sell_prices.csv",
    "weights_evaluation.csv",
)
M5_KEYS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
M5_DAYS = 1_969
"""Training days 1 to 1,941 followed by the 28 evaluation days."""
M5_MARKER = ".m5-complete"
"""Name of the M5 cache marker: JSON with the archive digest and, once the extraction is
complete, the file sizes. A marker without ``"sizes"`` records an extraction in progress."""
DOWNLOAD_TIMEOUT = 60.0
"""Seconds a single socket read may block before a dataset download is abandoned."""


class M5Data(NamedTuple):
    """The M5 competition data as dense arrays plus the identifier and calendar tables.

    Attributes
    ----------
    sales
        Daily unit sales ``(days, series)`` over the 1,941 training days followed by the
        28 evaluation days, series in the order of the sales file.
    price
        Weekly shelf price of every series repeated over its days, ``NaN`` on the days
        the item was not listed.
    keys
        One row per series: ``id`` (``item_id`` and ``store_id`` joined by ``_``) and the
        five identifier columns ``item_id``, ``dept_id``, ``cat_id``, ``store_id``,
        ``state_id``.
    calendar
        One row per day with the ``date`` parsed, the Walmart week ``wm_yr_wk``, the
        weekday columns, the event names and types (``null`` when there is none) and the
        SNAP flags of the three states.
    weights
        The official evaluation weights of the 42,840 aggregates (``Level_id``,
        ``Agg_Level_1``, ``Agg_Level_2``, ``Dollar_Sales``, ``weight``).
    """

    sales: Float[np.ndarray, " days series"]
    price: Float[np.ndarray, " days series"]
    keys: "polars.DataFrame"
    calendar: "polars.DataFrame"
    weights: "polars.DataFrame"


def _default_cache_dir() -> Path:
    """Return the directory that caches the downloaded datasets."""
    return Path.home() / ".cache" / "numpyro_forecast"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file as a hex string, read in chunks."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _download_m5(archive: Path, url: str, digest: str) -> None:
    """Download the M5 archive to ``archive`` once its SHA-256 digest matches ``digest``.

    The archive streams into a ``.part`` file, which is removed on any error and replaces
    ``archive`` only when its digest matches, so an existing ``archive`` stays in place
    until the new one is verified. Two downloads into one cache folder at the same time
    are not supported. The timeout bounds each socket read, not the whole transfer.
    """
    if not url.startswith("https://"):
        msg = f"the M5 archive URL must use https, got {url!r}"
        raise ValueError(msg)
    partial = archive.with_name(archive.name + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response,  # noqa: S310
            partial.open("wb") as target,
        ):
            shutil.copyfileobj(response, target)
        actual = _sha256(partial)
        if actual != digest:
            msg = (
                f"unexpected digest {actual} of the downloaded M5 archive; the file was discarded"
            )
            raise ValueError(msg)
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)


def _write_m5_marker(directory: Path, content: dict[str, object]) -> None:
    """Write ``content`` as JSON to the M5 marker and sync it with ``os.fsync``."""
    with (directory / M5_MARKER).open("w") as file:
        json.dump(content, file)
        file.flush()
        os.fsync(file.fileno())


def _m5_archive_sizes(archive: Path) -> dict[str, int]:
    """Return the uncompressed size of every M5 file recorded in the archive."""
    with zipfile.ZipFile(archive) as archive_file:
        return {name: archive_file.getinfo(name).file_size for name in M5_FILES}


def _m5_sizes_match(directory: Path, sizes: Mapping[str, object]) -> bool:
    """Return whether every M5 file in ``directory`` exists and has the size in ``sizes``."""
    try:
        return all((directory / name).stat().st_size == sizes.get(name) for name in M5_FILES)
    except OSError:
        return False


def _m5_cache_is_complete(directory: Path, digest: str) -> bool:
    """Return whether the marker holds ``digest`` and every M5 file has its recorded size.

    A missing or unreadable marker, a marker without sizes, a missing file and a size
    mismatch all count as an incomplete cache.
    """
    try:
        marker = json.loads((directory / M5_MARKER).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(marker, dict) or not isinstance(sizes := marker.get("sizes"), dict):
        return False
    return marker.get("digest") == digest and _m5_sizes_match(directory, sizes)


def _ensure_m5_files(cache_dir: str | Path | None, url: str, digest: str) -> Path:
    """Return the directory with the M5 files, downloading and unpacking the archive once.

    The cache is complete when the marker file ``M5_MARKER`` holds ``digest`` and every
    file has the size that the archive records for it: the file names alone cannot tell
    a partial extraction from a complete one. Otherwise an ``m5.zip`` already in the
    directory is checked against ``digest`` and downloaded again on a mismatch.

    A directory that has never had a marker (a cache written before the marker existed)
    is adopted without a new extraction when its ``m5.zip`` matches ``digest`` and every
    file has the size the archive records: the marker is written if the directory is
    writable, and a read-only directory is checked in the same way on every call.

    Any other cache miss writes an in-progress marker (one without ``"sizes"``, which is
    what makes it incomplete) and syncs it, extracts the members in place, syncs every
    file and writes the complete marker last. An interrupted extraction therefore leaves
    an incomplete marker and is redone on the next call, also after a crash before the
    data reached the disk, as far as ``os.fsync`` reaches it: on macOS it does not flush
    the drive's write cache, and the directory entry of a new marker is not synced, so a
    power loss can still leave damaged files of the right size. Two extractions into one
    cache folder at the same time are not supported.
    """
    directory = _default_cache_dir() / "m5" if cache_dir is None else Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if _m5_cache_is_complete(directory, digest):
        return directory
    archive = directory / "m5.zip"
    if archive.is_file() and _sha256(archive) == digest:
        sizes = _m5_archive_sizes(archive)
        if not (directory / M5_MARKER).exists() and _m5_sizes_match(directory, sizes):
            try:
                _write_m5_marker(directory, {"digest": digest, "sizes": sizes})
            except OSError:
                with contextlib.suppress(OSError):
                    (directory / M5_MARKER).unlink(missing_ok=True)
            return directory
    else:
        _download_m5(archive, url, digest)
        sizes = _m5_archive_sizes(archive)
    _write_m5_marker(directory, {"digest": digest, "extracting": True})
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(directory, members=M5_FILES)
    for name in M5_FILES:
        with (directory / name).open("r+b") as file:
            os.fsync(file.fileno())
    _write_m5_marker(directory, {"digest": digest, "sizes": sizes})
    return directory


def load_m5(cache_dir: str | Path | None = None) -> M5Data:
    """Load the M5 competition data (download and cache once, then read the files).

    The files come from [Nixtla's mirror](https://github.com/Nixtla/m5-forecasts) of the
    competition data: the daily unit sales of the 30,490 series over the training period
    (days 1 to 1,941) and the 28 evaluation days released after the competition, the
    calendar with the events and the SNAP days, the weekly shelf prices, and the official
    evaluation weights. The 50 MB archive is downloaded once into ``cache_dir`` (default
    ``~/.cache/numpyro_forecast/m5``) and checked against a pinned SHA-256 digest, also
    when an ``m5.zip`` is already in ``cache_dir``. A full extraction of the verified
    archive ends with a marker that records the size of every file, and the cache is used
    only while that marker matches, so an interrupted download or extraction is redone on
    the next call, as is a file that was deleted or truncated later. A cache folder written
    before this check existed has no marker. It is used as is (also when it is read-only)
    when its ``m5.zip`` matches the digest and every file has the size that the archive
    records; the content of the files is not compared. Without a matching ``m5.zip`` the
    files cannot be verified: the archive is downloaded again, which fails without a
    network connection. To use files copied from elsewhere, put the matching ``m5.zip``
    next to them.

    Reading needs polars (``pip install numpyro_forecast[dataframes]``).

    Parameters
    ----------
    cache_dir
        Directory that holds the archive and the extracted files; created if missing.

    Returns
    -------
    M5Data
        Dense ``(days, series)`` sales and price arrays, the identifier table, the
        calendar and the official weights.
    """
    pl = require("polars", extra="dataframes")
    directory = _ensure_m5_files(cache_dir, M5_URL, M5_SHA256)
    keys = list(M5_KEYS)
    day_columns = [f"d_{day}" for day in range(1, M5_DAYS + 1)]
    sales_df = (
        pl.scan_csv(directory / "sales_train_evaluation.csv")
        .join(
            pl.scan_csv(directory / "sales_test_evaluation.csv"),
            on=keys,
            how="left",
            maintain_order="left",
        )
        .with_columns(id=pl.concat_str([pl.col("item_id"), pl.col("store_id")], separator="_"))
        .collect(engine="streaming")
    )
    keys_df = sales_df.select("id", *keys)
    sales = sales_df.select(day_columns).to_numpy().T.astype(np.float32)
    del sales_df
    calendar = pl.read_csv(
        directory / "calendar.csv", try_parse_dates=True, null_values=["NA"]
    ).head(M5_DAYS)
    prices = pl.scan_csv(directory / "sell_prices.csv")
    weeks = prices.select("wm_yr_wk").unique().sort("wm_yr_wk").collect(engine="streaming")
    positions = keys_df.lazy().with_row_index("n").select("n", "store_id", "item_id")
    weekly_price = (
        prices.join(positions, on=["store_id", "item_id"])
        .collect(engine="streaming")
        .pivot(on="wm_yr_wk", index="n", values="sell_price")
        .sort("n")
        .select([str(week) for week in weeks["wm_yr_wk"].to_list()])
        .to_numpy()
        .astype(np.float32)
    )
    week_of_day = calendar["wm_yr_wk"].rank("dense").cast(pl.Int64).to_numpy() - 1
    price = weekly_price[:, week_of_day].T
    weights = pl.read_csv(directory / "weights_evaluation.csv")
    return M5Data(sales=sales, price=price, keys=keys_df, calendar=calendar, weights=weights)
