"""Dataset helpers for the example notebooks.

The BART loaders are thin wrappers around
`numpyro.examples.datasets.load_bart_od()`; `load_victoria_electricity()`
reads a small bundled CSV; `load_m5()` downloads the M5 competition files once
into a cache directory and reads them into dense arrays. All return arrays in
the package convention (time at axis ``-2``).
"""

import hashlib
import importlib.resources
import shutil
import urllib.request
import zipfile
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
        Number of training days (24 hours each).
    test_weeks
        Number of test weeks (``24 * 7`` hours each).

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
        If the requested ``train_days`` + ``test_weeks`` window exceeds the
        available history (which would otherwise wrap a negative slice index).
    """
    counts, stations = _load_counts()
    log_counts = jnp.log1p(jnp.transpose(counts, (1, 0, 2)))
    t_total = log_counts.shape[1]
    t1 = t_total - test_weeks * HOURS_PER_WEEK
    t0 = t1 - train_days * 24
    if t0 < 0 or t1 <= t0:
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
    """Return the SHA-256 digest of a file as a hex string."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_m5_files(cache_dir: str | Path | None, url: str, digest: str) -> Path:
    """Return the directory with the M5 files, downloading and unpacking the archive once.

    The archive streams into a ``.part`` file and moves into place only when its SHA-256
    digest matches, so an interrupted download never poisons the cache. The timeout
    bounds each socket read, not the whole transfer.
    """
    directory = _default_cache_dir() / "m5" if cache_dir is None else Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if all((directory / name).exists() for name in M5_FILES):
        return directory
    archive = directory / "m5.zip"
    if not archive.exists():
        if not url.startswith("https://"):
            msg = f"the M5 archive URL must use https, got {url!r}"
            raise ValueError(msg)
        partial = archive.with_name(archive.name + ".part")
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response,  # noqa: S310
            partial.open("wb") as target,
        ):
            shutil.copyfileobj(response, target)
        actual = _sha256(partial)
        if actual != digest:
            partial.unlink()
            msg = (
                f"unexpected digest {actual} of the downloaded M5 archive; the file was discarded"
            )
            raise ValueError(msg)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(directory, members=M5_FILES)
    return directory


def load_m5(cache_dir: str | Path | None = None) -> M5Data:
    """Load the M5 competition data (download and cache once, then read the files).

    The files come from [Nixtla's mirror](https://github.com/Nixtla/m5-forecasts) of the
    competition data: the daily unit sales of the 30,490 series over the training period
    (days 1 to 1,941) and the 28 evaluation days released after the competition, the
    calendar with the events and the SNAP days, the weekly shelf prices, and the official
    evaluation weights. The 50 MB archive is downloaded once into ``cache_dir`` (default
    ``~/.cache/numpyro_forecast/m5``) and checked against a pinned SHA-256 digest.

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
