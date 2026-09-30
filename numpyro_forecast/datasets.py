"""Dataset helpers for the example notebooks.

The BART loaders are thin wrappers around
`numpyro.examples.datasets.load_bart_od()`; `load_victoria_electricity()`
reads a small bundled CSV. Both return arrays in the package convention (time at
axis ``-2``). `load_breakfast_at_the_frat()` downloads the dunnhumby scanner
panel once and returns its three sheets as polars frames.
"""

import hashlib
import importlib.resources
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array

if TYPE_CHECKING:
    import polars

HOURS_PER_WEEK = 24 * 7
BREAKFAST_AT_THE_FRAT_URL = "https://ndownloader.figshare.com/files/57937129"
"""The figshare copy of the dunnhumby *Breakfast at the Frat* workbook."""
BREAKFAST_AT_THE_FRAT_SHA256 = "61b1d77dd6d9298fed204cc231f2b853a4c7f79376cfc30231646e1e51d0daba"
"""SHA-256 digest of the workbook, checked on every load."""
DOWNLOAD_TIMEOUT = 60.0
"""Seconds a single socket read may block before a dataset download is abandoned."""
_BREAKFAST_SHEETS = {
    "transactions": "dh Transaction Data",
    "products": "dh Products Lookup",
    "stores": "dh Store Lookup",
}


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


@dataclass(frozen=True)
class BreakfastAtTheFrat:
    """The three sheets of the dunnhumby *Breakfast at the Frat* workbook as polars frames.

    Column names and string values are lowercase (see `load_breakfast_at_the_frat()`).

    Attributes
    ----------
    transactions
        One row per store, product and week: units, visits, households, spend, the
        shelf and base prices, and the promotion flags ``feature``, ``display`` and
        ``tpr_only``.
    products
        The product lookup keyed by ``upc``: description, manufacturer, category,
        sub-category and size.
    stores
        The store lookup keyed by ``store_id``, exactly as in the workbook: two store
        ids appear twice with different price-segment labels.
    """

    transactions: "polars.DataFrame"
    products: "polars.DataFrame"
    stores: "polars.DataFrame"


def load_breakfast_at_the_frat(cache_dir: str | Path | None = None) -> BreakfastAtTheFrat:
    """Load the dunnhumby *Breakfast at the Frat* scanner panel as polars frames.

    The panel holds 156 weeks of weekly unit sales, prices and promotion mechanics
    for 55 products (58 lookup rows) in 77 stores (79 lookup rows). The loader downloads the
    [figshare copy](https://doi.org/10.6084/m9.figshare.30121060) of the workbook
    (about 31 MB) into ``cache_dir`` on first use, verifies its SHA-256 digest on
    every call, reads the transaction, product and store sheets with polars'
    calamine engine (every sheet carries a title row above the header), and
    lowercases every column name and every string value, so the examples work with
    one case. The figshare record declares a CC BY 4.0 license for the copy;
    dunnhumby's own terms govern the data.

    Parameters
    ----------
    cache_dir
        Directory that holds the cached workbook. Defaults to
        ``~/.cache/numpyro_forecast``.

    Returns
    -------
    BreakfastAtTheFrat
        The three sheets as polars frames with lowercase column names and string
        values.

    Raises
    ------
    ValueError
        If the digest of the cached workbook does not match the recorded one.
    ImportError
        If ``polars`` or ``fastexcel`` is not installed
        (``pip install numpyro_forecast[dataframes]``).
    """
    workbook = _ensure_breakfast_workbook(cache_dir)
    return _read_breakfast_workbook(workbook)


def _default_cache_dir() -> Path:
    """Return the directory that caches the downloaded datasets."""
    return Path.home() / ".cache" / "numpyro_forecast"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file as a hex string."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_breakfast_workbook(cache_dir: str | Path | None) -> Path:
    """Return the cached workbook path, downloading the file once and checking its digest."""
    directory = _default_cache_dir() if cache_dir is None else Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    workbook = directory / "breakfast_at_the_frat.xlsx"
    if not workbook.exists():
        # Download next to the target and move it into place only when the digest matches,
        # so an interrupted download never poisons the cache. The timeout bounds each socket
        # read, not the whole transfer, so a stalled connection raises instead of blocking.
        partial = workbook.with_name(workbook.name + ".part")
        with (
            urllib.request.urlopen(
                BREAKFAST_AT_THE_FRAT_URL, timeout=DOWNLOAD_TIMEOUT
            ) as response,
            partial.open("wb") as target,
        ):
            shutil.copyfileobj(response, target)
        digest = _sha256(partial)
        if digest != BREAKFAST_AT_THE_FRAT_SHA256:
            partial.unlink()
            msg = f"unexpected digest {digest} of the downloaded workbook; the file was discarded"
            raise ValueError(msg)
        partial.replace(workbook)
    digest = _sha256(workbook)
    if digest != BREAKFAST_AT_THE_FRAT_SHA256:
        msg = (
            f"unexpected workbook digest {digest} for {workbook}; delete the file to download "
            "it again"
        )
        raise ValueError(msg)
    return workbook


def _read_breakfast_workbook(workbook: Path) -> BreakfastAtTheFrat:
    """Read the three sheets and lowercase their column names and string values."""
    polars = require("polars", extra="dataframes")
    require("fastexcel", extra="dataframes")
    sheets = polars.read_excel(
        workbook,
        sheet_name=list(_BREAKFAST_SHEETS.values()),
        engine="calamine",
        read_options={"header_row": 1},
    )
    frames = {
        key: sheets[name]
        .rename(str.lower)
        .with_columns(polars.col(polars.String).str.to_lowercase())
        for key, name in _BREAKFAST_SHEETS.items()
    }
    return BreakfastAtTheFrat(**frames)
