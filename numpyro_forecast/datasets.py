"""BART ridership dataset helpers.

Thin wrappers around :func:`numpyro.examples.datasets.load_bart_od` that return
arrays in the package convention (time at axis ``-2``) for the two examples.
"""

import jax.numpy as jnp
from jaxtyping import Float

from numpyro_forecast.typing import Array

HOURS_PER_WEEK = 24 * 7


def bart_available() -> bool:
    """Return whether the BART dataset can be loaded (download succeeds).

    Returns
    -------
    bool
        ``True`` if :func:`load_bart_od` loads without error.
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
