"""Array shaping helpers for the package's time-axis layout.

Arrays put time at axis ``-2`` and the observation dim at ``-1`` (matching
Pyro). These helpers build horizon-shaped zeros, concatenate in-sample and
forecast-horizon segments along the time axis, and pad an in-sample array with
constant forecast-horizon rows (the frozen-gate recipe of
`numpyro_forecast.models.ssoe()`).
"""

import jax.numpy as jnp

from numpyro_forecast.typing import Array


def _zeros_like_data(data: Array, duration: int) -> Array:
    """Return zeros shaped like ``data`` with the time axis ``-2`` set to ``duration``.

    Shared core of `zero_data_like()` and
    `numpyro_forecast.models.Horizon.zero_data`: it exposes the
    shape/dtype of the data over the full forecast horizon without leaking
    observed values into the model.
    """
    shape = (*data.shape[:-2], duration, data.shape[-1])
    return jnp.zeros(shape, dtype=data.dtype)


def zero_data_like(data: Array, covariates: Array) -> Array:
    """Return zeros shaped like ``data`` but extended to the covariate duration.

    Mirrors Pyro's ``zero_data``: it exposes the shape/dtype of the data over the
    full forecast horizon without leaking observed values into the model. The
    functional API exposes the equivalent value as
    `numpyro_forecast.models.Horizon.zero_data`.

    Parameters
    ----------
    data
        Observed data with time at axis ``-2``, shape ``(*batch, t, obs)``.
    covariates
        Covariates with time at axis ``-2``, shape ``(*batch, duration, cov)``.

    Returns
    -------
    Array
        Zeros of shape ``(*batch, duration, obs)``.
    """
    return _zeros_like_data(data, covariates.shape[-2])


def concat_future(prefix: Array, suffix: Array, *, axis: int = -2) -> Array:
    """Concatenate in-sample and forecast-horizon arrays along the time axis.

    Parameters
    ----------
    prefix
        In-sample array.
    suffix
        Forecast-horizon array (same shape as ``prefix`` except along ``axis``).
    axis
        Time axis to concatenate along (defaults to ``-2``).

    Returns
    -------
    Array
        The concatenation of ``prefix`` and ``suffix`` along ``axis``.
    """
    return jnp.concatenate([prefix, suffix], axis=axis)


def pad_future(x: Array, future: int, *, value: float = 0.0) -> Array:
    """Append ``future`` rows filled with ``value`` along the time axis.

    The frozen-gate recipe of `numpyro_forecast.models.ssoe()`: an update
    gate observed in-sample (``(*batch, t_obs, obs)``) becomes a full-horizon
    ``xs`` leaf whose forecast rows are ``value`` (``0.0`` freezes the carry,
    ``1.0`` keeps an availability mask open). Zero ``future`` returns ``x``
    unchanged in shape.

    Parameters
    ----------
    x
        In-sample array with time at axis ``-2``, shape ``(*batch, t_obs, obs)``.
    future
        Number of forecast rows to append.
    value
        Fill value of the appended rows, cast to ``x.dtype``.

    Returns
    -------
    Array
        ``x`` followed by ``future`` constant rows, shape ``(*batch, t_obs + future, obs)``.
    """
    suffix = jnp.full((*x.shape[:-2], future, x.shape[-1]), value, dtype=x.dtype)
    return concat_future(x, suffix)
