"""Array shaping helpers for the package's time-axis layout.

Arrays put time at axis ``-2`` and the observation dim at ``-1`` (matching
Pyro). These helpers build horizon-shaped zeros and concatenate in-sample and
forecast-horizon segments along the time axis.
"""

import jax.numpy as jnp

from numpyro_forecast.typing import Array


def _zeros_like_data(data: Array, duration: int) -> Array:
    """Return zeros shaped like ``data`` with the time axis ``-2`` set to ``duration``.

    Shared core of :func:`zero_data_like` and
    :attr:`numpyro_forecast.functional.models.Horizon.zero_data`: it exposes the
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
    :attr:`numpyro_forecast.functional.models.Horizon.zero_data`.

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
