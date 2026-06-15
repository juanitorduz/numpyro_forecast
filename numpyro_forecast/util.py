"""Utility helpers: array shaping, distribution surgery, and seasonal features.

The distribution helpers (:func:`shift_loc`, :func:`slice_time`,
:func:`prefix_condition`) are implemented with :func:`functools.singledispatch`
so new distribution families can be registered without modifying call sites —
the functional analogue of Pyro's messenger-based dispatch.
"""

from functools import singledispatch

import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Float

from numpyro_forecast.typing import Array


def zero_data_like(data: Array, covariates: Array) -> Array:
    """Return zeros shaped like ``data`` but extended to the covariate duration.

    Mirrors Pyro's ``zero_data``: it exposes the shape/dtype of the data over the
    full forecast horizon without leaking observed values into the model.

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
    duration = covariates.shape[-2]
    shape = (*data.shape[:-2], duration, data.shape[-1])
    return jnp.zeros(shape, dtype=data.dtype)


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


@singledispatch
def shift_loc(noise_dist: dist.Distribution, loc: Array) -> dist.Distribution:
    """Re-center a zero-centered noise distribution at ``loc``.

    This converts Pyro's ``obs = data - prediction`` idiom into an additive
    shift of the observation distribution's location.

    Parameters
    ----------
    noise_dist
        A zero-centered location-family distribution.
    loc
        The deterministic mean to add to the distribution's location.

    Returns
    -------
    dist.Distribution
        A distribution centered at ``loc``.

    Raises
    ------
    NotImplementedError
        If ``noise_dist`` is of an unsupported type.
    """
    msg = f"shift_loc() does not support {type(noise_dist).__name__}"
    raise NotImplementedError(msg)


@shift_loc.register
def _(noise_dist: dist.Normal, loc: Array) -> dist.Distribution:
    return dist.Normal(loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.StudentT, loc: Array) -> dist.Distribution:
    return dist.StudentT(df=noise_dist.df, loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.Independent, loc: Array) -> dist.Distribution:
    base = shift_loc(noise_dist.base_dist, loc)
    return dist.Independent(base, noise_dist.reinterpreted_batch_ndims)


@singledispatch
def slice_time(noise_dist: dist.Distribution, index: slice) -> dist.Distribution:
    """Slice an elementwise distribution along the time axis ``-2``.

    The default implementation handles distributions with empty ``event_shape``
    whose ``batch_shape`` ends with ``(time, obs)`` (e.g. ``Normal``,
    ``StudentT``) by slicing each broadcast parameter.

    Parameters
    ----------
    noise_dist
        The distribution to slice.
    index
        A ``slice`` applied to the time axis ``-2`` of the batch shape.

    Returns
    -------
    dist.Distribution
        The same distribution family restricted to the selected time steps.

    Raises
    ------
    NotImplementedError
        If the distribution has a non-empty event shape.
    """
    if noise_dist.event_shape != ():
        msg = (
            f"slice_time() default does not support {type(noise_dist).__name__} "
            f"with event_shape {noise_dist.event_shape}"
        )
        raise NotImplementedError(msg)
    batch = noise_dist.batch_shape
    params = {
        name: jnp.broadcast_to(getattr(noise_dist, name), batch)[..., index, :]
        for name in type(noise_dist).arg_constraints
    }
    return type(noise_dist)(**params)


@slice_time.register
def _(noise_dist: dist.Independent, index: slice) -> dist.Distribution:
    base = slice_time(noise_dist.base_dist, index)
    return dist.Independent(base, noise_dist.reinterpreted_batch_ndims)


@singledispatch
def prefix_condition(noise_dist: dist.Distribution, data: Array) -> dist.Distribution:
    """Condition a ``(t+f)``-length distribution on a ``t``-length data prefix.

    For independent-over-time noise (the default) the conditional reduces to the
    forecast-horizon marginal, i.e. a time slice ``[t:]``. Correlated families
    (e.g. ``MultivariateNormal``) can register a genuine Gaussian conditional.

    Parameters
    ----------
    noise_dist
        The observation distribution over the full horizon ``(*batch, t+f, obs)``.
    data
        The observed prefix with shape ``(*batch, t, obs)``.

    Returns
    -------
    dist.Distribution
        The forecast-horizon distribution over ``(*batch, f, obs)``.
    """
    t = data.shape[-2]
    return slice_time(noise_dist, slice(t, None))


def fourier_features(
    duration: int,
    period: float,
    num_terms: int,
) -> Float[Array, " duration two_num_terms"]:
    """Build a Fourier seasonality design matrix.

    Parameters
    ----------
    duration
        Number of time steps.
    period
        Seasonal period (in time steps).
    num_terms
        Number of harmonics; the output has ``2 * num_terms`` columns
        (sine then cosine).

    Returns
    -------
    Float[Array, "duration two_num_terms"]
        The design matrix of shape ``(duration, 2 * num_terms)``.
    """
    time = jnp.arange(duration)[:, None]
    harmonics = jnp.arange(1, num_terms + 1)[None, :]
    angles = 2.0 * jnp.pi * harmonics * time / period
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def periodic_repeat(x: Array, duration: int, *, axis: int = -1) -> Array:
    """Tile a seasonal pattern to cover ``duration`` time steps.

    Parameters
    ----------
    x
        Seasonal pattern; the repeated axis has length equal to the period.
    duration
        Target length along ``axis``.
    axis
        Axis to repeat along (defaults to ``-1``).

    Returns
    -------
    Array
        ``x`` periodically repeated to length ``duration`` along ``axis``.
    """
    period = x.shape[axis]
    indices = jnp.arange(duration) % period
    return jnp.take(x, indices, axis=axis)
