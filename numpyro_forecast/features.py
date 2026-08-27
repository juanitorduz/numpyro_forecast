"""Seasonal feature builders: Fourier design matrices and periodic tiling.

Both helpers run host-side, outside any trace; `fourier_features()`
memoizes its design matrix per argument tuple.
"""

from functools import lru_cache

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from numpyro_forecast.typing import Array


@lru_cache(maxsize=128)
def _fourier_features(
    duration: int,
    period: float,
    num_terms: int,
) -> Float[Array, " duration two_num_terms"]:
    """Memoized Fourier-feature core.

    The design matrix is fully determined by ``(duration, period, num_terms)``
    and is built host-side, never inside a trace, so the result is cached per
    argument tuple rather than recomputed (see `fourier_features()`).
    """
    time = jnp.arange(duration)[:, None]
    harmonics = jnp.arange(1, num_terms + 1)[None, :]
    angles = 2.0 * jnp.pi * harmonics * time / period
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


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
    return _fourier_features(duration, float(period), num_terms)


def periodic_repeat(x: ArrayLike, duration: int, *, axis: int = -1) -> Array:
    """Tile a seasonal pattern to cover ``duration`` time steps.

    Parameters
    ----------
    x
        Seasonal pattern; the repeated axis has length equal to the period.
        Accepts any array-like (e.g. a raw ``numpyro.sample`` draw).
    duration
        Target length along ``axis``.
    axis
        Axis to repeat along (defaults to ``-1``).

    Returns
    -------
    Array
        ``x`` periodically repeated to length ``duration`` along ``axis``.
    """
    array = jnp.asarray(x)
    period = array.shape[axis]
    indices = jnp.arange(duration) % period
    return jnp.take(array, indices, axis=axis)
