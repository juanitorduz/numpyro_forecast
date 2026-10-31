"""Seasonal feature builders: Fourier design matrices and periodic tiling.

Both helpers can be called inside a model function, and so also run under a JAX
trace (`svi.run`, the jitted `~~numpyro_forecast.predictive.forecast()`).
`fourier_features()` caches a host NumPy copy of its design matrix per argument
tuple and converts it to a JAX array in the current default float dtype on every
call, so the cache never holds a tracer and the dtype follows the x64 flag.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Float

from numpyro_forecast.typing import Array


@lru_cache(maxsize=128)
def _fourier_features(
    duration: int,
    period: float,
    num_terms: int,
) -> Float[np.ndarray, " duration two_num_terms"]:
    """Memoized Fourier-feature core.

    The design matrix is fully determined by ``(duration, period, num_terms)``,
    so it is cached per argument tuple. It is built with NumPy in float64 and
    marked read-only: a NumPy array is never a JAX tracer, so the cached value
    does not depend on whether the first call ran inside a trace, and
    `fourier_features()` casts it to the default float dtype per call.
    """
    time = np.arange(duration, dtype=np.float64)[:, None]
    harmonics = np.arange(1, num_terms + 1, dtype=np.float64)[None, :]
    angles = 2.0 * np.pi * harmonics * time / period
    design = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    design.setflags(write=False)
    return design


def fourier_features(
    duration: int,
    period: float,
    num_terms: int,
) -> Float[Array, " duration two_num_terms"]:
    """Build a Fourier seasonality design matrix.

    Safe to call inside a model function (for example with ``h.duration`` of a
    `~~numpyro_forecast.models.Horizon`), also under `svi.run` or the jitted
    `~~numpyro_forecast.predictive.forecast()`. The matrix is computed in
    float64 and cached per argument tuple, then returned in the current
    default float dtype (``float64`` with x64 enabled, ``float32`` otherwise).

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
    design = _fourier_features(duration, float(period), num_terms)
    return jnp.asarray(design, dtype=jnp.result_type(float))


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
