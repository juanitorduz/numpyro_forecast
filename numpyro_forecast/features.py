"""Seasonal feature builders: Fourier design matrices and periodic tiling.

Both helpers can be called inside a model function, and so also run under a JAX
trace (`svi.run`, the jitted `~~numpyro_forecast.predictive.forecast()`).
`fourier_features()` caches a host NumPy copy of its design matrix per argument
tuple and converts it to a JAX array in the current default float dtype on every
call, so the cache never holds a tracer and the dtype follows the x64 flag.

The integer and real scalar arguments accept NumPy scalars as well as Python
ones (`~~numpyro_forecast.typing.IntScalar`,
`~~numpyro_forecast.typing.RealScalar`), for example a period or a duration
read from a NumPy array or a DataFrame. Both helpers convert them to Python
scalars at the boundary, so the memo is keyed by Python scalars only.
"""

import math
import operator
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Float

from numpyro_forecast.typing import Array, IntScalar, RealScalar


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
    duration: IntScalar,
    period: RealScalar,
    num_terms: IntScalar,
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
        Number of time steps, a non-negative Python or NumPy integer (not a
        ``float``).
    period
        Seasonal period (in time steps): a positive, finite Python or NumPy
        integer or float, for example ``7`` for daily data with a weekly season.
        A NumPy float is used at its exact binary value, so ``np.float32(0.1)``
        is the period ``0.10000000149011612``, not ``0.1``, and the two spellings
        are two memo entries.
    num_terms
        Number of harmonics, a non-negative Python or NumPy integer of at most
        ``floor(period / 2)``; the output has ``2 * num_terms`` columns (sine
        then cosine), none for ``0``. The matrix samples ``t = 0, 1, ...``, so
        harmonic ``k > period / 2`` is above the Nyquist frequency of that grid
        and aliases onto the lower frequency ``(period - k) / period``: same
        cosine samples, negated sine samples (harmonic ``period - k`` when
        ``period`` is an integer). For an even integer ``period``,
        ``num_terms == period / 2`` is valid but its last sine column is zero up
        to rounding; the matching cosine column alternates between ``1`` and
        ``-1``.

    Returns
    -------
    Float[Array, "duration two_num_terms"]
        The design matrix of shape ``(duration, 2 * num_terms)``.

    Raises
    ------
    ValueError
        If ``duration`` or ``num_terms`` is negative, ``period`` is not
        positive and finite (``0`` or ``nan`` would cache an all-``nan``
        matrix without an error, and ``inf`` a constant one), or ``num_terms``
        is larger than ``period / 2`` (aliased harmonics).
    """
    n_steps, cycle, n_terms = operator.index(duration), float(period), operator.index(num_terms)
    if n_steps < 0:
        msg = f"duration must be non-negative, got {n_steps}"
        raise ValueError(msg)
    if not (math.isfinite(cycle) and cycle > 0.0):
        msg = f"period must be positive and finite, got {cycle}"
        raise ValueError(msg)
    if n_terms < 0:
        msg = f"num_terms must be non-negative, got {n_terms}"
        raise ValueError(msg)
    if 2 * n_terms > cycle:
        msg = (
            f"num_terms must be at most {math.floor(cycle / 2)} for period {cycle}, "
            f"got {n_terms}: harmonics above period / 2 alias onto lower ones"
        )
        raise ValueError(msg)
    design = _fourier_features(n_steps, cycle, n_terms)
    return jnp.asarray(design, dtype=jnp.result_type(float))


def periodic_repeat(x: ArrayLike, duration: IntScalar, *, axis: int = -1) -> Array:
    """Tile a seasonal pattern to cover ``duration`` time steps.

    Parameters
    ----------
    x
        Seasonal pattern; the repeated axis has length equal to the period.
        Accepts any array-like (e.g. a raw ``numpyro.sample`` draw).
    duration
        Target length along ``axis``: a non-negative Python or NumPy integer
        (not a ``float``).
    axis
        Axis to repeat along (defaults to ``-1``).

    Returns
    -------
    Array
        ``x`` periodically repeated to length ``duration`` along ``axis``.

    Raises
    ------
    ValueError
        If ``duration`` is negative.
    """
    n_steps = operator.index(duration)
    if n_steps < 0:
        msg = f"duration must be non-negative, got {n_steps}"
        raise ValueError(msg)
    array = jnp.asarray(x)
    period = array.shape[axis]
    indices = jnp.arange(n_steps) % period
    return jnp.take(array, indices, axis=axis)
