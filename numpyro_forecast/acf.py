"""Batched autocorrelation and partial autocorrelation diagnostics.

This module provides the empirical autocorrelation function (:func:`acf`) and
partial autocorrelation function (:func:`pacf`) used to identify ARMA-style
dependence structure in time series. The ACF uses the FFT-based algorithm from
:func:`numpyro.diagnostics.autocorrelation` (itself adapted from Stan), ported
to :mod:`jax.numpy` so it is jittable and batched; the PACF applies the
Durbin-Levinson recursion to the ACF.

Unlike the model-data layout used elsewhere in the package (time at axis
``-2``), these functions reduce over time and return a lag spectrum, so they
take time on the **last** axis: inputs are shaped ``(*batch, time)`` and
outputs ``(*batch, max_lag + 1)``, with lag ``0`` (identically ``1.0``)
included. For package-layout data shaped ``(*batch, time, obs)``, pass
``jnp.moveaxis(data, -2, -1)`` to get per-series spectra ``(*batch, obs,
max_lag + 1)``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from numpyro.diagnostics import _fft_next_fast_len

from numpyro_forecast.typing import Array


@partial(jax.jit, static_argnames=("max_lag",))
def _acf(
    y: Float[Array, " *batch time"],
    *,
    max_lag: int,
) -> Float[Array, " *batch lags"]:
    """Jitted ACF core; the ``max_lag`` bounds guard lives in :func:`acf`."""
    m2 = 2 * _fft_next_fast_len(y.shape[-1])
    y_centered = y - y.mean(axis=-1, keepdims=True)
    freq = jnp.fft.rfft(y_centered, n=m2, axis=-1)
    autocov = jnp.fft.irfft(freq * jnp.conj(freq), n=m2, axis=-1)[..., : max_lag + 1]
    return autocov / autocov[..., :1]


@partial(jax.jit, static_argnames=("max_lag",))
def _pacf(
    y: Float[Array, " *batch time"],
    *,
    max_lag: int,
) -> Float[Array, " *batch lags"]:
    """Jitted PACF core; the ``max_lag`` bounds guard lives in :func:`pacf`."""
    rho = _acf(y, max_lag=max_lag)
    # Durbin-Levinson recursion over lags k = 1..max_lag. ``max_lag`` is static,
    # so the Python loop unrolls at trace time with static slice bounds and every
    # step is a batched array op; a lax.scan variant would need masked dot
    # products and only pays off for max_lag in the hundreds.
    phi = jnp.zeros_like(rho)
    # rho[..., 0] is exactly 1.0 for a non-degenerate series and NaN for a
    # constant one; reusing it (rather than a literal 1.0) propagates the NaN.
    out = [rho[..., 0]]
    for k in range(1, max_lag + 1):
        # At k == 1 the slices are empty, the sums are 0, and phi_kk = rho[..., 1].
        prev = phi[..., 1:k]
        numerator = rho[..., k] - jnp.sum(prev * rho[..., k - 1 : 0 : -1], axis=-1)
        denominator = 1.0 - jnp.sum(prev * rho[..., 1:k], axis=-1)
        phi_kk = numerator / denominator
        reflected = prev - phi_kk[..., None] * phi[..., k - 1 : 0 : -1]
        phi = phi.at[..., 1:k].set(reflected).at[..., k].set(phi_kk)
        out.append(phi_kk)
    return jnp.stack(out, axis=-1)


def _validate_max_lag(name: str, time: int, max_lag: int) -> None:
    """Raise ``ValueError`` unless ``1 <= max_lag < time``."""
    if not 1 <= max_lag < time:
        msg = f"{name} requires 1 <= max_lag < time length {time}, got max_lag={max_lag}"
        raise ValueError(msg)


def acf(
    y: Float[Array, " *batch time"] | Float[np.ndarray, " *batch time"],
    max_lag: int,
) -> Float[Array, " *batch lags"]:
    r"""Compute the empirical autocorrelation function up to ``max_lag``.

    The lag-:math:`k` autocorrelation of a series :math:`y_1, \ldots, y_T` with
    sample mean :math:`\bar{y}` is estimated with the biased normalization

    .. math::

        \hat{\rho}_k =
        \frac{\sum_{t=k+1}^{T} (y_t - \bar{y})(y_{t-k} - \bar{y})}
             {\sum_{t=1}^{T} (y_t - \bar{y})^2},

    computed for all lags at once via the FFT of the centered series (the
    Wiener-Khinchin identity), so the cost is :math:`O(T \log T)` instead of
    :math:`O(T \, k_{\max})`. The computation runs along the last axis and
    broadcasts over any leading batch axes.

    Parameters
    ----------
    y
        Time series with time on the last axis, shape ``(*batch, time)``.
    max_lag
        Largest lag to evaluate; must satisfy ``1 <= max_lag < time``.

    Returns
    -------
    Float[Array, "*batch lags"]
        Autocorrelations for lags ``0, 1, ..., max_lag`` (``max_lag + 1``
        values; lag ``0`` is identically ``1.0``).

    Raises
    ------
    ValueError
        If ``max_lag`` is not in ``[1, time)``.

    Notes
    -----
    A constant series has zero variance, so the result is meaningless: all NaN
    in eager execution (a ``0 / 0``), while under :func:`jax.jit` the rounding
    of the mean can instead yield arbitrary finite values.

    References
    ----------
    Adapted from :func:`numpyro.diagnostics.autocorrelation`, itself adapted
    from the Stan implementation.
    """
    _validate_max_lag("acf", y.shape[-1], max_lag)
    return _acf(y, max_lag=max_lag)


def pacf(
    y: Float[Array, " *batch time"] | Float[np.ndarray, " *batch time"],
    max_lag: int,
) -> Float[Array, " *batch lags"]:
    r"""Compute the empirical partial autocorrelation function up to ``max_lag``.

    The lag-:math:`k` partial autocorrelation is the correlation between
    :math:`y_t` and :math:`y_{t-k}` after removing the linear effect of the
    intermediate observations :math:`y_{t-1}, \ldots, y_{t-k+1}`; it equals the
    last coefficient :math:`\phi_{kk}` of the best linear predictor of order
    :math:`k`. The coefficients are obtained from the empirical
    autocorrelations (see :func:`acf`) with the Durbin-Levinson recursion

    .. math::

        \phi_{kk} =
        \frac{\hat{\rho}_k - \sum_{j=1}^{k-1} \phi_{k-1,j}\, \hat{\rho}_{k-j}}
             {1 - \sum_{j=1}^{k-1} \phi_{k-1,j}\, \hat{\rho}_j}.

    The recursion runs along the last axis and broadcasts over any leading
    batch axes.

    Parameters
    ----------
    y
        Time series with time on the last axis, shape ``(*batch, time)``.
    max_lag
        Largest lag to evaluate; must satisfy ``1 <= max_lag < time``.

    Returns
    -------
    Float[Array, "*batch lags"]
        Partial autocorrelations for lags ``0, 1, ..., max_lag``
        (``max_lag + 1`` values; lag ``0`` is identically ``1.0``).

    Raises
    ------
    ValueError
        If ``max_lag`` is not in ``[1, time)``.

    Notes
    -----
    A constant series has zero variance, so the result is meaningless (see
    :func:`acf`); a near-deterministic series can likewise overflow the
    recursion (the denominator above approaches zero) and produce infinities
    or NaNs.

    References
    ----------
    James Durbin (1960). "The Fitting of Time-Series Models". *Revue de
    l'Institut International de Statistique*.
    """
    _validate_max_lag("pacf", y.shape[-1], max_lag)
    return _pacf(y, max_lag=max_lag)
