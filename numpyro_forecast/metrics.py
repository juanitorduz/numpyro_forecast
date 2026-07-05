"""Probabilistic forecast metrics.

This module ports :func:`pyro.ops.stats.crps_empirical` to JAX and adds the
pinball (quantile) loss, the Winkler interval score, and a MASE metric factory.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float

from numpyro_forecast.typing import Array, Metric


@jax.jit
def _crps_empirical(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    """Jitted CRPS core; the ``>= 2`` samples guard lives in :func:`crps_empirical`."""
    num_samples = pred.shape[0]
    pred_sorted = jnp.sort(pred, axis=0)
    diff = pred_sorted[1:] - pred_sorted[:-1]
    # Build the rank weights i * (n - i) in the data dtype. The cast must precede
    # the multiply: an int32 * int32 product overflows (to negative values) for
    # large sample counts, and casting the overflowed result would not recover it.
    lower = jnp.arange(1, num_samples, dtype=pred.dtype)
    upper = jnp.arange(num_samples - 1, 0, -1, dtype=pred.dtype)
    weight = (lower * upper).reshape((num_samples - 1,) + (1,) * (diff.ndim - 1))
    absolute_error = jnp.abs(pred - truth).mean(axis=0)
    # Normalize in the data dtype too: a Python ``num_samples ** 2`` constant
    # overflows int32 inside the jitted kernel once ``num_samples`` exceeds ~46k.
    return absolute_error - (diff * weight).sum(axis=0) / jnp.asarray(num_samples, pred.dtype) ** 2


def crps_empirical(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    r"""Compute the empirical Continuous Ranked Probability Score (CRPS).

    The CRPS generalises the mean absolute error to probabilistic forecasts and
    is computed elementwise as

    .. math::

        \mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac{1}{2}\,\mathbb{E}|X - X'|,

    where :math:`X, X'` are independent draws from the forecast distribution
    :math:`F`. The expectations are estimated from the forecast ``sample`` axis
    using the sorted-sample :math:`O(n \log n)` identity.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first, shape ``(sample, *batch)``.
    truth
        Ground-truth values with shape ``(*batch)`` (broadcastable to ``pred``).

    Returns
    -------
    Float[Array, "*batch"]
        Elementwise CRPS, one value per ``batch`` location.

    References
    ----------
    Tilmann Gneiting, Adrian E. Raftery (2007). "Strictly Proper Scoring Rules,
    Prediction, and Estimation". *Journal of the American Statistical
    Association*.
    """
    num_samples = pred.shape[0]
    if num_samples < 2:
        msg = f"crps_empirical needs at least 2 samples, got {num_samples}"
        raise ValueError(msg)
    return _crps_empirical(pred, truth)


@partial(jax.jit, static_argnums=(2,))
def _pinball(pred: Array, truth: Array, quantile: float) -> Array:
    """Jitted mean pinball loss at ``quantile`` (static so the branch specializes)."""
    estimate = jnp.quantile(pred, quantile, axis=0)
    diff = truth - estimate
    return jnp.maximum(quantile * diff, (quantile - 1.0) * diff).mean()


def eval_pinball(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    quantile: float = 0.5,
) -> float:
    r"""Mean pinball (quantile) loss of the forecast ``quantile``.

    The pinball loss for the forecast :math:`\hat q` of quantile :math:`\tau` is
    :math:`\max(\tau (y - \hat q), (\tau - 1)(y - \hat q))`, averaged over all
    data elements. At ``quantile=0.5`` it is half the mean absolute error.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).
    quantile
        Target quantile in ``(0, 1)``.

    Returns
    -------
    float
        The mean pinball loss.

    Raises
    ------
    ValueError
        If ``quantile`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < quantile < 1.0:
        msg = f"quantile must be in (0, 1), got {quantile}"
        raise ValueError(msg)
    return float(_pinball(pred, truth, quantile))


@partial(jax.jit, static_argnums=(2,))
def _interval_score(pred: Array, truth: Array, alpha: float) -> Array:
    """Jitted mean Winkler interval score for the central ``alpha`` interval."""
    tail = (1.0 - alpha) / 2.0
    lo = jnp.quantile(pred, tail, axis=0)
    hi = jnp.quantile(pred, 1.0 - tail, axis=0)
    penalty = 2.0 / (1.0 - alpha)
    below = penalty * (lo - truth) * (truth < lo)
    above = penalty * (truth - hi) * (truth > hi)
    return (hi - lo + below + above).mean()


def eval_interval_score(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    alpha: float = 0.9,
) -> float:
    r"""Mean Winkler interval score for the central ``alpha`` prediction interval.

    For the central ``alpha`` interval :math:`[l, u]` (the :math:`(1-\alpha)/2`
    and :math:`1-(1-\alpha)/2` quantiles), the interval score is
    :math:`(u - l) + \tfrac{2}{1-\alpha}\big[(l - y)\mathbf 1_{y<l} + (y -
    u)\mathbf 1_{y>u}\big]`, averaged over all data elements. It rewards narrow
    intervals and penalizes ground truth falling outside them; lower is better.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).
    alpha
        Nominal interval level in ``(0, 1)``.

    Returns
    -------
    float
        The mean interval score.

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    return float(_interval_score(pred, truth, alpha))


def make_mase(train_data: Float[Array, "*batch time obs_dim"], *, seasonality: int = 1) -> Metric:
    """Build a Mean Absolute Scaled Error metric scaled by ``train_data``.

    MASE divides the forecast MAE (using the sample median as point estimate) by
    the in-sample MAE of the seasonal-naive forecast on ``train_data``,
    ``mean(|y_t - y_{t-seasonality}|)``. The scale is computed once at factory
    time; the returned metric has the standard ``(pred, truth) -> float``
    signature.

    Parameters
    ----------
    train_data
        Training data with time at axis ``-2``; leading batch axes are allowed
        and the seasonal-naive scale is averaged over all axes.
    seasonality
        Seasonal period (``>= 1``); ``1`` is the random-walk naive baseline.

    Returns
    -------
    Metric
        A ``(pred, truth) -> float`` callable computing MASE.

    Raises
    ------
    ValueError
        If ``seasonality < 1``, ``train_data`` is not longer than
        ``seasonality`` along the time axis, or the seasonal-naive scale is zero
        (a constant series, for which MASE is undefined).
    """
    if seasonality < 1:
        msg = f"seasonality must be >= 1, got {seasonality}"
        raise ValueError(msg)
    if train_data.shape[-2] <= seasonality:
        msg = (
            f"train_data must be longer than seasonality along the time axis "
            f"(got length {train_data.shape[-2]}, seasonality {seasonality})"
        )
        raise ValueError(msg)
    scale = float(
        jnp.abs(train_data[..., seasonality:, :] - train_data[..., :-seasonality, :]).mean()
    )
    if scale == 0.0:
        msg = (
            "seasonal-naive scale is zero (constant training series); MASE is "
            "undefined. Use a different metric or seasonality."
        )
        raise ValueError(msg)

    def mase(pred: Array, truth: Array) -> float:
        mae = jnp.abs(jnp.median(pred, axis=0) - truth).mean()
        return float(mae / scale)

    return mase
