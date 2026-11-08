"""Probabilistic forecast metrics.

This module ports `pyro.ops.stats.crps_empirical()` to JAX and adds the
pinball (quantile) loss, the Winkler interval score, and a MASE metric factory.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from numpyro_forecast.offload import float_view
from numpyro_forecast.typing import Array, Metric
from numpyro_forecast.validation import require_pred_truth_shapes


@jax.jit
def _crps_empirical(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    """Jitted CRPS core; the ``>= 2`` samples guard lives in `crps_empirical()`."""
    num_samples = pred.shape[0]
    pred_sorted = jnp.sort(pred, axis=0)
    diff = pred_sorted[1:] - pred_sorted[:-1]
    # Normalized rank weights (i / n) * ((n - i) / n) <= 1/4: neither the weights
    # nor the weighted sum (at most a quarter of the sample range) overflow a
    # low-precision dtype; float16 cannot hold i * (n - i) or n ** 2 from 256
    # draws (#184). The integer ranks divided by n give the default float dtype,
    # where the ranks are exact even past the float16 range, and no int32 product
    # i * (n - i) forms (it overflows to negative values once n exceeds ~46k).
    lower = jnp.arange(1, num_samples) / num_samples
    upper = jnp.arange(num_samples - 1, 0, -1) / num_samples
    weight = (
        (lower * upper).astype(pred.dtype).reshape((num_samples - 1,) + (1,) * (diff.ndim - 1))
    )
    absolute_error = jnp.abs(pred - truth).mean(axis=0)
    return absolute_error - (diff * weight).sum(axis=0)


def crps_empirical(
    pred: Real[ArrayLike, " sample *batch"],
    truth: Real[ArrayLike, " *batch"],
) -> Float[Array, " *batch"]:
    r"""Compute the empirical Continuous Ranked Probability Score (CRPS).

    The CRPS generalises the mean absolute error to probabilistic forecasts and
    is computed elementwise as

    $$
    \mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac{1}{2}\,\mathbb{E}|X - X'|,
    $$

    where $X, X'$ are independent draws from the forecast distribution
    $F$. The expectations are estimated from the forecast ``sample`` axis
    using the sorted-sample $O(n \log n)$ identity.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first, shape ``(sample, *batch)``.
        May be host-committed (e.g. draws sampled with ``device="host"``),
        regardless of whether ``truth`` is: either operand is moved to device
        memory first (`~~numpyro_forecast.offload.float_view()`). Integer
        draws (a count model) are cast to the default float dtype there; a
        floating ``pred`` keeps its dtype.
    truth
        Ground-truth values with shape ``(*batch)`` (matching ``pred`` without the sample axis).
        Integer values are cast like ``pred``.

    Returns
    -------
    Float[Array, "*batch"]
        Elementwise CRPS, one value per ``batch`` location.

    Raises
    ------
    ValueError
        If ``truth.shape`` is not ``pred.shape[1:]`` (see `~~numpyro_forecast.typing.Metric`), or
        ``pred`` has fewer than 2 samples.

    References
    ----------
    Tilmann Gneiting, Adrian E. Raftery (2007). "Strictly Proper Scoring Rules,
    Prediction, and Estimation". *Journal of the American Statistical
    Association*.
    """
    require_pred_truth_shapes(pred, truth)
    pred_arr = float_view(pred)
    num_samples = pred_arr.shape[0]
    if num_samples < 2:
        msg = f"crps_empirical needs at least 2 samples, got {num_samples}"
        raise ValueError(msg)
    return _crps_empirical(pred_arr, float_view(truth))


@partial(jax.jit, static_argnames=("quantile",))
def _eval_pinball(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    quantile: float,
) -> Array:
    """Jitted pinball-loss core; `eval_pinball()` places and casts the operands."""
    estimate = jnp.quantile(pred, quantile, axis=0)
    diff = truth - estimate
    return jnp.maximum(quantile * diff, (quantile - 1.0) * diff).mean()


def eval_pinball(
    pred: Real[ArrayLike, " sample *batch"],
    truth: Real[ArrayLike, " *batch"],
    *,
    quantile: float = 0.5,
) -> Array:
    r"""Mean pinball (quantile) loss of the forecast ``quantile``.

    The pinball loss for the forecast $\hat q$ of quantile $\tau$ is
    $\max(\tau (y - \hat q), (\tau - 1)(y - \hat q))$, averaged over all
    data elements. At ``quantile=0.5`` it is half the mean absolute error. A
    pure JAX scalar kernel (see `~~numpyro_forecast.typing.Metric`);
    ``quantile`` is static so each level specializes its own branch. ``pred``
    and ``truth`` are moved to device memory first
    (`~~numpyro_forecast.offload.float_view()`), so either (or
    both) may be host-committed, e.g. draws sampled with ``device="host"``.
    An integer operand (the draws of a count model) is cast to the default
    float dtype in the same step.

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
    Array
        The mean pinball loss as a scalar array.

    Raises
    ------
    ValueError
        If ``quantile`` is not strictly inside ``(0, 1)``, or ``truth.shape`` is not
        ``pred.shape[1:]`` (see `~~numpyro_forecast.typing.Metric`).
    """
    require_pred_truth_shapes(pred, truth)
    if not 0.0 < quantile < 1.0:
        msg = f"quantile must be in (0, 1), got {quantile}"
        raise ValueError(msg)
    return _eval_pinball(float_view(pred), float_view(truth), quantile=quantile)


@partial(jax.jit, static_argnames=("alpha",))
def _eval_interval_score(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    alpha: float,
) -> Array:
    """Jitted interval-score core; `eval_interval_score()` places and casts the operands."""
    tail = (1.0 - alpha) / 2.0
    lo = jnp.quantile(pred, tail, axis=0)
    hi = jnp.quantile(pred, 1.0 - tail, axis=0)
    penalty = 2.0 / (1.0 - alpha)
    # A comparison with NaN is False, so a NaN truth or quantile takes the
    # penalty branch and the NaN reaches the mean. A multiply by the mask
    # (truth < lo) compiles to a select that sets a NaN cell to zero.
    below = jnp.where(truth >= lo, 0.0, penalty * (lo - truth))
    above = jnp.where(truth <= hi, 0.0, penalty * (truth - hi))
    return (hi - lo + below + above).mean()


def eval_interval_score(
    pred: Real[ArrayLike, " sample *batch"],
    truth: Real[ArrayLike, " *batch"],
    *,
    alpha: float = 0.9,
) -> Array:
    r"""Mean Winkler interval score for the central ``alpha`` prediction interval.

    For the central ``alpha`` interval $[l, u]$ (the $(1-\alpha)/2$
    and $1-(1-\alpha)/2$ quantiles), the interval score is
    $(u - l) + \tfrac{2}{1-\alpha}\big[(l - y)\mathbf 1_{y<l} + (y - u)\mathbf 1_{y>u}\big]$,
    averaged over all data elements. It rewards narrow intervals and penalizes
    ground truth falling outside them; lower is better.
    A pure JAX scalar kernel (see `~~numpyro_forecast.typing.Metric`).
    ``pred`` and ``truth`` are moved to device memory first
    (`~~numpyro_forecast.offload.float_view()`), so either (or
    both) may be host-committed, e.g. draws sampled with ``device="host"``.
    An integer operand (the draws of a count model) is cast to the default
    float dtype in the same step.

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
    Array
        The mean interval score as a scalar array. NaN when a ``truth`` value
        or a forecast draw is NaN (see `~~numpyro_forecast.evaluate.DEFAULT_METRICS`).

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly inside ``(0, 1)``, or ``truth.shape`` is not
        ``pred.shape[1:]`` (see `~~numpyro_forecast.typing.Metric`).
    """
    require_pred_truth_shapes(pred, truth)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    return _eval_interval_score(float_view(pred), float_view(truth), alpha=alpha)


@partial(jax.jit, static_argnames=("scale",))
def _mase(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    scale: float,
) -> Array:
    """Jitted MASE core; the `make_mase()` closure places and casts the operands."""
    mae = jnp.abs(jnp.median(pred, axis=0) - truth).mean()
    return mae / scale


def make_mase(
    train_data: Real[ArrayLike, " *batch time obs_dim"], *, seasonality: int = 1
) -> Metric:
    """Build a Mean Absolute Scaled Error metric scaled by ``train_data``.

    MASE divides the forecast MAE (using the sample median as point estimate) by
    the in-sample MAE of the seasonal-naive forecast on ``train_data``,
    ``mean(|y_t - y_{t-seasonality}|)``. The scale is computed once at factory
    time; the returned metric has the standard scalar-array signature (see
    `~~numpyro_forecast.typing.Metric`). ``train_data`` and the returned
    metric's ``pred``/``truth`` are all moved to device memory first
    (`~~numpyro_forecast.offload.float_view()`), so any of
    them may be host-committed, e.g. draws sampled with ``device="host"``.
    Integer counts are cast to the default float dtype in the same step, so
    the seasonal-naive differences of ``train_data`` are computed in floating
    point: in the integer dtype a ``uint8`` difference wraps and the ``abs`` of
    the ``int16`` minimum stays negative.

    Parameters
    ----------
    train_data
        Training data with time at axis ``-2``; leading batch axes are allowed
        and the seasonal-naive scale is averaged over all axes. Integer counts
        are accepted.
    seasonality
        Seasonal period (``>= 1``); ``1`` is the random-walk naive baseline.

    Returns
    -------
    Metric
        A ``(pred, truth)`` callable computing MASE as a scalar array. It raises ``ValueError`` if
        ``truth.shape`` is not ``pred.shape[1:]`` (see `~~numpyro_forecast.typing.Metric`).

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
    # Cast before the lag difference: integer differences wrap (#163, pymc-labs/pymc-forecast#72).
    train_arr = float_view(train_data)
    if train_arr.shape[-2] <= seasonality:
        msg = (
            f"train_data must be longer than seasonality along the time axis "
            f"(got length {train_arr.shape[-2]}, seasonality {seasonality})"
        )
        raise ValueError(msg)
    scale = float(
        jnp.abs(train_arr[..., seasonality:, :] - train_arr[..., :-seasonality, :]).mean()
    )
    if scale == 0.0:
        msg = (
            "seasonal-naive scale is zero (constant training series); MASE is "
            "undefined. Use a different metric or seasonality."
        )
        raise ValueError(msg)

    def mase(pred: Real[ArrayLike, " sample *batch"], truth: Real[ArrayLike, " *batch"]) -> Array:
        """MASE for one ``(pred, truth)`` pair, scaled by the enclosing ``train_data``."""
        require_pred_truth_shapes(pred, truth)
        return _mase(float_view(pred), float_view(truth), scale=scale)

    return mase
