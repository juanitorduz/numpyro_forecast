"""Probabilistic forecast metrics.

This module ports :func:`pyro.ops.stats.crps_empirical` to JAX.
"""

import jax.numpy as jnp
from jaxtyping import Float

from numpyro_forecast.typing import Array


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
    if pred.shape[1:] != truth.shape:
        msg = (
            "pred and truth shapes mismatch: "
            f"pred.shape[1:]={pred.shape[1:]} vs truth.shape={truth.shape}"
        )
        raise ValueError(msg)

    num_samples = pred.shape[0]
    if num_samples < 2:
        msg = f"crps_empirical needs at least 2 samples, got {num_samples}"
        raise ValueError(msg)

    pred_sorted = jnp.sort(pred, axis=0)
    diff = pred_sorted[1:] - pred_sorted[:-1]
    weight = jnp.arange(1, num_samples) * jnp.arange(num_samples - 1, 0, -1)
    weight = weight.reshape(weight.shape + (1,) * (diff.ndim - 1))

    absolute_error = jnp.abs(pred - truth).mean(axis=0)
    return absolute_error - (diff * weight).sum(axis=0) / num_samples**2
