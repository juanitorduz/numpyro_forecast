"""Tests for the empirical CRPS implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from jaxtyping import TypeCheckError

from numpyro_forecast.metrics import (
    crps_empirical,
    eval_interval_score,
    eval_pinball,
    make_mase,
)


def _brute_force_crps(pred: Array, truth: Array) -> Array:
    """Reference O(n^2) CRPS: E|X-y| - 0.5 E|X-X'|."""
    term1 = jnp.abs(pred - truth).mean(axis=0)
    pairwise = jnp.abs(pred[:, None] - pred[None, :]).mean(axis=(0, 1))
    return term1 - 0.5 * pairwise


def test_crps_matches_brute_force(rng_key: Array) -> None:
    pred = random.normal(rng_key, (200, 3, 4))
    truth = random.normal(random.PRNGKey(7), (3, 4))
    got = crps_empirical(pred, truth)
    expected = _brute_force_crps(pred, truth)
    assert got.shape == (3, 4)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_crps_large_sample_no_int_overflow(rng_key: Array) -> None:
    # With n past ~46k both the rank weight ``i * (n - i)`` and the ``n ** 2``
    # normalization overflow int32 if computed as integers, which silently
    # corrupts (or raises in) the CRPS. Compare against an overflow-free float64
    # NumPy reference built from the same sorted-sample identity.
    n = 100_001
    pred = random.normal(rng_key, (n, 1))
    truth = jnp.array([0.3])
    got = crps_empirical(pred, truth)

    pred_np = np.asarray(pred, dtype=np.float64)
    truth_np = np.asarray(truth, dtype=np.float64)
    pred_sorted = np.sort(pred_np, axis=0)
    diff = pred_sorted[1:] - pred_sorted[:-1]
    i = np.arange(1, n, dtype=np.float64)
    weight = (i * (n - i))[:, None]
    absolute_error = np.abs(pred_np - truth_np).mean(axis=0)
    reference = absolute_error - (diff * weight).sum(axis=0) / n**2

    assert bool(jnp.all(got >= 0.0))
    assert jnp.allclose(got, jnp.asarray(reference), atol=1e-4)


def test_crps_deterministic_prediction_is_absolute_error() -> None:
    # All samples identical -> the dispersion term vanishes -> CRPS = |c - y|.
    pred = jnp.full((50, 2), 3.0)
    truth = jnp.array([1.0, 5.0])
    got = crps_empirical(pred, truth)
    assert jnp.allclose(got, jnp.array([2.0, 2.0]), atol=1e-6)


def test_crps_is_nonnegative(rng_key: Array) -> None:
    pred = random.normal(rng_key, (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    assert bool(jnp.all(crps_empirical(pred, truth) >= -1e-6))


def test_crps_shape_mismatch_raises() -> None:
    # The jaxtyping/beartype import hook enforces the shared ``*batch`` axis
    # between ``pred`` and ``truth`` at call time, before the manual check.
    with pytest.raises(TypeCheckError):
        crps_empirical(jnp.zeros((10, 3)), jnp.zeros((4,)))


def test_crps_needs_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2 samples"):
        crps_empirical(jnp.zeros((1, 3)), jnp.zeros((3,)))


# --- P7: pinball, interval score, MASE ---------------------------------------


def test_pinball_median_is_half_mae() -> None:
    pred = jnp.linspace(0.0, 1.0, 101)[:, None] * jnp.ones((101, 4))
    truth = jnp.full((4,), 0.7)
    # At tau=0.5 the pinball loss is half the absolute error of the median.
    median = jnp.median(pred, axis=0)
    expected = 0.5 * jnp.abs(median - truth).mean()
    assert jnp.allclose(eval_pinball(pred, truth, quantile=0.5), float(expected), atol=1e-5)


def test_pinball_minimized_at_true_quantile() -> None:
    # With truth drawn from the same law as the samples, the pinball loss is
    # minimized when we score the quantile that matches the target quantile.
    pred = random.normal(random.PRNGKey(0), (2_000, 400))
    truth = jnp.quantile(pred, 0.8, axis=0)
    at_truth = eval_pinball(pred, truth, quantile=0.8)
    below = eval_pinball(pred, truth, quantile=0.5)
    above = eval_pinball(pred, truth, quantile=0.95)
    assert at_truth < below
    assert at_truth < above


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.5])
def test_pinball_rejects_out_of_range_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match=r"quantile must be in"):
        eval_pinball(jnp.zeros((5, 2)), jnp.zeros((2,)), quantile=quantile)


def test_interval_score_rewards_tight_covering_interval() -> None:
    truth = jnp.zeros((4,))
    tight = 0.1 * random.normal(random.PRNGKey(0), (500, 4))
    wide = 5.0 * random.normal(random.PRNGKey(1), (500, 4))
    # Both cover 0, but the tight interval scores lower (better).
    assert eval_interval_score(tight, truth) < eval_interval_score(wide, truth)


def test_interval_score_penalizes_misses() -> None:
    samples = random.normal(random.PRNGKey(0), (500, 3))
    inside = eval_interval_score(samples, jnp.zeros((3,)))
    outside = eval_interval_score(samples, jnp.full((3,), 10.0))
    assert outside > inside


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2])
def test_interval_score_rejects_out_of_range_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must be in"):
        eval_interval_score(jnp.zeros((5, 2)), jnp.zeros((2,)), alpha=alpha)


def test_make_mase_scales_by_seasonal_naive() -> None:
    train = jnp.arange(10.0)[:, None]  # constant step-1 differences of 1.0
    mase = make_mase(train, seasonality=1)
    # Seasonal-naive scale is 1.0, so MASE equals the forecast MAE.
    pred = jnp.full((20, 3, 1), 5.0)
    truth = jnp.full((3, 1), 7.0)
    assert jnp.allclose(mase(pred, truth), 2.0, atol=1e-5)


def test_make_mase_rejects_bad_seasonality() -> None:
    with pytest.raises(ValueError, match="seasonality must be"):
        make_mase(jnp.arange(10.0)[:, None], seasonality=0)


def test_make_mase_rejects_short_train() -> None:
    with pytest.raises(ValueError, match="longer than seasonality"):
        make_mase(jnp.arange(3.0)[:, None], seasonality=5)


def test_make_mase_rejects_constant_series() -> None:
    with pytest.raises(ValueError, match="scale is zero"):
        make_mase(jnp.ones((10, 1)), seasonality=1)
