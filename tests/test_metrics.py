"""Tests for the empirical CRPS implementation."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from jaxtyping import TypeCheckError

from numpyro_forecast.functional._offload import _host_sharding
from numpyro_forecast.metrics import (
    crps_empirical,
    eval_interval_score,
    eval_pinball,
    make_mase,
)
from numpyro_forecast.typing import Metric


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


@pytest.mark.parametrize(
    ("commit_pred", "commit_truth"),
    [(True, False), (False, True), (True, True)],
    ids=["host_pred_device_truth", "device_pred_host_truth", "both_host"],
)
def test_crps_accepts_host_committed_inputs(
    rng_key: Array, commit_pred: bool, commit_truth: bool
) -> None:
    """``crps_empirical`` must accept any mix of host-committed/device-resident inputs.

    Mixing a host-committed ``jax.Array`` (e.g. draws sampled with
    ``device="host"``) with a device-resident one inside the fused jitted CRPS
    kernel used to raise (``memory_space of all inputs ... must be the
    same``); ``crps_empirical`` now moves a host-committed operand to device
    memory first, so any mix matches the fully device-resident result.
    """
    pred = random.normal(rng_key, (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    expected = crps_empirical(pred, truth)

    pred_in = jax.device_put(pred, _host_sharding(pred)) if commit_pred else pred
    truth_in = jax.device_put(truth, _host_sharding(truth)) if commit_truth else truth
    got = crps_empirical(pred_in, truth_in)

    np.testing.assert_allclose(got, expected)


# --- P7: pinball, interval score, MASE ---------------------------------------


@pytest.mark.parametrize(
    ("commit_pred", "commit_truth"),
    [(True, False), (False, True), (True, True)],
    ids=["host_pred_device_truth", "device_pred_host_truth", "both_host"],
)
def test_pinball_accepts_host_committed_inputs(commit_pred: bool, commit_truth: bool) -> None:
    """``eval_pinball`` must accept any mix of host-committed/device-resident inputs."""
    pred = random.normal(random.PRNGKey(0), (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    expected = eval_pinball(pred, truth, quantile=0.3)

    pred_in = jax.device_put(pred, _host_sharding(pred)) if commit_pred else pred
    truth_in = jax.device_put(truth, _host_sharding(truth)) if commit_truth else truth
    got = eval_pinball(pred_in, truth_in, quantile=0.3)

    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize(
    ("commit_pred", "commit_truth"),
    [(True, False), (False, True), (True, True)],
    ids=["host_pred_device_truth", "device_pred_host_truth", "both_host"],
)
def test_interval_score_accepts_host_committed_inputs(
    commit_pred: bool, commit_truth: bool
) -> None:
    """``eval_interval_score`` must accept any mix of host-committed/device-resident inputs."""
    pred = random.normal(random.PRNGKey(0), (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    expected = eval_interval_score(pred, truth, alpha=0.8)

    pred_in = jax.device_put(pred, _host_sharding(pred)) if commit_pred else pred
    truth_in = jax.device_put(truth, _host_sharding(truth)) if commit_truth else truth
    got = eval_interval_score(pred_in, truth_in, alpha=0.8)

    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize(
    ("commit_pred", "commit_truth"),
    [(True, False), (False, True), (True, True)],
    ids=["host_pred_device_truth", "device_pred_host_truth", "both_host"],
)
def test_mase_accepts_host_committed_inputs(commit_pred: bool, commit_truth: bool) -> None:
    """The ``mase`` closure must accept any mix of host-committed/device-resident inputs."""
    train = jnp.sin(jnp.arange(12.0))[:, None]
    mase = make_mase(train, seasonality=3)
    pred = random.normal(random.PRNGKey(0), (100, 5, 1))
    truth = random.normal(random.PRNGKey(1), (5, 1))
    expected = mase(pred, truth)

    pred_in = jax.device_put(pred, _host_sharding(pred)) if commit_pred else pred
    truth_in = jax.device_put(truth, _host_sharding(truth)) if commit_truth else truth
    got = mase(pred_in, truth_in)

    np.testing.assert_allclose(got, expected)


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


def _mase_scale(mase: Metric) -> float:
    """Back out the factory-time scale: with a zero point forecast the metric is mae/scale."""
    pred = jnp.zeros((1, 1, 1))
    truth = jnp.ones((1, 1))  # mae = |0 - 1| = 1, so metric == 1/scale
    return 1.0 / float(mase(pred, truth))


def test_make_mase_accepts_batched_train_data() -> None:
    """A ``(*batch, time, obs)`` train_data builds a metric; the scale averages over batch.

    Regression: the annotation was tightened to exactly 2-d, so any batched
    caller hit a runtime ``TypeCheckError`` despite the docstring's package-wide
    time-at-axis(-2) contract.
    """
    train = jnp.stack([jnp.arange(8.0)[:, None], 3.0 * jnp.arange(8.0)[:, None]])
    mase = make_mase(train, seasonality=1)
    per_series = [
        float(jnp.abs(train[b, 1:, :] - train[b, :-1, :]).mean()) for b in range(train.shape[0])
    ]
    expected_scale = float(np.mean(per_series))
    assert jnp.allclose(_mase_scale(mase), expected_scale, atol=1e-6)


def test_make_mase_time_axis_is_minus_two() -> None:
    """A ``(time, obs)`` input and its ``(1, time, obs)`` unsqueeze yield the same scale."""
    train = jnp.sin(jnp.arange(12.0))[:, None]
    scale_2d = _mase_scale(make_mase(train, seasonality=3))
    scale_3d = _mase_scale(make_mase(train[None], seasonality=3))
    assert jnp.allclose(scale_2d, scale_3d, atol=1e-6)


def test_make_mase_rejects_bad_seasonality() -> None:
    with pytest.raises(ValueError, match="seasonality must be"):
        make_mase(jnp.arange(10.0)[:, None], seasonality=0)


def test_make_mase_rejects_short_train() -> None:
    with pytest.raises(ValueError, match="longer than seasonality"):
        make_mase(jnp.arange(3.0)[:, None], seasonality=5)


def test_make_mase_rejects_constant_series() -> None:
    with pytest.raises(ValueError, match="scale is zero"):
        make_mase(jnp.ones((10, 1)), seasonality=1)


def test_make_mase_accepts_host_committed_train_data() -> None:
    """``make_mase`` itself must accept a host-committed ``train_data``."""
    train = jnp.sin(jnp.arange(12.0))[:, None]
    train_host = jax.device_put(train, _host_sharding(train))
    expected_scale = _mase_scale(make_mase(train, seasonality=3))
    got_scale = _mase_scale(make_mase(train_host, seasonality=3))
    assert jnp.allclose(got_scale, expected_scale, atol=1e-6)


# --- The array-metric contract: scalar-array returns, vmap composability -----


def test_metrics_return_scalar_arrays() -> None:
    """Every metric returns a 0-d array (host floats only at result boundaries)."""
    pred = random.normal(random.PRNGKey(0), (100, 6, 1))
    truth = random.normal(random.PRNGKey(1), (6, 1))
    mase = make_mase(jnp.arange(10.0)[:, None], seasonality=1)
    for value in (
        eval_pinball(pred, truth, quantile=0.3),
        eval_interval_score(pred, truth, alpha=0.8),
        mase(pred, truth),
    ):
        assert value.shape == ()


def test_metrics_are_vmappable() -> None:
    """Metrics are pure JAX functions: vmapping over a leading axis works.

    This is the property ``backtest_vectorized`` relies on to score every
    window in one fused computation, including partial-bound variants.
    """
    pred = random.normal(random.PRNGKey(0), (4, 100, 6, 1))
    truth = random.normal(random.PRNGKey(1), (4, 6, 1))
    mase = make_mase(jnp.arange(10.0)[:, None], seasonality=1)
    for metric in (
        partial(eval_pinball, quantile=0.3),
        partial(eval_interval_score, alpha=0.8),
        mase,
    ):
        values = jax.vmap(metric)(pred, truth)
        assert values.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(values)))
