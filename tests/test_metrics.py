"""Tests for the empirical CRPS implementation."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
from conftest import commit_host
from jax import Array, random
from jaxtyping import TypeCheckError

from numpyro_forecast.evaluate import (
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
)
from numpyro_forecast.metrics import (
    _check_pred_truth,
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


def test_crps_empirical_truth_docstring_states_the_shape_contract() -> None:
    """The ``truth`` entry documents ``pred.shape[1:]``, not broadcasting (#199).

    The ``*batch`` annotation binds ``truth`` to ``pred.shape[1:]``, so a
    broadcastable ``truth`` such as ``(1,)`` against ``(5, 4)`` draws is
    rejected; the docstring must not promise broadcasting.
    """
    doc = crps_empirical.__doc__
    assert doc is not None
    assert "broadcastable" not in doc
    assert "(matching ``pred`` without the sample axis)" in doc
    with pytest.raises(TypeCheckError):
        crps_empirical(jnp.zeros((5, 4)), jnp.zeros((1,)))


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

    Mixing a pinned ``jax.Array`` (the ``device="host"`` fallback) with a
    device-resident one inside the fused jitted CRPS kernel used to raise
    (``memory_space of all inputs ... must be the same``); ``crps_empirical``
    now moves a host-resident operand to device memory first, so any mix
    matches the fully device-resident result.
    """
    pred = random.normal(rng_key, (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    expected = crps_empirical(pred, truth)

    pred_in = commit_host(pred, "pinned") if commit_pred else pred
    truth_in = commit_host(truth, "pinned") if commit_truth else truth
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

    pred_in = commit_host(pred, "pinned") if commit_pred else pred
    truth_in = commit_host(truth, "pinned") if commit_truth else truth
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

    pred_in = commit_host(pred, "pinned") if commit_pred else pred
    truth_in = commit_host(truth, "pinned") if commit_truth else truth
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

    pred_in = commit_host(pred, "pinned") if commit_pred else pred
    truth_in = commit_host(truth, "pinned") if commit_truth else truth
    got = mase(pred_in, truth_in)

    np.testing.assert_allclose(got, expected)


def test_crps_accepts_cpu_committed_pred_under_accelerator_default(
    rng_key: Array, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPU-committed ``pred`` (the primary ``device="host"`` result) scores unchanged.

    Structural coverage of ``_device_view``'s CPU-committed branch: on a
    CPU-only machine the move is a same-device no-op, but the kernel must
    accept the operand under an accelerator default backend.
    """
    pred = random.normal(rng_key, (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    expected = crps_empirical(pred, truth)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    np.testing.assert_allclose(crps_empirical(commit_host(pred, "cpu"), truth), expected)


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
    train_host = commit_host(train, "pinned")
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


# --- The pred/truth shape contract without the runtime type hook (#196) -----


def _bias(pred: Array, truth: Array) -> Array:
    """A custom metric for `evaluate_forecast()`: the mean error of the sample mean."""
    return (pred.mean(axis=0) - truth).mean()


@pytest.fixture
def no_type_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the jaxtyping runtime checks off; jaxtyping reads the flag on every call."""
    monkeypatch.setattr(jaxtyping.config, "jaxtyping_disable", True)


@pytest.mark.usefixtures("no_type_hook")
@pytest.mark.parametrize(
    "metric",
    [
        pytest.param(crps_empirical, id="crps_empirical"),
        pytest.param(eval_mae, id="eval_mae"),
        pytest.param(eval_rmse, id="eval_rmse"),
        pytest.param(eval_crps, id="eval_crps"),
        pytest.param(partial(eval_crps, batch_size=2), id="eval_crps batch_size=2"),
        pytest.param(eval_coverage, id="eval_coverage"),
        pytest.param(partial(eval_coverage, batch_size=2), id="eval_coverage batch_size=2"),
        pytest.param(eval_pinball, id="eval_pinball"),
        pytest.param(eval_interval_score, id="eval_interval_score"),
        pytest.param(make_mase(jnp.arange(10.0)[:, None] ** 1.5), id="make_mase"),
        pytest.param(evaluate_forecast, id="evaluate_forecast"),
        pytest.param(
            partial(evaluate_forecast, metrics={"bias": _bias}),
            id="evaluate_forecast custom metric",
        ),
    ],
)
def test_metric_rejects_truth_shape_mismatch_without_type_hook(
    metric: Callable[[Array, Array], object],
) -> None:
    """A ``(f,)`` truth against ``(S, f, 1)`` draws raises ``ValueError`` with the hook off."""
    pred = random.normal(random.PRNGKey(0), (20, 4, 1))
    with pytest.raises(ValueError, match=r"pred.shape\[1:\] = \(4, 1\), got \(4,\)"):
        metric(pred, jnp.arange(4.0))


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param(eval_coverage, id="eval_coverage"),
        pytest.param(partial(eval_coverage, batch_size=2), id="eval_coverage batch_size=2"),
        pytest.param(
            partial(evaluate_forecast, metrics={"bias": _bias}),
            id="evaluate_forecast custom metric",
        ),
    ],
)
def test_metric_rejects_scalar_truth(metric: Callable[[Array, float], object]) -> None:
    """A scalar ``truth`` passes the jaxtyping hook, so only the explicit check rejects it."""
    pred = random.normal(random.PRNGKey(0), (20, 4, 1))
    with pytest.raises(ValueError, match=r"pred.shape\[1:\] = \(4, 1\), got \(\)"):
        metric(pred, 0.0)


def test_check_pred_truth_rejects_truth_shape_mismatch_in_process() -> None:
    """The helper itself has no shape annotations, so it raises under the default hook too."""
    with pytest.raises(ValueError, match=r"pred.shape\[1:\] = \(4, 1\), got \(4,\)"):
        _check_pred_truth(jnp.zeros((20, 4, 1)), jnp.zeros((4,)))


def test_check_pred_truth_rejects_scalar_pred() -> None:
    """A point forecast passed as a scalar ``pred`` has no sample axis to drop."""
    with pytest.raises(ValueError, match="sample axis first"):
        _check_pred_truth(jnp.asarray(1.0), jnp.asarray(1.0))


def test_check_pred_truth_reads_static_shapes_of_tracers() -> None:
    """Under ``jit`` and ``vmap`` the check sees tracer shapes and needs no host value."""

    def checked(pred: Array, truth: Array) -> Array:
        _check_pred_truth(pred, truth)
        return pred.mean(axis=0) - truth

    pred = jnp.zeros((3, 20, 4, 1))
    truth = jnp.zeros((3, 4, 1))
    assert jax.jit(jax.vmap(checked))(pred, truth).shape == (3, 4, 1)
    with pytest.raises(ValueError, match="truth must have shape"):
        jax.jit(jax.vmap(checked))(pred, truth[..., 0])
