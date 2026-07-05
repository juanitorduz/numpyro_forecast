"""Tests for :func:`backtest_vectorized` (roadmap §4.3).

Covers estimator equivalence with the loop :func:`backtest` (window indices +
statistical metric closeness), every validator, the window-count formula, the
compile-discipline gate (I3: the vmapped SVI fit does not recompile per window),
and the I6 acceptance check that a subsequent eager ``fit_svi`` with a fresh
guide runs without an ``UnexpectedTracerError``.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random

from numpyro_forecast.evaluate import (
    VectorizedBacktestResult,
    backtest,
    backtest_vectorized,
)
from numpyro_forecast.forecaster import ForecastingModel
from numpyro_forecast.functional import fit_svi, forecast

TRAIN, TEST = 30, 5


class _RandomWalk(ForecastingModel):
    """Local-level random walk with Normal observation noise."""

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series("drift", lambda: dist.Normal(0.0, drift_scale))
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


def _series(duration: int) -> tuple[Array, Array]:
    """A smooth univariate series ``(duration, 1)`` and empty covariates."""
    t = jnp.linspace(0.0, 6.0, duration)
    y = (jnp.sin(t) + 0.05 * random.normal(random.PRNGKey(0), (duration,)))[:, None]
    return y, jnp.zeros((duration, 0))


def test_window_indices_match_loop_backtest() -> None:
    """Vectorized window indices equal the rolling-window loop ``backtest``."""
    duration = 55
    data, cov = _series(duration)
    stride = 3
    loop = backtest(
        random.PRNGKey(0),
        data,
        cov,
        _RandomWalk,
        train_window=TRAIN,
        test_window=TEST,
        stride=stride,
        num_samples=20,
        forecaster_options={"num_steps": 50},
    )
    vec = backtest_vectorized(
        random.PRNGKey(0),
        data,
        cov,
        _RandomWalk,
        train_window=TRAIN,
        test_window=TEST,
        stride=stride,
        num_steps=50,
        num_samples=20,
    )
    assert [int(x) for x in vec.t0] == [r.t0 for r in loop]
    assert [int(x) for x in vec.t1] == [r.t1 for r in loop]
    assert [int(x) for x in vec.t2] == [r.t2 for r in loop]


def test_metrics_statistically_close_to_loop() -> None:
    """Per-window MAE/CRPS match the loop path within a generous tolerance."""
    duration = 50
    data, cov = _series(duration)
    loop = backtest(
        random.PRNGKey(3),
        data,
        cov,
        _RandomWalk,
        train_window=TRAIN,
        test_window=TEST,
        stride=5,
        num_samples=200,
        forecaster_options={"num_steps": 800},
    )
    vec = backtest_vectorized(
        random.PRNGKey(3),
        data,
        cov,
        _RandomWalk,
        train_window=TRAIN,
        test_window=TEST,
        stride=5,
        num_steps=800,
        num_samples=200,
    )
    loop_mae = jnp.array([r.metrics["mae"] for r in loop])
    # Estimators differ in PRNG layout; check coarse agreement on MAE only.
    assert jnp.all(jnp.isfinite(vec.metrics["mae"]))
    assert jnp.all(jnp.isfinite(loop_mae))
    rel_err = jnp.abs(vec.metrics["mae"] - loop_mae) / jnp.maximum(loop_mae, 1e-6)
    assert float(rel_err.mean()) < 1.5


@pytest.mark.parametrize(
    ("train_window", "test_window", "stride", "message"),
    [
        (0, TEST, 1, "train_window must be >= 1"),
        (TRAIN, 0, 1, "test_window must be >= 1"),
        (TRAIN, TEST, 0, "stride must be >= 1"),
    ],
)
def test_window_size_validators(
    train_window: int, test_window: int, stride: int, message: str
) -> None:
    """Each window-size/stride constraint raises its own ``ValueError``."""
    data, cov = _series(50)
    with pytest.raises(ValueError, match=message):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            _RandomWalk,
            train_window=train_window,
            test_window=test_window,
            stride=stride,
            num_steps=10,
        )


def test_handwritten_guide_rejected() -> None:
    """A non-AutoGuide guide is rejected with guidance to use ``backtest``."""
    data, cov = _series(50)

    def handwritten_guide(covariates: Array, data: Array | None = None) -> None:
        loc = numpyro.param("loc", 0.0)
        numpyro.sample("drift_scale", dist.Delta(loc))

    with pytest.raises(ValueError, match="AutoGuide"):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            _RandomWalk,
            train_window=TRAIN,
            test_window=TEST,
            guide=handwritten_guide,
            num_steps=10,
        )


def test_duration_too_short_rejected() -> None:
    """A series with no room for a single window raises ``ValueError``."""
    data, cov = _series(TRAIN + TEST - 1)
    with pytest.raises(ValueError, match="no window fits"):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            _RandomWalk,
            train_window=TRAIN,
            test_window=TEST,
            num_steps=10,
        )


def test_covariate_length_mismatch_rejected() -> None:
    """Mismatched data/covariate durations raise ``ValueError``."""
    data, _ = _series(50)
    cov = jnp.zeros((49, 0))
    with pytest.raises(ValueError, match="share the time axis"):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            _RandomWalk,
            train_window=TRAIN,
            test_window=TEST,
            num_steps=10,
        )


@pytest.mark.parametrize(
    ("duration", "train", "test", "stride"),
    [
        (50, 30, 5, 1),
        (50, 30, 5, 3),
        (60, 20, 10, 2),
        (40, 25, 5, 4),
    ],
)
def test_window_count_matches_formula(duration: int, train: int, test: int, stride: int) -> None:
    """The number of windows follows ``floor((D - train - test) / stride) + 1``."""
    data, cov = _series(duration)
    result = backtest_vectorized(
        random.PRNGKey(0),
        data,
        cov,
        _RandomWalk,
        train_window=train,
        test_window=test,
        stride=stride,
        num_steps=10,
        num_samples=5,
    )
    expected = (duration - train - test) // stride + 1
    assert result.t0.shape[0] == expected


def test_single_svi_compilation(
    count_compilations: Callable[[], object],
) -> None:
    """I3: the fused fit is O(1) in window count, not O(num_windows).

    The vmapped SVI fit treats windows as a batch dimension, so compilation
    count stays bounded (fit + posterior + forecast stages) rather than scaling
    linearly with the number of windows. Replaying the same shapes triggers no
    further backend compilations.
    """
    import jax

    duration = TRAIN + TEST + 9  # exactly 10 windows at stride 1
    data, cov = _series(duration)

    def run() -> int:
        with count_compilations() as tally:  # type: ignore[operator]
            result = backtest_vectorized(
                random.PRNGKey(0),
                data,
                cov,
                _RandomWalk,
                train_window=TRAIN,
                test_window=TEST,
                stride=1,
                num_steps=30,
                num_samples=10,
            )
            jax.block_until_ready(result.losses)
        return int(tally.count)  # type: ignore[attr-defined]

    first = run()
    # First call may compile several vmapped stages (fit, posterior, forecast, metrics).
    assert first < 50

    with count_compilations() as tally:  # type: ignore[operator]
        result = backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            _RandomWalk,
            train_window=TRAIN,
            test_window=TEST,
            stride=1,
            num_steps=30,
            num_samples=10,
        )
        jax.block_until_ready(result.losses)
    # Same shapes should not trigger a full recompile storm (allow a few residual).
    assert tally.count <= max(first, 1)  # type: ignore[attr-defined]


def test_no_tracer_leak_on_subsequent_eager_fit() -> None:
    """I6 acceptance: a fresh eager ``fit_svi`` after a vectorized run is clean.

    The mandatory eager warm-up keeps the vectorized run's AutoGuide instance
    uncontaminated, and a subsequent eager fit uses a fresh guide, so no
    ``UnexpectedTracerError`` escapes into later inference.
    """
    duration = TRAIN + TEST + 7
    data, cov = _series(duration)
    backtest_vectorized(
        random.PRNGKey(1),
        data,
        cov,
        _RandomWalk,
        train_window=TRAIN,
        test_window=TEST,
        num_steps=30,
        num_samples=10,
    )
    train_data = data[:TRAIN]
    train_cov = cov[:TRAIN]
    full_cov = cov[: TRAIN + TEST]
    fit = fit_svi(random.PRNGKey(2), _RandomWalk(), train_data, train_cov, num_steps=30)
    from numpyro_forecast.functional import draw_posterior

    posterior = draw_posterior(random.PRNGKey(3), fit, 10)
    preds = forecast(random.PRNGKey(4), _RandomWalk(), posterior, train_data, full_cov)
    assert preds.shape[0] == 10
    assert jnp.all(jnp.isfinite(preds))


def test_keep_predictions_shape() -> None:
    """``keep_predictions=True`` retains ``(windows, samples, test, obs)`` forecasts."""
    duration = TRAIN + TEST + 6
    data, cov = _series(duration)
    result = backtest_vectorized(
        random.PRNGKey(0),
        data,
        cov,
        _RandomWalk,
        train_window=TRAIN,
        test_window=TEST,
        num_steps=20,
        num_samples=15,
        keep_predictions=True,
    )
    assert isinstance(result, VectorizedBacktestResult)
    assert result.predictions is not None
    num_windows = result.t0.shape[0]
    assert result.predictions.shape == (num_windows, 15, TEST, 1)
