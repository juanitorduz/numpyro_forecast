"""Tests for backtesting and evaluation metrics."""

import jax.numpy as jnp
import pytest
from conftest import RandomWalkModel
from jax import Array, random

from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    BacktestResult,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
)
from numpyro_forecast.forecaster import HMCForecaster


def test_eval_mae_uses_median() -> None:
    pred = jnp.array([1.0, 2.0, 9.0]).reshape(3, 1)  # median 2
    truth = jnp.array([0.0])
    assert eval_mae(pred, truth) == 2.0


def test_eval_rmse_uses_mean() -> None:
    pred = jnp.array([0.0, 4.0]).reshape(2, 1)  # mean 2
    truth = jnp.array([0.0])
    assert eval_rmse(pred, truth) == 2.0


def test_eval_crps_returns_float() -> None:
    pred = random.normal(random.PRNGKey(0), (50, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    value = eval_crps(pred, truth)
    assert isinstance(value, float)
    assert value >= 0.0


def test_eval_coverage_perfect_and_zero() -> None:
    # Samples spread symmetrically around 0; truth at 0 is inside any central band.
    pred = jnp.linspace(-1.0, 1.0, 101).reshape(101, 1)
    assert eval_coverage(pred, jnp.array([0.0])) == 1.0
    # Truth far outside the sample support falls outside the band.
    assert eval_coverage(pred, jnp.array([100.0])) == 0.0


def test_eval_coverage_returns_float() -> None:
    pred = random.normal(random.PRNGKey(0), (200, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    value = eval_coverage(pred, truth, alpha=0.8)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_default_metrics_keys() -> None:
    assert set(DEFAULT_METRICS) == {"mae", "rmse", "crps", "coverage"}


def test_evaluate_forecast_matches_individual_metrics() -> None:
    pred = random.normal(random.PRNGKey(0), (200, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    report = evaluate_forecast(pred, truth)
    assert set(report) == set(DEFAULT_METRICS)
    assert report["mae"] == eval_mae(pred, truth)
    assert report["rmse"] == eval_rmse(pred, truth)
    assert report["crps"] == eval_crps(pred, truth)
    assert report["coverage"] == eval_coverage(pred, truth)


def test_evaluate_forecast_honors_custom_metrics() -> None:
    pred = random.normal(random.PRNGKey(0), (50, 3))
    truth = random.normal(random.PRNGKey(1), (3,))
    report = evaluate_forecast(pred, truth, metrics={"mae": eval_mae})
    assert set(report) == {"mae"}
    assert report["mae"] == eval_mae(pred, truth)


def test_evaluate_forecast_multidim_batch() -> None:
    # Exercises the ``*batch`` part of the ``(sample, *batch)`` annotation.
    pred = random.normal(random.PRNGKey(0), (200, 5, 2))  # (sample, time, obs)
    truth = random.normal(random.PRNGKey(1), (5, 2))
    report = evaluate_forecast(pred, truth)
    assert set(report) == set(DEFAULT_METRICS)
    assert all(isinstance(value, float) for value in report.values())


def test_backtest_expanding_window(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options={"num_steps": 30},
    )
    # Windows at t1 in {12, 16, 20} (stop = 24 - 4 + 1 = 21).
    assert [r.t1 for r in results] == [12, 16, 20]
    for r in results:
        assert isinstance(r, BacktestResult)
        assert r.t0 == 0  # expanding window
        assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}
        assert r.train_walltime >= 0.0
        # AutoNormal exposes per-site variational params (e.g. *_auto_loc).
        assert any("drift_scale" in name for name in r.params)


def test_backtest_result_to_dict() -> None:
    result = BacktestResult(
        t0=0,
        t1=10,
        t2=14,
        num_samples=20,
        train_walltime=0.5,
        test_walltime=0.25,
        metrics={"mae": 1.0},
        params={"sigma": 0.3},
    )
    flat = result.to_dict()
    assert flat["t0"] == 0
    assert flat["t1"] == 10
    assert flat["metrics"] == {"mae": 1.0}
    assert flat["params"] == {"sigma": 0.3}
    assert set(flat) == {
        "t0",
        "t1",
        "t2",
        "num_samples",
        "train_walltime",
        "test_walltime",
        "metrics",
        "params",
    }


def test_backtest_hmc_has_no_scalar_params(rng_key: Array) -> None:
    # HMCForecaster has no ``params`` mapping, so ``_scalar_params`` returns {}.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (20, 1)), axis=-2)
    covariates = jnp.zeros((20, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        forecaster_fn=HMCForecaster,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=10,
        forecaster_options={"num_warmup": 10, "num_samples": 10},
    )
    assert results
    for r in results:
        assert r.params == {}


def test_backtest_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share the time axis length"):
        backtest(random.PRNGKey(0), jnp.zeros((20, 1)), jnp.zeros((18, 0)), RandomWalkModel)


def test_backtest_callable_options_and_transform(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    seen_windows: list[tuple[int, int, int]] = []

    def options_for(t0: int, t1: int, t2: int) -> dict[str, int]:
        seen_windows.append((t0, t1, t2))
        return {"num_steps": 30}

    transform_calls = {"count": 0}

    def transform(pred: Array, truth: Array) -> tuple[Array, Array]:
        transform_calls["count"] += 1
        return jnp.exp(pred), jnp.exp(truth)

    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options=options_for,
        transform=transform,
    )
    assert len(results) == 3
    assert seen_windows == [(0, 12, 16), (0, 16, 20), (0, 20, 24)]
    assert transform_calls["count"] == 3
