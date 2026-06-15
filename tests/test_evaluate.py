"""Tests for backtesting and evaluation metrics."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array, random

from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    BacktestResult,
    backtest,
    eval_crps,
    eval_mae,
    eval_rmse,
)
from numpyro_forecast.forecaster import ForecastingModel


class _RandomWalkModel(ForecastingModel):
    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series("drift", lambda: dist.Normal(0.0, drift_scale))
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


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


def test_default_metrics_keys() -> None:
    assert set(DEFAULT_METRICS) == {"mae", "rmse", "crps"}


def test_backtest_expanding_window(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        data,
        covariates,
        _RandomWalkModel,
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
        assert set(r.metrics) == {"mae", "rmse", "crps"}
        assert r.train_walltime >= 0.0
        # AutoNormal exposes per-site variational params (e.g. *_auto_loc).
        assert any("drift_scale" in name for name in r.params)
