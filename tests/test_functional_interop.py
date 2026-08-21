"""Interchangeability tests between the functional and OOP APIs."""

import jax.numpy as jnp
from conftest import RandomWalkModel, empty_covariates, rw_body, svi_forecast_fn
from jax import random

from numpyro_forecast.evaluate import backtest
from numpyro_forecast.forecaster import Forecaster, HMCForecaster
from numpyro_forecast.functional import (
    draw_posterior,
    fit_svi,
    forecast,
    forecasting_model,
)


def test_functional_model_through_oop_forecaster() -> None:
    func_model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    forecaster = Forecaster(
        random.PRNGKey(1), func_model, data, empty_covariates(30), num_steps=30
    )
    fc = forecaster(random.PRNGKey(2), data, empty_covariates(36), num_samples=8)
    assert fc.shape == (8, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_oop_forecaster_parallel_matches_serial() -> None:
    func_model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    forecaster = Forecaster(
        random.PRNGKey(1), func_model, data, empty_covariates(30), num_steps=30
    )
    serial = forecaster(
        random.PRNGKey(2), data, empty_covariates(36), num_samples=8, parallel=False
    )
    vmapped = forecaster(
        random.PRNGKey(2), data, empty_covariates(36), num_samples=8, parallel=True
    )
    assert vmapped.shape == serial.shape == (8, 6, 1)
    assert jnp.allclose(vmapped, serial, atol=1e-4)


def test_functional_model_through_hmc_forecaster() -> None:
    func_model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (20, 1)), axis=-2)
    forecaster = HMCForecaster(
        random.PRNGKey(1),
        func_model,
        data,
        empty_covariates(20),
        num_warmup=15,
        num_samples=15,
    )
    fc = forecaster(random.PRNGKey(2), data, empty_covariates(26), num_samples=8)
    assert fc.shape == (8, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_oop_model_through_functional_fit_and_forecast() -> None:
    oop_model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    fit = fit_svi(random.PRNGKey(1), oop_model, data, empty_covariates(30), num_steps=30)
    post = draw_posterior(random.PRNGKey(2), fit, 8)
    fc = forecast(random.PRNGKey(3), oop_model, post, data, empty_covariates(36))
    assert fc.shape == (8, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_functional_model_in_backtest() -> None:
    # Demonstrates the functional model type works as-is inside backtest's
    # closure contract (a plain NumPyro forecast_fn, not the OOP Forecaster).
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        random.PRNGKey(1),
        data,
        covariates,
        lambda: forecasting_model(rw_body),
        forecast_fn=svi_forecast_fn(num_steps=20),
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=10,
    )
    assert len(results) == 3
    for r in results:
        assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}


def test_oop_and_functional_fits_and_forecasts_are_identical() -> None:
    # Same model both ways, same keys: SVI is deterministic, so params and the
    # resulting forecast samples must match bit for bit.
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    cov_train, cov_full = empty_covariates(30), empty_covariates(36)
    key_fit, key_fc = random.PRNGKey(7), random.PRNGKey(9)

    oop = Forecaster(key_fit, RandomWalkModel(), data, cov_train, num_steps=40)
    fc_oop = oop(key_fc, data, cov_full, num_samples=8)

    func_model = forecasting_model(rw_body)
    fit = fit_svi(key_fit, func_model, data, cov_train, num_steps=40)
    for name, value in oop.params.items():
        assert jnp.array_equal(value, fit.params[name]), name

    key_post, key_pred = random.split(key_fc)  # mirror _BaseForecaster.__call__
    post = draw_posterior(key_post, fit, 8)
    fc_func = forecast(key_pred, func_model, post, data, cov_full)
    assert jnp.array_equal(fc_oop, fc_func)
