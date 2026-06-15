"""End-to-end smoke tests for the example models."""

import jax.numpy as jnp
import pytest
from example_models import HierarchicalForecaster, UnivariateForecaster
from jax import Array, random

from numpyro_forecast.datasets import bart_available, load_bart_hierarchical, load_bart_weekly
from numpyro_forecast.evaluate import eval_crps
from numpyro_forecast.forecaster import Forecaster
from numpyro_forecast.util import fourier_features


def _fit_and_forecast_univariate(
    y: Array,
    period: float,
    num_terms: int,
    future: int,
    rng_key: Array,
    num_steps: int,
) -> Array:
    duration = y.shape[0]
    covariates = fourier_features(duration, period, num_terms)
    t = duration - future
    key_fit, key_forecast = random.split(rng_key)
    forecaster = Forecaster(
        UnivariateForecaster(),
        y[:t],
        covariates[:t],
        num_steps=num_steps,
        rng_key=key_fit,
    )
    return forecaster(y[:t], covariates, num_samples=100, rng_key=key_forecast)


def test_univariate_synthetic(rng_key: Array) -> None:
    t = jnp.arange(120.0)
    season = jnp.sin(2 * jnp.pi * t / 52.0)
    y = (5.0 + season)[:, None] + 0.1 * random.normal(rng_key, (120, 1))
    pred = _fit_and_forecast_univariate(
        y, period=52.0, num_terms=5, future=12, rng_key=rng_key, num_steps=200
    )
    assert pred.shape == (100, 12, 1)
    crps = eval_crps(pred, y[-12:])
    assert jnp.isfinite(jnp.asarray(crps))


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_univariate_bart_smoke(rng_key: Array) -> None:
    y = load_bart_weekly()[-120:]  # last 120 weeks keeps the smoke test fast
    pred = _fit_and_forecast_univariate(
        y, period=52.18, num_terms=10, future=12, rng_key=rng_key, num_steps=200
    )
    assert pred.shape == (100, 12, 1)
    crps = eval_crps(pred, y[-12:])
    assert jnp.isfinite(jnp.asarray(crps))
    assert crps >= 0.0


def _fit_and_forecast_hierarchical(
    y: Array,
    period: int,
    future: int,
    rng_key: Array,
    num_steps: int,
) -> Array:
    n_origin, duration, n_destin = y.shape
    covariates = jnp.zeros((n_origin, duration, n_destin))
    t = duration - future
    key_fit, key_forecast = random.split(rng_key)
    forecaster = Forecaster(
        HierarchicalForecaster(period=period),
        y[:, :t, :],
        covariates[:, :t, :],
        num_steps=num_steps,
        rng_key=key_fit,
    )
    return forecaster(y[:, :t, :], covariates, num_samples=50, rng_key=key_forecast)


def test_hierarchical_synthetic(rng_key: Array) -> None:
    n_origin, duration, n_destin = 3, 48, 3
    season = jnp.sin(2 * jnp.pi * jnp.arange(duration) / 12.0)
    y = 2.0 + season[None, :, None] + 0.1 * random.normal(rng_key, (n_origin, duration, n_destin))
    pred = _fit_and_forecast_hierarchical(y, period=12, future=6, rng_key=rng_key, num_steps=150)
    assert pred.shape == (50, 3, 6, 3)
    assert jnp.isfinite(jnp.asarray(eval_crps(pred, y[:, -6:, :])))


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_hierarchical_bart_smoke(rng_key: Array) -> None:
    y, _split, _stations = load_bart_hierarchical()
    y = y[:4, -120:, :4]  # subsample stations and hours to keep the smoke test fast
    pred = _fit_and_forecast_hierarchical(
        y, period=24 * 7, future=24, rng_key=rng_key, num_steps=80
    )
    assert pred.shape == (50, 4, 24, 4)
    assert jnp.isfinite(jnp.asarray(eval_crps(pred, y[:, -24:, :])))
