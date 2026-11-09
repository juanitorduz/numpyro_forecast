"""End-to-end smoke tests for the example models."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
import pytest
from example_models import make_hierarchical_model, make_trend_inflation_model, univariate_model
from jax import Array, random
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.datasets import (
    bart_available,
    load_bart_hierarchical,
    load_bart_weekly,
    load_us_inflation,
)
from numpyro_forecast.evaluate import eval_crps
from numpyro_forecast.features import fourier_features
from numpyro_forecast.predictive import draw_posterior, forecast
from numpyro_forecast.typing import ForecastModel

PosteriorFactory = Callable[[Array, ForecastModel, Array, Array], dict[str, Array]]


def _svi_posterior_factory(num_steps: int, num_samples: int = 100) -> PosteriorFactory:
    """A plain-NumPyro SVI posterior factory with fixed step/draw counts (BART smoke tests).

    Mirrors ``conftest``'s ``posterior_factory`` fixture but with custom hyperparameters
    the fast-test defaults are too small for.
    """

    def draw(
        rng_key: Array, model: ForecastModel, data: Array, covariates: Array
    ) -> dict[str, Array]:
        key_fit, key_draw = random.split(rng_key)
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        state = svi.run(key_fit, num_steps, covariates, data, progress_bar=False)
        return cast("dict[str, Array]", draw_posterior(key_draw, guide, state.params, num_samples))

    return draw


def _fit_and_forecast_univariate(
    rng_key: Array,
    y: Array,
    period: float,
    num_terms: int,
    future: int,
    posterior_factory: PosteriorFactory,
) -> Array:
    duration = y.shape[0]
    covariates = fourier_features(duration, period, num_terms)
    t = duration - future
    key_fit, key_forecast = random.split(rng_key)
    posterior = posterior_factory(key_fit, univariate_model, y[:t], covariates[:t])
    pred = forecast(key_forecast, univariate_model, posterior, y[:t], covariates)
    assert isinstance(pred, jax.Array)
    return pred


def test_univariate_synthetic(
    posterior_factory: PosteriorFactory, rng_key: Array, fast_mcmc: dict[str, int]
) -> None:
    t = jnp.arange(120.0)
    season = jnp.sin(2 * jnp.pi * t / 52.0)
    y = (5.0 + season)[:, None] + 0.1 * random.normal(rng_key, (120, 1))
    pred = _fit_and_forecast_univariate(
        rng_key, y, period=52.0, num_terms=5, future=12, posterior_factory=posterior_factory
    )
    assert pred.shape == (fast_mcmc["num_samples"], 12, 1)
    crps = eval_crps(pred, y[-12:])
    assert jnp.isfinite(jnp.asarray(crps))


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_univariate_bart_smoke(rng_key: Array) -> None:
    y = load_bart_weekly()[-120:]
    pred = _fit_and_forecast_univariate(
        rng_key,
        y,
        period=52.18,
        num_terms=10,
        future=12,
        posterior_factory=_svi_posterior_factory(200),
    )
    assert pred.shape == (100, 12, 1)
    crps = eval_crps(pred, y[-12:])
    assert jnp.isfinite(jnp.asarray(crps))
    assert crps >= 0.0


def _fit_and_forecast_hierarchical(
    rng_key: Array,
    y: Array,
    period: int,
    future: int,
    posterior_factory: PosteriorFactory,
) -> Array:
    n_origin, duration, n_destin = y.shape
    covariates = jnp.zeros((n_origin, duration, n_destin))
    t = duration - future
    model = make_hierarchical_model(period=period)
    key_fit, key_forecast = random.split(rng_key)
    posterior = posterior_factory(key_fit, model, y[:, :t, :], covariates[:, :t, :])
    pred = forecast(key_forecast, model, posterior, y[:, :t, :], covariates)
    assert isinstance(pred, jax.Array)
    return pred


def test_hierarchical_synthetic(
    posterior_factory: PosteriorFactory, rng_key: Array, fast_mcmc: dict[str, int]
) -> None:
    n_origin, duration, n_destin = 3, 48, 3
    season = jnp.sin(2 * jnp.pi * jnp.arange(duration) / 12.0)
    y = 2.0 + season[None, :, None] + 0.1 * random.normal(rng_key, (n_origin, duration, n_destin))
    pred = _fit_and_forecast_hierarchical(
        rng_key, y, period=12, future=6, posterior_factory=posterior_factory
    )
    assert pred.shape == (fast_mcmc["num_samples"], 3, 6, 3)
    assert jnp.isfinite(jnp.asarray(eval_crps(pred, y[:, -6:, :])))


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_hierarchical_bart_smoke(rng_key: Array) -> None:
    y, _split, _stations = load_bart_hierarchical()
    y = y[:4, -120:, :4]
    pred = _fit_and_forecast_hierarchical(
        rng_key, y, period=24 * 7, future=24, posterior_factory=_svi_posterior_factory(80, 50)
    )
    assert pred.shape == (50, 4, 24, 4)
    assert jnp.isfinite(jnp.asarray(eval_crps(pred, y[:, -24:, :])))


def _nuts_posterior(
    rng_key: Array, model: ForecastModel, data: Array, fast_mcmc: dict[str, int]
) -> dict[str, Array]:
    mcmc = MCMC(
        NUTS(model),
        num_warmup=fast_mcmc["num_warmup"],
        num_samples=fast_mcmc["num_samples"],
        num_chains=fast_mcmc["num_chains"],
        progress_bar=False,
    )
    mcmc.run(rng_key, jnp.zeros((data.shape[0], 0)), data)
    return mcmc.get_samples()


@pytest.mark.parametrize("num_measures", [1, 4])
@pytest.mark.parametrize("stochastic_volatility", [True, False])
@pytest.mark.parametrize("common_transitory", [False, True])
def test_trend_inflation_prior_predictive_shapes(
    rng_key: Array, num_measures: int, stochastic_volatility: bool, common_transitory: bool
) -> None:
    model = make_trend_inflation_model(
        num_measures,
        stochastic_volatility=stochastic_volatility,
        common_transitory=common_transitory,
    )
    prior = Predictive(model, num_samples=5)(rng_key, jnp.zeros((30, 0)))
    assert prior["obs"].shape == (5, 30, num_measures)
    assert prior["trend"].shape == (5, 30, 1)
    assert prior["gap_vol"].shape == (5, 30, num_measures)
    assert ("common_shock" in prior) == common_transitory
    if common_transitory:
        assert prior["common_shock"].shape == (5, 30, 1)


def test_trend_inflation_stochastic_volatility_moves_over_the_horizon(
    rng_key: Array, fast_mcmc: dict[str, int]
) -> None:
    model = make_trend_inflation_model(4, common_transitory=True)
    key_data, key_fit, key_scale = random.split(rng_key, 3)
    data = Predictive(model, num_samples=1)(key_data, jnp.zeros((40, 0)))["obs"][0]
    posterior = _nuts_posterior(key_fit, model, data, fast_mcmc)
    future = Predictive(model, posterior, return_sites=["gap_vol", "common_vol"])(
        key_scale, jnp.zeros((48, 0)), data
    )
    for site in ("gap_vol", "common_vol"):
        horizon_scale = future[site][:, 40:]
        assert not bool(jnp.all(horizon_scale == horizon_scale[:, :1]))


def test_trend_inflation_us_smoke(rng_key: Array, fast_mcmc: dict[str, int]) -> None:
    rates = load_us_inflation().rates[-60:]
    model = make_trend_inflation_model(rates.shape[1], common_transitory=True)
    key_fit, key_forecast = random.split(rng_key)
    posterior = _nuts_posterior(key_fit, model, rates, fast_mcmc)
    pred = forecast(key_forecast, model, posterior, rates, jnp.zeros((68, 0)))
    assert pred.shape == (fast_mcmc["num_samples"], 8, 4)
    assert bool(jnp.all(jnp.isfinite(pred)))
