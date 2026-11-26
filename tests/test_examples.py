"""End-to-end smoke tests for the example models."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from example_models import (
    low_frequency_basis,
    make_hierarchical_model,
    make_rts_model,
    make_trend_inflation_model,
    univariate_model,
)
from jax import Array, random
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.datasets import (
    bart_available,
    load_bart_hierarchical,
    load_bart_weekly,
    load_state_employment,
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


def test_trend_inflation_stochastic_volatility_moves_over_the_horizon(rng_key: Array) -> None:
    model = make_trend_inflation_model(4, common_transitory=True)
    future = Predictive(model, num_samples=5, return_sites=["gap_vol", "common_vol"])(
        rng_key, jnp.zeros((48, 0)), jnp.zeros((40, 4))
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


@pytest.mark.parametrize(("num_periods", "num_terms"), [(107, 3), (347, 9)])
def test_low_frequency_basis_matches_the_random_walk_eigenvectors(
    num_periods: int, num_terms: int
) -> None:
    cumsum = np.tril(np.ones((num_periods, num_periods)))
    demean = np.eye(num_periods) - 1.0 / num_periods
    eigenvalues, eigenvectors = np.linalg.eigh(demean @ cumsum @ cumsum.T @ demean.T)
    order = np.argsort(eigenvalues)[::-1][:num_terms]
    expected = eigenvectors[:, order] * np.sqrt(eigenvalues[order])
    expected = expected * np.sign(expected[0])
    basis = np.asarray(low_frequency_basis(num_periods, num_terms))
    assert basis.shape == (num_periods, num_terms)
    np.testing.assert_allclose(basis, expected, atol=1e-5 * np.sqrt(eigenvalues[order[0]]))


@pytest.mark.parametrize("outliers", [False, True])
@pytest.mark.parametrize("stochastic_volatility", [False, True])
def test_rts_prior_predictive_shapes(
    rng_key: Array, outliers: bool, stochastic_volatility: bool
) -> None:
    model = make_rts_model(
        jnp.zeros((12, 3)),
        student_t=True,
        outliers=outliers,
        stochastic_volatility=stochastic_volatility,
    )
    history = jnp.zeros((72, 3))
    prior = Predictive(model, num_samples=4)(rng_key, history)
    assert prior["obs"].shape == (4, 72, 3)
    assert prior["mu_t"].shape == (4, 72, 3)
    assert prior["phi"].shape == (4, 12, 3)
    assert prior["nu"].shape == (4, 3)
    assert ("outlier" in prior) == outliers
    assert ("log_var" in prior) == stochastic_volatility
    if stochastic_volatility:
        assert prior["log_var"].shape == (4, 72, 3)
    forecast_draws = Predictive(model, num_samples=4)(
        rng_key, jnp.concatenate([history, jnp.zeros((8, 3))]), history
    )
    assert forecast_draws["forecast"].shape == (4, 8, 3)


def test_rts_outlier_leaves_the_lag_window(rng_key: Array) -> None:
    growth = load_state_employment().growth[:84, :2]
    y_init, history = growth[:12], growth[12:]
    model = make_rts_model(y_init, student_t=True, outliers=True)
    draw = Predictive(model, num_samples=1)(rng_key, history)
    values = {name: value[0] for name, value in draw.items() if name != "obs"}
    values["outlier_eta"] = jnp.zeros_like(history)
    clean = numpyro.handlers.trace(numpyro.handlers.condition(model, data=values)).get_trace(
        history, history
    )
    spike = 50.0
    spiked = history.at[20, 0].add(spike)
    kappa = jnp.asarray(clean["kappa"]["value"])
    values["outlier_eta"] = values["outlier_eta"].at[20, 0].set(spike / kappa[0])
    with_outlier = numpyro.handlers.trace(
        numpyro.handlers.condition(model, data=values)
    ).get_trace(spiked, spiked)
    np.testing.assert_allclose(
        np.asarray(with_outlier["mu_t"]["value"]),
        np.asarray(clean["mu_t"]["value"]),
        atol=1e-3,
    )


def test_rts_state_employment_smoke(rng_key: Array, fast_mcmc: dict[str, int]) -> None:
    growth = load_state_employment().growth[:120, :4]
    y_init, history = growth[:12], growth[12:]
    model = make_rts_model(y_init, hierarchical=False)
    key_fit, key_forecast = random.split(rng_key)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=fast_mcmc["num_warmup"],
        num_samples=fast_mcmc["num_samples"],
        num_chains=fast_mcmc["num_chains"],
        progress_bar=False,
    )
    mcmc.run(key_fit, history, history)
    covariates = jnp.concatenate([history, jnp.zeros((6, 4))])
    pred = forecast(key_forecast, model, mcmc.get_samples(), history, covariates)
    assert pred.shape == (fast_mcmc["num_samples"], 6, 4)
    assert bool(jnp.all(jnp.isfinite(pred)))


def test_rts_ols_plug_in_forecast_mean_is_the_one_step_prediction(rng_key: Array) -> None:
    growth = load_state_employment().growth[:120, :3]
    y_init, history = growth[:12], growth[12:]
    num_lags, num_states = y_init.shape
    panel = jnp.concatenate([y_init, history])
    rows = jnp.arange(history.shape[0])[:, None] + (num_lags - 1 - jnp.arange(num_lags))[None, :]
    design = jnp.concatenate([jnp.ones((history.shape[0], 1, num_states)), panel[rows]], axis=1)
    coefficients = jnp.stack(
        [jnp.linalg.lstsq(design[:, :, s], history[:, s])[0] for s in range(num_states)],
        axis=1,
    )  # (13, states): intercept then the 12 lag coefficients
    intercept, phi = coefficients[0], coefficients[1:]
    residual = history - jnp.einsum("tks,ks->ts", design, coefficients)
    log_sigma2 = jnp.log(residual.var(axis=0))
    num_draws = 4_000
    posterior = {
        "mu": jnp.broadcast_to(intercept / (1.0 - phi.sum(axis=0)), (num_draws, num_states)),
        "phi": jnp.broadcast_to(phi, (num_draws, num_lags, num_states)),
        "log_omega2": jnp.broadcast_to(log_sigma2.mean(), (num_draws,)),
        "log_sigma2": jnp.broadcast_to(log_sigma2, (num_draws, num_states)),
    }
    model = make_rts_model(y_init, hierarchical=False)
    pred = forecast(
        rng_key, model, posterior, history, jnp.concatenate([history, jnp.zeros((6, 3))])
    )
    one_step = intercept + jnp.einsum("ls,ls->s", phi, history[-num_lags:][::-1])
    tolerance = 3.0 * jnp.exp(0.5 * log_sigma2) / jnp.sqrt(num_draws)
    assert bool(jnp.all(jnp.abs(pred[:, 0].mean(axis=0) - one_step) < tolerance))
