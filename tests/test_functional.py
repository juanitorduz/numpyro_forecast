"""Tests for the functional forecasting API."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import RandomWalkModel, empty_covariates
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.evaluate import backtest
from numpyro_forecast.forecaster import Forecaster, HMCForecaster
from numpyro_forecast.functional import (
    Horizon,
    MCMCFit,
    SVIFit,
    draw_posterior,
    fit_mcmc,
    fit_svi,
    forecast,
    forecasting_model,
    predict,
    time_series,
)
from numpyro_forecast.typing import Array, ForecastModel


def _rw_body(h: Horizon, covariates: Array) -> None:
    """Random-walk model body using the functional primitives (test helper)."""
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = time_series(h, "drift", lambda: dist.Normal(0.0, drift_scale))
    predict(h, dist.Normal(0.0, sigma), jnp.cumsum(drift, axis=-2))


def test_horizon_from_data_training() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, data)
    assert h.duration == 20
    assert h.t_obs == 20
    assert h.future == 0
    assert h.data is data


def test_horizon_from_data_forecast() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    assert h.duration == 25
    assert h.t_obs == 20
    assert h.future == 5


def test_horizon_from_data_prior() -> None:
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, None)
    assert h.duration == 20
    assert h.t_obs == 20
    assert h.future == 0
    assert h.data is None


def test_horizon_rejects_data_longer_than_covariates() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((15, 0))
    with pytest.raises(ValueError, match="data must not be longer than covariates"):
        Horizon.from_data(covariates, data)


def test_horizon_zero_data_shape() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    assert h.zero_data is not None
    assert h.zero_data.shape == (25, 1)


def test_horizon_zero_data_none_for_prior() -> None:
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, None)
    assert h.zero_data is None


def test_horizon_rejects_inconsistent_duration() -> None:
    with pytest.raises(ValueError, match="duration must equal t_obs \\+ future"):
        Horizon(data=None, t_obs=5, future=10, duration=20)


def test_horizon_rejects_negative_future() -> None:
    with pytest.raises(ValueError, match="t_obs and future must be non-negative"):
        Horizon(data=None, t_obs=5, future=-1, duration=4)


def test_time_series_predict_training_sites() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, data)
    tr = trace(seed(lambda: _rw_body(h, covariates), random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (20, 1)
    assert "obs" in tr
    assert "drift_future" not in tr
    assert "obs_future" not in tr
    assert "forecast" not in tr


def test_time_series_predict_forecast_sites() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    tr = trace(seed(lambda: _rw_body(h, covariates), random.PRNGKey(0))).get_trace()
    # In-sample site keeps its training shape; the horizon uses a separate site.
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert tr["forecast"]["value"].shape == (5, 1)


def test_time_series_reparam_applies() -> None:
    data = jnp.zeros((10, 1))
    covariates = jnp.zeros((10, 0))
    h = Horizon.from_data(covariates, data)

    def body() -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        drift = time_series(
            h, "drift", lambda: dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0)
        )
        predict(h, dist.Normal(0.0, 1.0), jnp.cumsum(drift, axis=-2))

    tr = trace(seed(body, random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (10, 1)
    # LocScaleReparam introduces a decentered companion site.
    assert any("decentered" in name for name in tr)


def test_predict_forecast_requires_data() -> None:
    # future > 0 but data is None: forecasting needs observed data to condition on.
    h = Horizon(data=None, t_obs=10, future=5, duration=15)
    with pytest.raises(RuntimeError, match="forecasting requires observed data"):
        trace(
            seed(lambda: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((15, 1))), random.PRNGKey(0))
        ).get_trace()


def _assert_traces_equal(
    model_a: ForecastModel, model_b: ForecastModel, covariates: Array, data: Array
) -> None:
    key = random.PRNGKey(0)
    tr_a = trace(seed(model_a, key)).get_trace(covariates, data)
    tr_b = trace(seed(model_b, key)).get_trace(covariates, data)
    assert set(tr_a) == set(tr_b)
    for name in tr_a:
        if tr_a[name].get("value") is not None:
            assert jnp.array_equal(tr_a[name]["value"], tr_b[name]["value"]), name


def test_forecasting_model_matches_oop_training_trace() -> None:
    func_model = forecasting_model(_rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (20, 1)), axis=-2)
    _assert_traces_equal(func_model, RandomWalkModel(), empty_covariates(20), data)


def test_forecasting_model_matches_oop_forecast_trace() -> None:
    func_model = forecasting_model(_rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (20, 1)), axis=-2)
    _assert_traces_equal(func_model, RandomWalkModel(), empty_covariates(25), data)


def test_forecasting_model_prior_sampling() -> None:
    # data=None: pure prior sampling. The whole horizon is in-sample, so "obs" is
    # sampled (not observed) and there are no forecast-horizon sites.
    func_model = forecasting_model(_rw_body)
    tr = trace(seed(func_model, random.PRNGKey(0))).get_trace(empty_covariates(15))
    assert tr["drift"]["value"].shape == (15, 1)
    assert tr["obs"]["is_observed"] is False
    assert "drift_future" not in tr
    assert "forecast" not in tr


def _svi_fit(t: int, num_steps: int = 40) -> SVIFit:
    model = forecasting_model(_rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    return fit_svi(random.PRNGKey(1), model, data, empty_covariates(t), num_steps=num_steps)


def test_fit_svi_returns_populated_fit() -> None:
    fit = _svi_fit(t=30, num_steps=40)
    assert isinstance(fit, SVIFit)
    assert fit.losses.shape == (40,)
    assert any("drift_scale" in name for name in fit.params)


def test_fit_svi_rejects_unequal_duration() -> None:
    model = forecasting_model(_rw_body)
    data = jnp.zeros((30, 1))
    with pytest.raises(ValueError, match="equal duration"):
        fit_svi(random.PRNGKey(0), model, data, empty_covariates(25), num_steps=10)


def test_draw_posterior_svi_leading_sample_axis() -> None:
    fit = _svi_fit(t=30)
    post = draw_posterior(random.PRNGKey(2), fit, 8)
    assert post["drift"].shape == (8, 30, 1)


def test_draw_posterior_rejects_non_positive() -> None:
    fit = _svi_fit(t=30, num_steps=20)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        draw_posterior(random.PRNGKey(2), fit, 0)


def _mcmc_fit(t: int, num_warmup: int = 20, num_samples: int = 20) -> MCMCFit:
    model = forecasting_model(_rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    return fit_mcmc(
        random.PRNGKey(1),
        model,
        data,
        empty_covariates(t),
        num_warmup=num_warmup,
        num_samples=num_samples,
    )


def test_fit_mcmc_returns_populated_fit() -> None:
    fit = _mcmc_fit(t=20, num_samples=20)
    assert isinstance(fit, MCMCFit)
    assert "drift_scale" in fit.samples
    assert fit.samples["drift"].shape[0] == 20


def test_fit_mcmc_rejects_unequal_duration() -> None:
    model = forecasting_model(_rw_body)
    data = jnp.zeros((20, 1))
    with pytest.raises(ValueError, match="equal duration"):
        fit_mcmc(
            random.PRNGKey(0),
            model,
            data,
            empty_covariates(15),
            num_warmup=5,
            num_samples=5,
        )


def test_draw_posterior_mcmc_leading_sample_axis() -> None:
    fit = _mcmc_fit(t=20)
    post = draw_posterior(random.PRNGKey(2), fit, 7)
    assert post["drift"].shape == (7, 20, 1)


def _fit_data(t: int = 30, num_steps: int = 40) -> tuple[ForecastModel, Array, SVIFit]:
    model = forecasting_model(_rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    fit = fit_svi(random.PRNGKey(1), model, data, empty_covariates(t), num_steps=num_steps)
    return model, data, fit


def test_forecast_shape_and_finite() -> None:
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_batched_shape_and_finite() -> None:
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_rejects_covariates_not_longer() -> None:
    model, data, fit = _fit_data(num_steps=20)
    post = draw_posterior(random.PRNGKey(2), fit, 5)
    with pytest.raises(ValueError, match="covariates must extend beyond data"):
        forecast(random.PRNGKey(3), model, post, data, empty_covariates(30))


# --- Interchangeability between the functional and OOP APIs -------------------


def test_functional_model_through_oop_forecaster() -> None:
    func_model = forecasting_model(_rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    forecaster = Forecaster(
        random.PRNGKey(1), func_model, data, empty_covariates(30), num_steps=30
    )
    fc = forecaster(random.PRNGKey(2), data, empty_covariates(36), num_samples=8)
    assert fc.shape == (8, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_functional_model_through_hmc_forecaster() -> None:
    func_model = forecasting_model(_rw_body)
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
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        random.PRNGKey(1),
        data,
        covariates,
        lambda: forecasting_model(_rw_body),
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=10,
        forecaster_options={"num_steps": 20},
    )
    assert len(results) == 3
    for r in results:
        assert set(r.metrics) == {"mae", "rmse", "crps"}


def test_oop_and_functional_fits_and_forecasts_are_identical() -> None:
    # Same model both ways, same keys: SVI is deterministic, so params and the
    # resulting forecast samples must match bit for bit.
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    cov_train, cov_full = empty_covariates(30), empty_covariates(36)
    key_fit, key_fc = random.PRNGKey(7), random.PRNGKey(9)

    oop = Forecaster(key_fit, RandomWalkModel(), data, cov_train, num_steps=40)
    fc_oop = oop(key_fc, data, cov_full, num_samples=8)

    func_model = forecasting_model(_rw_body)
    fit = fit_svi(key_fit, func_model, data, cov_train, num_steps=40)
    for name, value in oop.params.items():
        assert jnp.array_equal(value, fit.params[name]), name

    key_post, key_pred = random.split(key_fc)  # mirror _BaseForecaster.__call__
    post = draw_posterior(key_post, fit, 8)
    fc_func = forecast(key_pred, func_model, post, data, cov_full)
    assert jnp.array_equal(fc_oop, fc_func)
