"""Tests for the functional model-building primitives (``functional.models``)."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import RandomWalkModel, empty_covariates, rw_body
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.functional import (
    Horizon,
    fit_mcmc,
    forecast,
    forecasting_model,
    predict,
    predict_glm,
    time_series,
)
from numpyro_forecast.surgery import shift_loc
from numpyro_forecast.typing import Array, ForecastModel


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
    tr = trace(seed(lambda: rw_body(h, covariates), random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (20, 1)
    assert "obs" in tr
    assert "drift_future" not in tr
    assert "obs_future" not in tr
    assert "forecast" not in tr


def test_time_series_predict_forecast_sites() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    tr = trace(seed(lambda: rw_body(h, covariates), random.PRNGKey(0))).get_trace()
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
    func_model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (20, 1)), axis=-2)
    _assert_traces_equal(func_model, RandomWalkModel(), empty_covariates(20), data)


def test_forecasting_model_matches_oop_forecast_trace() -> None:
    func_model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (20, 1)), axis=-2)
    _assert_traces_equal(func_model, RandomWalkModel(), empty_covariates(25), data)


def test_forecasting_model_prior_sampling() -> None:
    # data=None: pure prior sampling. The whole horizon is in-sample, so "obs" is
    # sampled (not observed) and there are no forecast-horizon sites.
    func_model = forecasting_model(rw_body)
    tr = trace(seed(func_model, random.PRNGKey(0))).get_trace(empty_covariates(15))
    assert tr["drift"]["value"].shape == (15, 1)
    assert tr["obs"]["is_observed"] is False
    assert "drift_future" not in tr
    assert "forecast" not in tr


# --- P13: predict_glm + predict refactor (I9) --------------------------------


def _predict_glm_body(h: Horizon, covariates: Array) -> None:
    """Random-walk body written directly with predict_glm and a shift_loc link."""
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = time_series(h, "drift", lambda: dist.Normal(0.0, drift_scale))
    predict_glm(h, lambda mu: shift_loc(dist.Normal(0.0, sigma), mu), jnp.cumsum(drift, axis=-2))


def _traces_equal(model_a: ForecastModel, model_b: ForecastModel, *args: Array) -> None:
    key = random.PRNGKey(0)
    trace_a = trace(seed(model_a, key)).get_trace(*args)
    trace_b = trace(seed(model_b, key)).get_trace(*args)
    assert set(trace_a) == set(trace_b)
    for name, site_a in trace_a.items():
        site_b = trace_b[name]
        assert site_a["type"] == site_b["type"]
        if "value" in site_a and site_a["value"] is not None:
            assert jnp.allclose(site_a["value"], site_b["value"])
        if site_a["type"] == "sample":
            la = site_a["fn"].log_prob(site_a["value"])
            lb = site_b["fn"].log_prob(site_b["value"])
            assert jnp.allclose(la, lb)


@pytest.mark.parametrize("future", [0, 6])
def test_predict_predict_glm_trace_equivalence(future: int) -> None:
    """Invariant I9: predict == predict_glm o shift_loc (identical traces)."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (24, 1)), axis=-2)
    covariates = empty_covariates(24 + future)
    model_predict = forecasting_model(rw_body)
    model_glm = forecasting_model(_predict_glm_body)
    _traces_equal(model_predict, model_glm, covariates, data)


def test_predict_glm_rejects_float_data_for_discrete_obs() -> None:
    def poisson_body(h: Horizon, covariates: Array) -> None:
        rate = time_series(h, "log_rate", lambda: dist.Normal(0.0, 1.0))
        predict_glm(h, lambda eta: dist.Poisson(jnp.exp(eta)), jnp.cumsum(rate, axis=-2))

    model = forecasting_model(poisson_body)
    float_data = jnp.abs(random.normal(random.PRNGKey(0), (12, 1)))
    with pytest.raises(ValueError, match="discrete support"):
        trace(seed(model, random.PRNGKey(1))).get_trace(empty_covariates(12), float_data)


def test_predict_glm_accepts_integer_data_for_discrete_obs() -> None:
    def poisson_body(h: Horizon, covariates: Array) -> None:
        rate = time_series(h, "log_rate", lambda: dist.Normal(0.0, 1.0))
        predict_glm(h, lambda eta: dist.Poisson(jnp.exp(eta)), jnp.cumsum(rate, axis=-2))

    model = forecasting_model(poisson_body)
    int_data = jnp.asarray(random.poisson(random.PRNGKey(0), 3.0, (12, 1)), dtype=jnp.int32)
    tr = trace(seed(model, random.PRNGKey(1))).get_trace(empty_covariates(12), int_data)
    assert "obs" in tr


def test_poisson_local_level_end_to_end() -> None:
    """A Poisson GLM local level: forecasts are integer, non-negative, and track the rate."""

    def poisson_body(h: Horizon, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 0.5))
        log_rate = time_series(h, "log_rate", lambda: dist.Normal(0.0, drift_scale))
        predict_glm(h, lambda eta: dist.Poisson(jnp.exp(eta)), jnp.cumsum(log_rate, axis=-2))

    model = forecasting_model(poisson_body)
    true_rate = 5.0
    data = jnp.asarray(random.poisson(random.PRNGKey(0), true_rate, (40, 1)), dtype=jnp.int32)
    fit = fit_mcmc(
        random.PRNGKey(1),
        model,
        data,
        empty_covariates(40),
        num_warmup=200,
        num_samples=200,
    )
    # MCMC posterior samples (mcmc.get_samples()) go straight to forecast(), with
    # no draw_posterior step (that's guide-based only).
    post = fit.samples
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(46))
    assert fc.shape == (200, 6, 1)
    assert bool(jnp.all(fc >= 0.0))
    assert bool(jnp.all(fc == jnp.floor(fc)))  # integer-valued counts
    # The forecast median should be in the right ballpark of the true rate.
    assert 2.0 < float(jnp.median(fc)) < 10.0
