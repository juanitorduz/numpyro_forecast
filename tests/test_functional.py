"""Tests for the functional forecasting API."""

import jax
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
    _pad_posterior,
    _predict,
    draw_posterior,
    fit_mcmc,
    fit_svi,
    forecast,
    forecasting_model,
    predict,
    predict_glm,
    predict_in_sample,
    time_series,
)
from numpyro_forecast.typing import Array, ForecastModel
from numpyro_forecast.util import shift_loc


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


def test_draw_posterior_rejects_unsupported_fit_type() -> None:
    # The singledispatch fallback rejects anything that is not a known fit.
    with pytest.raises(NotImplementedError, match="does not support"):
        draw_posterior(random.PRNGKey(0), object(), 4)


def test_draw_posterior_mcmc_thins_without_replacement() -> None:
    # Fewer draws requested than the chain holds: thin on an evenly spaced grid,
    # which is deterministic, strictly increasing, and duplicate-free.
    fit = MCMCFit(samples={"x": jnp.arange(10.0)[:, None]})
    post = draw_posterior(random.PRNGKey(0), fit, 5)
    values = post["x"][:, 0]
    assert post["x"].shape == (5, 1)
    assert len(set(values.tolist())) == 5
    assert bool(jnp.all(jnp.diff(values) > 0))


def test_draw_posterior_mcmc_equal_returns_every_draw() -> None:
    # Requesting exactly the chain length returns the draws unchanged, in order.
    fit = MCMCFit(samples={"x": jnp.arange(6.0)[:, None]})
    post = draw_posterior(random.PRNGKey(0), fit, 6)
    assert jnp.array_equal(post["x"][:, 0], jnp.arange(6.0))


def test_draw_posterior_mcmc_oversample_resamples_with_replacement() -> None:
    # More draws requested than available: resample with replacement, drawing
    # only from the existing posterior values.
    fit = MCMCFit(samples={"x": jnp.arange(4.0)[:, None]})
    post = draw_posterior(random.PRNGKey(0), fit, 16)
    assert post["x"].shape == (16, 1)
    assert set(post["x"][:, 0].tolist()) <= set(jnp.arange(4.0).tolist())


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


def test_forecast_parallel_matches_serial() -> None:
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    serial = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=False)
    vmapped = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=True)
    assert vmapped.shape == serial.shape == (10, 6, 1)
    assert jnp.allclose(vmapped, serial, atol=1e-4)


def test_forecast_parallel_matches_serial_batched() -> None:
    # With batch_size fixed, only the within-chunk mapping changes (vmap vs
    # lax.map), so the chunked-vmap path must match the chunked-serial path.
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    kwargs = {"batch_size": 3}
    serial = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=False, **kwargs
    )
    vmapped = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=True, **kwargs
    )
    assert jnp.allclose(vmapped, serial, atol=1e-4)


def test_forecast_rejects_covariates_not_longer() -> None:
    model, data, fit = _fit_data(num_steps=20)
    post = draw_posterior(random.PRNGKey(2), fit, 5)
    with pytest.raises(ValueError, match="covariates must extend beyond data"):
        forecast(random.PRNGKey(3), model, post, data, empty_covariates(30))


def test_predict_in_sample_shape_and_finite() -> None:
    model, _data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    obs = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30))
    assert obs.shape == (10, 30, 1)
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_predict_in_sample_batched_matches_unbatched_shape() -> None:
    model, _data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    full = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30))
    batched = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=3)
    assert batched.shape == full.shape == (10, 30, 1)
    assert bool(jnp.all(jnp.isfinite(batched)))


def test_predict_in_sample_parallel_matches_serial() -> None:
    model, _data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    serial = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), parallel=False
    )
    vmapped = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), parallel=True
    )
    assert vmapped.shape == serial.shape == (10, 30, 1)
    assert jnp.allclose(vmapped, serial, atol=1e-4)


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


def test_oop_forecaster_parallel_matches_serial() -> None:
    func_model = forecasting_model(_rw_body)
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
        assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}


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


# --- P5: chunk padding & compile discipline ----------------------------------


def test_pad_posterior_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _pad_posterior({}, 4)


def test_pad_posterior_rejects_non_positive_batch() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _pad_posterior({"x": jnp.zeros((5, 1))}, 0)


def test_pad_posterior_rejects_ragged_sample_axis() -> None:
    posterior = {"a": jnp.zeros((5, 1)), "b": jnp.zeros((6, 1))}
    with pytest.raises(ValueError, match="disagree on the sample axis"):
        _pad_posterior(posterior, 4)


def test_pad_posterior_pads_to_multiple_and_reports_original() -> None:
    posterior = {"x": jnp.arange(5.0)[:, None]}
    padded, num = _pad_posterior(posterior, 4)
    assert num == 5
    assert padded["x"].shape == (8, 1)  # next multiple of 4
    # The original prefix is preserved; pad rows wrap around from the start.
    assert jnp.array_equal(padded["x"][:5], posterior["x"])


def test_pad_posterior_no_pad_when_already_multiple() -> None:
    posterior = {"x": jnp.arange(8.0)[:, None]}
    padded, num = _pad_posterior(posterior, 4)
    assert num == 8
    assert padded["x"] is posterior["x"]


@pytest.mark.parametrize("num_samples", [1, 3, 4, 5, 12])
def test_forecast_chunked_shapes_and_finite(num_samples: int) -> None:
    # Sweep num_samples around batch_size b=4: {1, b-1, b, b+1, 3b}.
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, num_samples)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=4)
    assert fc.shape == (num_samples, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_chunked_close_to_unchunked() -> None:
    # Chunking changes the PRNG layout, so draws differ; the sample means still
    # agree within Monte Carlo error (same distribution).
    model, data, fit = _fit_data()
    n = 400
    post = draw_posterior(random.PRNGKey(2), fit, n)
    chunked = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=8)
    unchunked = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    assert chunked.shape == unchunked.shape == (n, 6, 1)
    standard_error = unchunked.std(axis=0) / jnp.sqrt(n)
    assert jnp.all(
        jnp.abs(chunked.mean(axis=0) - unchunked.mean(axis=0)) < 8.0 * standard_error + 0.05
    )


def test_single_compile_while_chunking(count_compilations) -> None:
    """Invariant I3: fixed shapes ⇒ fixed compile counts for chunked forecasts.

    Padding makes every chunk share the ``(batch_size, future, obs)`` shape, so
    the forecast kernel (`_predict`) compiles a single variant across the whole
    ``num_samples`` sweep. The compile-count harness (roadmap §4.5) then proves
    that replaying the sweep triggers zero further backend compilations once
    every shape has been seen, and the ``_predict`` cache introspection pins the
    forecast kernel specifically to one variant.
    """
    model, data, fit = _fit_data()
    covariates = empty_covariates(36)
    # Pre-build posteriors OUTSIDE the counted block (draw/JIT would compile).
    posteriors = [draw_posterior(random.PRNGKey(2), fit, n) for n in (5, 8, 12)]
    jax.block_until_ready(posteriors)

    def _sweep() -> None:
        for post in posteriors:
            jax.block_until_ready(
                forecast(random.PRNGKey(3), model, post, data, covariates, batch_size=4)
            )

    _predict.clear_cache()  # ty: ignore[unresolved-attribute]
    _sweep()  # warm-up: compiles the single fixed-shape forecast kernel
    assert _predict._cache_size() == 1  # ty: ignore[unresolved-attribute]

    # Replaying the sweep hits every cached shape, so nothing recompiles.
    with count_compilations() as tally:
        _sweep()
    assert tally.count == 0
    assert _predict._cache_size() == 1  # ty: ignore[unresolved-attribute]


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
    model_predict = forecasting_model(_rw_body)
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
    post = draw_posterior(random.PRNGKey(2), fit, 200)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(46))
    assert fc.shape == (200, 6, 1)
    assert bool(jnp.all(fc >= 0.0))
    assert bool(jnp.all(fc == jnp.floor(fc)))  # integer-valued counts
    # The forecast median should be in the right ballpark of the true rate.
    assert 2.0 < float(jnp.median(fc)) < 10.0
