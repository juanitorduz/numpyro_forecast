"""Tests for the forecasting base class and SVI/MCMC forecasters."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array, random
from numpyro.handlers import seed, trace

from numpyro_forecast.forecaster import Forecaster, ForecastingModel, HMCForecaster


class _RandomWalkModel(ForecastingModel):
    """Local-level random walk with Normal observation noise (for tests)."""

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series("drift", lambda: dist.Normal(0.0, drift_scale))
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


def _empty_covariates(duration: int) -> Array:
    return jnp.zeros((duration, 0))


def test_predict_sites_training(rng_key: Array) -> None:
    model = _RandomWalkModel()
    data = jnp.zeros((20, 1))
    tr = trace(seed(model, rng_key)).get_trace(_empty_covariates(20), data)
    assert tr["drift"]["value"].shape == (20, 1)
    assert "obs" in tr
    assert "obs_future" not in tr
    assert "forecast" not in tr


def test_predict_sites_forecast(rng_key: Array) -> None:
    model = _RandomWalkModel()
    data = jnp.zeros((20, 1))
    tr = trace(seed(model, rng_key)).get_trace(_empty_covariates(25), data)
    # In-sample site keeps its training shape; the horizon uses a separate site.
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert tr["forecast"]["value"].shape == (5, 1)


def test_state_unavailable_outside_call() -> None:
    model = _RandomWalkModel()
    try:
        _ = model.future
    except RuntimeError as err:
        assert "model state" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


def test_forecaster_svi_shape_and_guide_not_resized(rng_key: Array) -> None:
    model = _RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (50, 1)), axis=-2)
    forecaster = Forecaster(model, data, _empty_covariates(50), num_steps=60, rng_key=rng_key)
    # Guide latents are sized to the in-sample length.
    post = forecaster.guide.sample_posterior(rng_key, forecaster.params, sample_shape=(4,))
    assert post["drift"].shape == (4, 50, 1)

    # Fit once, forecast at two different horizons without refitting.
    k1, k2 = random.split(rng_key)
    fc10 = forecaster(data, _empty_covariates(60), num_samples=32, rng_key=k1)
    fc5 = forecaster(data, _empty_covariates(55), num_samples=16, rng_key=k2)
    assert fc10.shape == (32, 10, 1)
    assert fc5.shape == (16, 5, 1)
    assert bool(jnp.all(jnp.isfinite(fc10)))


def test_forecaster_batch_size_matches_single_shot(rng_key: Array) -> None:
    model = _RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(model, data, _empty_covariates(40), num_steps=40, rng_key=rng_key)
    fc = forecaster(data, _empty_covariates(46), num_samples=10, rng_key=rng_key, batch_size=3)
    assert fc.shape == (10, 6, 1)


def test_forecaster_conditions_on_data(rng_key: Array) -> None:
    # A roughly constant series at level 5: forecasts must track it (not the
    # zero-mean prior), which is only possible if the posterior conditions on it.
    model = _RandomWalkModel()
    data = 5.0 + 0.01 * random.normal(rng_key, (60, 1))
    forecaster = Forecaster(model, data, _empty_covariates(60), num_steps=600, rng_key=rng_key)
    fc = forecaster(data, _empty_covariates(66), num_samples=200, rng_key=rng_key)
    median_first_step = float(jnp.median(fc[:, 0, 0]))
    assert 3.0 < median_first_step < 7.0


def test_hmc_forecaster_shape(rng_key: Array) -> None:
    model = _RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    forecaster = HMCForecaster(
        model, data, _empty_covariates(30), num_warmup=20, num_samples=20, rng_key=rng_key
    )
    fc = forecaster(data, _empty_covariates(36), num_samples=10, rng_key=rng_key)
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))
