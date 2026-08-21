"""Tests for the forecasting base class and SVI/MCMC forecasters."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import RandomWalkModel, empty_covariates
from jax import Array, random
from numpyro.handlers import seed, trace

from numpyro_forecast.forecaster import (
    Forecaster,
    ForecastingModel,
    HMCForecaster,
    _BaseForecaster,
)


def test_predict_sites_training(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.zeros((20, 1))
    tr = trace(seed(model, rng_key)).get_trace(empty_covariates(20), data)
    assert tr["drift"]["value"].shape == (20, 1)
    assert "obs" in tr
    assert "obs_future" not in tr
    assert "forecast" not in tr


def test_predict_sites_forecast(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.zeros((20, 1))
    tr = trace(seed(model, rng_key)).get_trace(empty_covariates(25), data)
    # In-sample site keeps its training shape; the horizon uses a separate site.
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert tr["forecast"]["value"].shape == (5, 1)


def test_state_unavailable_outside_call() -> None:
    model = RandomWalkModel()
    try:
        _ = model.future
    except RuntimeError as err:
        assert "model state" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


class _DurationProbeModel(ForecastingModel):
    """Records the horizon properties seen during a model call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: tuple[int, int, int] | None = None

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        self.seen = (self.duration, self.t_obs, self.future)
        drift = self.time_series("drift", lambda: dist.Normal(0.0, 1.0))
        self.predict(dist.Normal(0.0, 1.0), jnp.cumsum(drift, axis=-2))


def test_horizon_properties_during_call(rng_key: Array) -> None:
    model = _DurationProbeModel()
    data = jnp.zeros((12, 1))
    trace(seed(model, rng_key)).get_trace(empty_covariates(20), data)
    assert model.seen == (20, 12, 8)


def test_call_rejects_data_longer_than_covariates(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.zeros((20, 1))
    with pytest.raises(ValueError, match="data must not be longer than covariates"):
        trace(seed(model, rng_key)).get_trace(empty_covariates(15), data)


def test_forecaster_svi_shape_and_guide_not_resized(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (50, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(50), num_steps=60)
    # Guide latents are sized to the in-sample length.
    post = forecaster.guide.sample_posterior(rng_key, forecaster.params, sample_shape=(4,))
    assert post["drift"].shape == (4, 50, 1)

    # Fit once, forecast at two different horizons without refitting.
    k1, k2 = random.split(rng_key)
    fc10 = forecaster(k1, data, empty_covariates(60), num_samples=32)
    fc5 = forecaster(k2, data, empty_covariates(55), num_samples=16)
    assert fc10.shape == (32, 10, 1)
    assert fc5.shape == (16, 5, 1)
    assert bool(jnp.all(jnp.isfinite(fc10)))


def test_forecaster_batch_size_matches_single_shot(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(40), num_steps=40)
    fc = forecaster(rng_key, data, empty_covariates(46), num_samples=10, batch_size=3)
    assert fc.shape == (10, 6, 1)


def test_forecaster_device_bitwise_matches_no_device(rng_key: Array) -> None:
    """The device knob relocates the chunked forecast draws without changing them."""
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(40), num_steps=40)
    plain = forecaster(rng_key, data, empty_covariates(46), num_samples=10, batch_size=3)
    hosted = forecaster(
        rng_key, data, empty_covariates(46), num_samples=10, batch_size=3, device="cpu"
    )
    assert isinstance(hosted, jax.Array)  # "cpu" commits to a device, unlike "host"
    assert hosted.shape == (10, 6, 1)
    assert hosted.devices() == {jax.devices("cpu")[0]}
    assert jnp.array_equal(plain, hosted)


def test_forecaster_predict_in_sample_device_bitwise_matches_no_device(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(40), num_steps=40)
    plain = forecaster.predict_in_sample(
        rng_key, empty_covariates(40), num_samples=10, batch_size=4
    )
    hosted = forecaster.predict_in_sample(
        rng_key, empty_covariates(40), num_samples=10, batch_size=4, device="cpu"
    )
    assert isinstance(hosted, jax.Array)  # "cpu" commits to a device, unlike "host"
    assert hosted.shape == (10, 40, 1)
    assert hosted.devices() == {jax.devices("cpu")[0]}
    assert jnp.array_equal(plain, hosted)


def test_forecaster_host_device_returns_numpy_bitwise_match(rng_key: Array) -> None:
    """``device="host"`` yields a NumPy result with the exact same draws."""
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(40), num_steps=40)
    plain = forecaster(rng_key, data, empty_covariates(46), num_samples=10, batch_size=3)
    hosted = forecaster(
        rng_key, data, empty_covariates(46), num_samples=10, batch_size=3, device="host"
    )
    assert isinstance(hosted, np.ndarray)
    assert hosted.shape == (10, 6, 1)
    assert np.array_equal(np.asarray(plain), hosted)


def test_forecaster_forwards_batch_and_device_to_posterior_draw(rng_key: Array) -> None:
    """The posterior draw is chunked/offloaded with the same knobs as the predictive."""
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(40), num_steps=40)

    captured: dict[str, object] = {}
    real_draw = forecaster._draw_posterior

    def spy_draw(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_draw(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    forecaster._draw_posterior = spy_draw  # ty: ignore[invalid-assignment]
    forecaster(rng_key, data, empty_covariates(46), num_samples=10, batch_size=3, device="host")
    assert captured == {"batch_size": 3, "device": "host"}


def test_forecaster_conditions_on_data(rng_key: Array) -> None:
    # A roughly constant series at level 5: forecasts must track it (not the
    # zero-mean prior), which is only possible if the posterior conditions on it.
    model = RandomWalkModel()
    data = 5.0 + 0.01 * random.normal(rng_key, (60, 1))
    forecaster = Forecaster(rng_key, model, data, empty_covariates(60), num_steps=600)
    fc = forecaster(rng_key, data, empty_covariates(66), num_samples=200)
    median_first_step = float(jnp.median(fc[:, 0, 0]))
    assert 3.0 < median_first_step < 7.0


def test_forecaster_shape_both_backends(
    forecaster_factory: Callable[..., _BaseForecaster],
    sample_univariate: Array,
    rng_key: Array,
) -> None:
    t = 50
    train_data = sample_univariate[:t]
    key_fit, key_forecast = random.split(rng_key)
    forecaster = forecaster_factory(key_fit, RandomWalkModel(), train_data, empty_covariates(t))
    fc = forecaster(key_forecast, train_data, empty_covariates(56), num_samples=10)
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecaster_predict_in_sample_shape_and_finite(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (40, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(40), num_steps=40)
    obs = forecaster.predict_in_sample(rng_key, empty_covariates(40), num_samples=10)
    assert obs.shape == (10, 40, 1)
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_forecaster_predict_in_sample_rejects_non_positive_num_samples(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(30), num_steps=20)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        forecaster.predict_in_sample(rng_key, empty_covariates(30), num_samples=0)


def test_forecaster_predict_in_sample_rejects_duration_mismatch(
    forecaster_factory: Callable[..., _BaseForecaster],
    rng_key: Array,
) -> None:
    t = 30
    data = jnp.cumsum(0.1 * random.normal(rng_key, (t, 1)), axis=-2)
    forecaster = forecaster_factory(rng_key, RandomWalkModel(), data, empty_covariates(t))
    with pytest.raises(ValueError, match="same duration as the fitted data"):
        forecaster.predict_in_sample(rng_key, empty_covariates(t + 5), num_samples=10)


def test_hmc_forecaster_shape(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    forecaster = HMCForecaster(
        rng_key, model, data, empty_covariates(30), num_warmup=20, num_samples=20
    )
    fc = forecaster(rng_key, data, empty_covariates(36), num_samples=10)
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_rejects_covariates_not_longer(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(30), num_steps=20)
    with pytest.raises(ValueError, match="covariates must extend beyond data"):
        forecaster(rng_key, data, empty_covariates(30), num_samples=10)


def test_forecast_rejects_non_positive_num_samples(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    forecaster = Forecaster(rng_key, model, data, empty_covariates(30), num_steps=20)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        forecaster(rng_key, data, empty_covariates(36), num_samples=0)


def test_svi_forecaster_rejects_unequal_duration(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.zeros((30, 1))
    with pytest.raises(ValueError, match="equal duration"):
        Forecaster(rng_key, model, data, empty_covariates(25), num_steps=20)


def test_hmc_forecaster_rejects_unequal_duration(rng_key: Array) -> None:
    model = RandomWalkModel()
    data = jnp.zeros((30, 1))
    with pytest.raises(ValueError, match="equal duration"):
        HMCForecaster(rng_key, model, data, empty_covariates(25), num_warmup=5, num_samples=5)


def _handwritten_guide(covariates: Array, data: Array | None = None) -> None:
    """A mean-field hand-written guide for RandomWalkModel (in-sample only)."""
    t = covariates.shape[-2]
    drift_scale_loc = numpyro.param("drift_scale_loc", -1.0)
    sigma_loc = numpyro.param("sigma_loc", -1.0)
    numpyro.sample("drift_scale", dist.LogNormal(drift_scale_loc, 0.1))
    numpyro.sample("sigma", dist.LogNormal(sigma_loc, 0.1))
    drift_loc = numpyro.param("drift_loc", jnp.zeros((t, 1)))
    with numpyro.plate("time", t, dim=-2):
        numpyro.sample("drift", dist.Normal(drift_loc, 0.1))


def test_forecaster_hand_written_guide_end_to_end(rng_key: Array) -> None:
    """Regression: a Forecaster built with a hand-written guide still forecasts.

    draw_posterior became AutoGuide-only; Forecaster._draw_posterior must detect
    a hand-written guide (not an AutoGuide instance) and draw via a direct
    Predictive call instead of handing it to draw_posterior (which would raise a
    beartype TypeCheckError at forecast time).
    """
    model = RandomWalkModel()
    data = jnp.cumsum(0.1 * random.normal(rng_key, (20, 1)), axis=-2)
    forecaster = Forecaster(
        rng_key, model, data, empty_covariates(20), guide=_handwritten_guide, num_steps=5
    )
    fc = forecaster(rng_key, data, empty_covariates(24), num_samples=4)
    assert fc.shape == (4, 4, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))
