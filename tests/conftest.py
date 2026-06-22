"""Shared fixtures for numpyro_forecast tests."""

from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random

from numpyro_forecast.forecaster import (
    Forecaster,
    ForecastingModel,
    HMCForecaster,
    _BaseForecaster,
)
from numpyro_forecast.typing import ForecastModel


class RandomWalkModel(ForecastingModel):
    """Local-level random walk with Normal observation noise (shared by tests)."""

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series("drift", lambda: dist.Normal(0.0, drift_scale))
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


def empty_covariates(duration: int) -> Array:
    """Return a ``(duration, 0)`` covariate array (no exogenous features)."""
    return jnp.zeros((duration, 0))


@pytest.fixture
def rng_key() -> Array:
    """A deterministic PRNG key."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_univariate() -> Array:
    """Short synthetic univariate series shaped ``(time, 1)``."""
    t = jnp.linspace(0, 4 * jnp.pi, 60)
    y = jnp.sin(t) + 0.1 * random.normal(random.PRNGKey(0), (60,))
    return y[:, None]


@pytest.fixture
def fast_svi() -> dict[str, int]:
    """Minimal SVI settings for fast tests."""
    return {"num_steps": 50}


@pytest.fixture
def fast_mcmc() -> dict[str, int]:
    """Minimal MCMC settings for fast tests."""
    return {"num_warmup": 50, "num_samples": 50, "num_chains": 1}


@pytest.fixture(params=["svi", "nuts"])
def forecaster_factory(
    request: pytest.FixtureRequest,
    fast_svi: dict[str, int],
    fast_mcmc: dict[str, int],
) -> Callable[..., _BaseForecaster]:
    """Build a fitted forecaster with either SVI or NUTS, using fast settings.

    Parametrized over both inference backends so a single test exercises a model
    under ``Forecaster`` (SVI) and ``HMCForecaster`` (NUTS).
    """
    if request.param == "svi":

        def make_svi(
            model: ForecastModel, data: Array, covariates: Array, *, rng_key: Array
        ) -> _BaseForecaster:
            return Forecaster(
                model, data, covariates, rng_key=rng_key, num_steps=fast_svi["num_steps"]
            )

        return make_svi

    def make_nuts(
        model: ForecastModel, data: Array, covariates: Array, *, rng_key: Array
    ) -> _BaseForecaster:
        return HMCForecaster(
            model,
            data,
            covariates,
            rng_key=rng_key,
            num_warmup=fast_mcmc["num_warmup"],
            num_samples=fast_mcmc["num_samples"],
            num_chains=fast_mcmc["num_chains"],
        )

    return make_nuts
