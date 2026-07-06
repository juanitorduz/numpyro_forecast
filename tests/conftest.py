"""Shared fixtures for numpyro_forecast tests."""

import types
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import jax
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
from numpyro_forecast.functional import (
    Horizon,
    MCMCFit,
    SVIFit,
    fit_mcmc,
    fit_svi,
    forecasting_model,
    predict,
    time_series,
)
from numpyro_forecast.typing import ForecastModel

# ---------------------------------------------------------------------------
# Compile-count harness (roadmap §4.5). A single process-wide JAX monitoring
# listener counts backend compilations; the ``count_compilations`` fixture
# exposes a context manager that reports the delta over a tightly scoped block.
# Pre-create every array OUTSIDE the block (array constructors compile too) and
# call ``block_until_ready`` INSIDE it so the compile is attributed correctly.
# ---------------------------------------------------------------------------

_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


class _CompileCounter:
    """Process-wide backend-compilation tally."""

    def __init__(self) -> None:
        self.total = 0


_COMPILE_COUNTER = _CompileCounter()
_HARNESS_AVAILABLE = False


def _install_compile_listener() -> None:
    """Register the JAX monitoring listener that feeds :data:`_COMPILE_COUNTER`."""
    global _HARNESS_AVAILABLE

    def _listener(event: str, duration_secs: float, **_: object) -> None:
        if event == _BACKEND_COMPILE_EVENT:
            _COMPILE_COUNTER.total += 1

    try:
        jax.monitoring.register_event_duration_secs_listener(_listener)
    except Exception:  # harness unavailability is non-fatal
        _HARNESS_AVAILABLE = False
    else:
        _HARNESS_AVAILABLE = True


_install_compile_listener()


@pytest.fixture
def count_compilations() -> Callable[[], AbstractContextManager[types.SimpleNamespace]]:
    """Return a factory of context managers that count backend compilations.

    Usage::

        with count_compilations() as tally:
            jax.block_until_ready(jitted(x))
        assert tally.count == 1

    When the monitoring backend is unavailable the block imperatively xfails
    (non-strict), never a silent skip (roadmap §4.5).
    """

    @contextmanager
    def _tracker() -> Iterator[types.SimpleNamespace]:
        if not _HARNESS_AVAILABLE:  # pragma: no cover - backend-dependent
            pytest.xfail("compile-count harness unavailable on this JAX backend")
        obj = types.SimpleNamespace(count=0)
        start = _COMPILE_COUNTER.total
        try:
            yield obj
        finally:
            obj.count = _COMPILE_COUNTER.total - start

    return _tracker


@pytest.fixture
def sample_hierarchical() -> Array:
    """Short synthetic hierarchical series shaped ``(group, time, 1)``.

    Three groups sharing a seasonal shape with per-group level offsets; the
    layout matches the package contract (batch left, time at ``-2``, obs at
    ``-1``).
    """
    time = jnp.linspace(0, 4 * jnp.pi, 40)
    base = jnp.sin(time)
    offsets = jnp.array([-1.0, 0.0, 1.0])[:, None]
    noise = 0.1 * random.normal(random.PRNGKey(7), (3, 40))
    series = offsets + base[None, :] + noise
    return series[..., None]


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


def rw_body(h: Horizon, covariates: Array) -> None:
    """Random-walk model body using the functional primitives (shared test helper)."""
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = time_series(h, "drift", lambda: dist.Normal(0.0, drift_scale))
    predict(h, dist.Normal(0.0, sigma), jnp.cumsum(drift, axis=-2))


def svi_fit(t: int, num_steps: int = 40) -> SVIFit:
    """Fit the shared random-walk body with SVI on a synthetic series (test helper)."""
    model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    return fit_svi(random.PRNGKey(1), model, data, empty_covariates(t), num_steps=num_steps)


def mcmc_fit(t: int, num_warmup: int = 20, num_samples: int = 20) -> MCMCFit:
    """Fit the shared random-walk body with MCMC on a synthetic series (test helper)."""
    model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    return fit_mcmc(
        random.PRNGKey(1),
        model,
        data,
        empty_covariates(t),
        num_warmup=num_warmup,
        num_samples=num_samples,
    )


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
            rng_key: Array, model: ForecastModel, data: Array, covariates: Array
        ) -> _BaseForecaster:
            return Forecaster(rng_key, model, data, covariates, num_steps=fast_svi["num_steps"])

        return make_svi

    def make_nuts(
        rng_key: Array, model: ForecastModel, data: Array, covariates: Array
    ) -> _BaseForecaster:
        return HMCForecaster(
            rng_key,
            model,
            data,
            covariates,
            num_warmup=fast_mcmc["num_warmup"],
            num_samples=fast_mcmc["num_samples"],
            num_chains=fast_mcmc["num_chains"],
        )

    return make_nuts
