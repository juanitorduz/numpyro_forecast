"""Shared fixtures for numpyro_forecast tests."""

import types
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal

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
    forecast,
    forecasting_model,
    predict,
    predict_in_sample,
    time_series,
)
from numpyro_forecast.typing import ForecastFn, ForecastModel, InSampleFn

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


def as_autoguide(guide: "AutoGuide | Callable[..., None]") -> AutoGuide:
    """Narrow a resolved guide to ``AutoGuide`` for ty (test helper).

    ``SVIFit.guide`` is typed as ``AutoGuide | Callable[..., None]`` (hand-written
    guides are supported at runtime); tests that know the guide is a resolved
    ``AutoGuide`` use this to narrow it before passing it to the guide-only
    :func:`~numpyro_forecast.functional.posterior.draw_posterior`.
    """
    assert isinstance(guide, AutoGuide)
    return guide


def svi_guide_params(t: int, num_steps: int = 40) -> tuple[AutoNormal, dict[str, Array]]:
    """Fit the shared random-walk body with raw NumPyro SVI (test helper).

    Deliberately built on plain NumPyro rather than :func:`fit_svi`/``SVIFit``
    (scheduled for removal), so tests that only need a guide/params pair for
    :func:`~numpyro_forecast.functional.posterior.draw_posterior` exercise the
    guide-based contract directly instead of the fit-wrapper machinery.
    """
    model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    result = svi.run(random.PRNGKey(1), num_steps, empty_covariates(t), data, progress_bar=False)
    return guide, result.params


def nuts_samples(
    t: int, num_warmup: int = 20, num_samples: int = 20, num_chains: int = 1
) -> dict[str, Array]:
    """Fit the shared random-walk body with raw NumPyro NUTS (test helper).

    Deliberately built on plain NumPyro rather than :func:`fit_mcmc`/``MCMCFit``
    (scheduled for removal), returning ``mcmc.get_samples()`` directly (flattened,
    ``group_by_chain=False``): this is the posterior dict MCMC users pass straight
    to :func:`~numpyro_forecast.functional.prediction.forecast`/``predict_in_sample``/
    :func:`~numpyro_forecast.convert.to_datatree`, with no ``draw_posterior`` step
    in between. ``chain_method="sequential"`` mirrors :func:`~numpyro_forecast.functional.mcmc.fit_mcmc`'s
    default, so ``num_chains > 1`` runs on a single CPU device.
    """
    model = forecasting_model(rw_body)
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), empty_covariates(t), data)
    return mcmc.get_samples()


def svi_forecast_fn(num_steps: int = 30) -> ForecastFn:
    """Build a ``backtest``-compatible :data:`ForecastFn` closure from plain NumPyro.

    Fits ``model`` with ``AutoNormal`` + ``SVI.run`` and forecasts with the
    package's :func:`~numpyro_forecast.functional.forecast`. Deliberately built on
    plain NumPyro rather than the OOP ``Forecaster``/``fit_svi``/``draw_posterior``
    (scheduled for removal), so backtest tests exercise the closure contract
    directly instead of the machinery being replaced.
    """

    def forecast_fn(
        rng_key: Array,
        model: ForecastModel,
        train_data: Array,
        train_covariates: Array,
        test_covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> "Array | np.ndarray":
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        key_fit, key_post, key_pred = random.split(rng_key, 3)
        state = svi.run(key_fit, num_steps, train_covariates, train_data, progress_bar=False)
        posterior = guide.sample_posterior(key_post, state.params, sample_shape=(num_samples,))
        return forecast(
            key_pred, model, posterior, train_data, test_covariates, batch_size=batch_size
        )

    return forecast_fn


def svi_in_sample_fn(num_steps: int = 30) -> InSampleFn:
    """Build a ``backtest``-compatible :data:`InSampleFn` closure from plain NumPyro.

    Mirrors :func:`svi_forecast_fn` but scores the in-sample fit with the
    package's :func:`~numpyro_forecast.functional.predict_in_sample`.
    """

    def in_sample_fn(
        rng_key: Array,
        model: ForecastModel,
        train_data: Array,
        train_covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> "Array | np.ndarray":
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        key_fit, key_post, key_pred = random.split(rng_key, 3)
        state = svi.run(key_fit, num_steps, train_covariates, train_data, progress_bar=False)
        posterior = guide.sample_posterior(key_post, state.params, sample_shape=(num_samples,))
        return predict_in_sample(
            key_pred, model, posterior, train_covariates, batch_size=batch_size
        )

    return in_sample_fn


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
