"""Shared fixtures for numpyro_forecast tests."""

import types
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.functional import (
    Horizon,
    draw_posterior,
    forecast,
    predict,
    predict_in_sample,
    time_series,
)
from numpyro_forecast.functional._offload import _host_sharding
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


_HOST_MEMORY_KINDS = ("pinned_host", "unpinned_host")
"""Host memory kinds an explicit ``device="pinned_host"`` result may carry."""


def fail_devices_for(platform: str) -> Callable[[str | None], list[jax.Device]]:
    """Build a ``jax.devices`` stand-in whose ``platform`` backend is missing.

    Simulates ``numpyro.set_platform("cuda")`` (``jax_platforms`` restricted to
    the accelerator), where ``jax.devices("cpu")`` raises ``RuntimeError``.
    """
    real_devices = jax.devices

    def fake_devices(backend: str | None = None) -> list[jax.Device]:
        if backend == platform:
            msg = f"Unknown backend {platform}. Available backends are ['cuda']"
            raise RuntimeError(msg)
        return real_devices(backend)

    return fake_devices


def assert_host_resident(x: object) -> None:
    """Assert every leaf of ``x`` is a jax Array committed to the CPU backend device.

    The host-offload contract: ``device="host"`` keeps results jax-native but
    in pageable host memory, so each leaf must be a :class:`jax.Array` committed
    to ``jax.devices("cpu")[0]`` with the plain ``"device"`` memory kind (the
    pinned fallback has its own check, :func:`assert_pinned_host_resident`).

    On a CPU-only machine ``committed`` is what separates a host result from a
    ``device=None`` one: both live on the CPU device, but only the offloaded
    result is committed. Committedness propagates through gathers, jitted calls,
    and concatenation, so a stage *fed* a committed posterior returns committed
    arrays even with ``device=None``; for those stages a per-chunk transfer spy
    (see ``test_host_pipeline.py``) is the strong signal, not this helper.
    """
    cpu = jax.devices("cpu")[0]
    leaves = jax.tree_util.tree_leaves(x)
    assert leaves, "expected at least one array leaf"
    for leaf in leaves:
        assert isinstance(leaf, jax.Array), f"expected a jax.Array, got {type(leaf)}"
        assert leaf.committed, "leaf is not committed, so it is not a host-offloaded result"
        assert leaf.devices() == {cpu}, f"leaf lives on {leaf.devices()}, not the CPU device"
        assert leaf.sharding.memory_kind == "device", (
            f"leaf carries memory kind {leaf.sharding.memory_kind!r}, expected pageable 'device'"
        )


def assert_numpy_host(x: object) -> None:
    """Assert every leaf of ``x`` is a NumPy array (pageable host memory).

    The contract of the backend-free ``device="host"`` path taken when the JAX
    CPU backend is not initialized (and of an explicit ``device="numpy"``):
    each chunk is copied with ``jax.device_get``, so nothing is pinned and no
    backend beyond the accelerator's is needed.
    """
    leaves = jax.tree_util.tree_leaves(x)
    assert leaves, "expected at least one array leaf"
    for leaf in leaves:
        assert isinstance(leaf, np.ndarray), f"expected a NumPy array, got {type(leaf)}"


def assert_pinned_host_resident(x: object) -> None:
    """Assert every leaf of ``x`` is a jax Array in a pinned/unpinned host memory kind.

    The contract of an explicit ``device="pinned_host"`` request: results stay
    off the accelerator by living in its host memory kind on the accelerator's
    own device (a pool capped at 64 GB by default on CUDA).
    """
    leaves = jax.tree_util.tree_leaves(x)
    assert leaves, "expected at least one array leaf"
    for leaf in leaves:
        assert isinstance(leaf, jax.Array), f"expected a jax.Array, got {type(leaf)}"
        assert leaf.sharding.memory_kind in _HOST_MEMORY_KINDS, (
            f"leaf lives in memory kind {leaf.sharding.memory_kind!r}, not host memory"
        )


def commit_host(x: Array, kind: str) -> Array:
    """Commit ``x`` the way one of the two ``device="host"`` paths would.

    ``"cpu"`` commits to the CPU backend device (the primary path), ``"pinned"``
    to the host memory kind of ``x``'s own device (explicit ``device="pinned_host"``).
    """
    if kind == "cpu":
        return jax.device_put(x, jax.devices("cpu")[0])
    if kind == "pinned":
        return jax.device_put(x, _host_sharding(x))
    msg = f"unknown host commit kind {kind!r}"
    raise ValueError(msg)


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


def empty_covariates(duration: int) -> Array:
    """Return a ``(duration, 0)`` covariate array (no exogenous features)."""
    return jnp.zeros((duration, 0))


def rw_body(h: Horizon, covariates: Array) -> None:
    """Random-walk model body using the functional primitives (shared test helper)."""
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = time_series(h, "drift", lambda: dist.Normal(0.0, drift_scale))
    predict(h, dist.Normal(0.0, sigma), jnp.cumsum(drift, axis=-2))


def rw_model(covariates: Array, data: Array | None = None) -> None:
    """Plain-function random-walk model on :func:`rw_body` (shared by tests).

    The functional-only replacement for the former ``RandomWalkModel``
    (a subclass of the former OOP ``ForecastingModel`` base class): derives
    the :class:`Horizon` from the shapes itself via ``Horizon.from_data``, so
    every non-legacy test drives it as a plain ``(covariates, data=None)``
    callable.
    """
    rw_body(Horizon.from_data(covariates, data), covariates)


def rw_model_factory() -> ForecastModel:
    """A :data:`~numpyro_forecast.typing.ModelFactory` returning a fresh ``rw_model`` wrapper.

    ``rw_model`` itself is a plain module-level function: every ``ModelFactory``
    call would return the exact same object, unlike the former ``RandomWalkModel``
    class (a fresh instance per call). Tests that rely on ``model_fn()`` producing
    a distinct object each call (e.g. the ``reuse_model=False`` compile-cache
    invariant in ``evaluate.backtest``) use this instead of ``lambda: rw_model``.
    """

    def model(covariates: Array, data: Array | None = None) -> None:
        rw_model(covariates, data)

    return model


def svi_guide_params(t: int, num_steps: int = 40) -> tuple[AutoNormal, dict[str, Array]]:
    """Fit the shared random-walk body with raw NumPyro SVI (test helper).

    Deliberately built on plain NumPyro rather than a fit-wrapper, so tests that
    only need a guide/params pair for
    :func:`~numpyro_forecast.functional.posterior.draw_posterior` exercise the
    guide-based contract directly.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    guide = AutoNormal(rw_model)
    svi = SVI(rw_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    result = svi.run(random.PRNGKey(1), num_steps, empty_covariates(t), data, progress_bar=False)
    return guide, result.params


def nuts_samples(
    t: int, num_warmup: int = 20, num_samples: int = 20, num_chains: int = 1
) -> dict[str, Array]:
    """Fit the shared random-walk body with raw NumPyro NUTS (test helper).

    Deliberately built on plain NumPyro rather than a fit-wrapper, returning
    ``mcmc.get_samples()`` directly (flattened, ``group_by_chain=False``): this is
    the posterior dict MCMC users pass straight to
    :func:`~numpyro_forecast.functional.prediction.forecast`/``predict_in_sample``/
    :func:`~numpyro_forecast.convert.to_datatree`, with no ``draw_posterior`` step
    in between. ``chain_method="sequential"`` keeps ``num_chains > 1`` on a single
    CPU device.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    mcmc = MCMC(
        NUTS(rw_model),
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
    plain NumPyro rather than a fit-wrapper, so backtest tests exercise the
    closure contract directly.
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
    ) -> Array:
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        key_fit, key_post, key_pred = random.split(rng_key, 3)
        state = svi.run(key_fit, num_steps, train_covariates, train_data, progress_bar=False)
        posterior = guide.sample_posterior(key_post, state.params, sample_shape=(num_samples,))
        return cast(
            "Array",
            forecast(
                key_pred, model, posterior, train_data, test_covariates, batch_size=batch_size
            ),
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
    ) -> Array:
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        key_fit, key_post, key_pred = random.split(rng_key, 3)
        state = svi.run(key_fit, num_steps, train_covariates, train_data, progress_bar=False)
        posterior = guide.sample_posterior(key_post, state.params, sample_shape=(num_samples,))
        return cast(
            "Array",
            predict_in_sample(key_pred, model, posterior, train_covariates, batch_size=batch_size),
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
def posterior_factory(
    request: pytest.FixtureRequest,
    fast_svi: dict[str, int],
    fast_mcmc: dict[str, int],
) -> Callable[[Array, ForecastModel, Array, Array], dict[str, Array]]:
    """Build a posterior-sample dict with either SVI or NUTS, using fast settings.

    Parametrized over both inference backends so a single test exercises a model
    under a variational posterior (``AutoNormal`` + ``SVI.run`` + ``draw_posterior``)
    and an MCMC posterior (raw ``MCMC(NUTS(model))`` + ``get_samples()``), with the
    same ``(rng_key, model, data, covariates) -> posterior`` call signature. The
    functional-only replacement for the former ``forecaster_factory`` (which built
    an OOP ``Forecaster``/``HMCForecaster`` instead of a bare posterior dict).
    """
    num_samples = fast_mcmc["num_samples"]

    if request.param == "svi":

        def draw_svi(
            rng_key: Array, model: ForecastModel, data: Array, covariates: Array
        ) -> dict[str, Array]:
            key_fit, key_draw = random.split(rng_key)
            guide = AutoNormal(model)
            svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
            state = svi.run(key_fit, fast_svi["num_steps"], covariates, data, progress_bar=False)
            return cast(
                "dict[str, Array]", draw_posterior(key_draw, guide, state.params, num_samples)
            )

        return draw_svi

    def draw_nuts(
        rng_key: Array, model: ForecastModel, data: Array, covariates: Array
    ) -> dict[str, Array]:
        mcmc = MCMC(
            NUTS(model),
            num_warmup=fast_mcmc["num_warmup"],
            num_samples=num_samples,
            num_chains=fast_mcmc["num_chains"],
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(rng_key, covariates, data)
        return mcmc.get_samples()

    return draw_nuts
