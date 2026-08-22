"""Tests for functional predictive sampling (``functional.prediction``)."""

from collections.abc import Mapping
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_host_resident, empty_covariates, rw_model, svi_guide_params
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.exceptions import DeviceMemoryError
from numpyro_forecast.functional import (
    draw_posterior,
    forecast,
    predict_in_sample,
)
from numpyro_forecast.functional._offload import (
    _device_view,
    _draw_chunked,
    _host_memory_kind,
    _host_sharding,
    _leaf_view,
    _resolve_device,
)
from numpyro_forecast.functional.prediction import (
    _chunk_indices,
    _chunked_draws,
    _predict,
    _sample_axis_size,
)
from numpyro_forecast.typing import Array, ForecastModel


def _fit_data(
    t: int = 30, num_steps: int = 40
) -> tuple[ForecastModel, Array, AutoNormal, dict[str, Array]]:
    model = rw_model
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    guide, params = svi_guide_params(t, num_steps=num_steps)
    return model, data, guide, params


def test_forecast_accepts_raw_mcmc_get_samples() -> None:
    """Raw ``MCMC(kernel(model)).get_samples()`` forecasts, with no fit-wrapper.

    Moved from ``test_kernels.py`` (roadmap §3): proves that plain NumPyro MCMC
    output flows straight into :func:`forecast`, the compatibility invariant that
    matters now that there is no fit-wrapper between them.
    """
    data = jnp.zeros((15, 1))
    covariates = empty_covariates(15)
    mcmc = MCMC(NUTS(rw_model), num_warmup=10, num_samples=10, progress_bar=False)
    mcmc.run(random.PRNGKey(0), covariates, data)
    samples = mcmc.get_samples()
    fc = forecast(random.PRNGKey(2), rw_model, samples, data, empty_covariates(18))
    assert fc.shape == (10, 3, 1)
    assert jnp.all(jnp.isfinite(fc))


def test_forecast_shape_and_finite() -> None:
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_batched_shape_and_finite() -> None:
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    assert fc.shape == (10, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_parallel_matches_serial() -> None:
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    serial = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=False)
    vmapped = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=True)
    assert vmapped.shape == serial.shape == (10, 6, 1)
    assert jnp.allclose(vmapped, serial, atol=1e-4)


def test_forecast_parallel_matches_serial_batched() -> None:
    # With batch_size fixed, only the within-chunk mapping changes (vmap vs
    # lax.map), so the chunked-vmap path must match the chunked-serial path.
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    kwargs = {"batch_size": 3}
    serial = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=False, **kwargs
    )
    vmapped = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), parallel=True, **kwargs
    )
    assert jnp.allclose(vmapped, serial, atol=1e-4)


def test_forecast_rejects_covariates_not_longer() -> None:
    model, data, guide, params = _fit_data(num_steps=20)
    post = draw_posterior(random.PRNGKey(2), guide, params, 5)
    with pytest.raises(ValueError, match="covariates must extend beyond data"):
        forecast(random.PRNGKey(3), model, post, data, empty_covariates(30))


def test_predict_in_sample_shape_and_finite() -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    obs = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30))
    assert obs.shape == (10, 30, 1)
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_predict_in_sample_batched_matches_unbatched_shape() -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    full = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30))
    batched = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=3)
    assert batched.shape == full.shape == (10, 30, 1)
    assert bool(jnp.all(jnp.isfinite(batched)))


def test_predict_in_sample_parallel_matches_serial() -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    serial = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), parallel=False
    )
    vmapped = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), parallel=True
    )
    assert vmapped.shape == serial.shape == (10, 30, 1)
    assert jnp.allclose(vmapped, serial, atol=1e-4)


# --- P5: chunk indexing & compile discipline ----------------------------------


def test_sample_axis_size_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _sample_axis_size({})


def test_sample_axis_size_rejects_ragged_sample_axis() -> None:
    posterior = {"a": jnp.zeros((5, 1)), "b": jnp.zeros((6, 1))}
    with pytest.raises(ValueError, match="disagree on the sample axis"):
        _sample_axis_size(posterior)


def test_sample_axis_size_returns_shared_length() -> None:
    posterior = {"a": jnp.zeros((5, 1)), "b": jnp.zeros((5, 3))}
    assert _sample_axis_size(posterior) == 5


def test_chunk_indices_rejects_non_positive_batch() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _chunk_indices(5, 0)


def test_chunk_indices_wraps_final_block() -> None:
    blocks = _chunk_indices(10, 4)
    assert [tuple(int(i) for i in block) for block in blocks] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 0, 1),  # the tail wraps around to re-use leading draws
    ]


def test_chunk_indices_no_wrap_when_exact_multiple() -> None:
    blocks = _chunk_indices(8, 4)
    assert [tuple(int(i) for i in block) for block in blocks] == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_chunked_draws_splits_wraps_and_truncates() -> None:
    """The chunk driver feeds fixed-size wrapped chunks, distinct subkeys, and truncates."""
    calls: list[tuple[Array, Array]] = []

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        calls.append((key, jnp.asarray(post["x"])))
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    out = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    # 10 samples make 3 wrapped chunks of 4; the result is cut back to 10.
    assert out.shape == (10, 1)
    assert jnp.allclose(out, posterior["x"] * 2.0)
    assert [chunk.shape for _, chunk in calls] == [(4, 1), (4, 1), (4, 1)]
    # The tail chunk wraps around to the leading draws (the old padding scheme,
    # bit for bit, without the padded posterior copy).
    assert jnp.array_equal(calls[-1][1][:, 0], jnp.array([8.0, 9.0, 0.0, 1.0]))
    keys = [key for key, _ in calls] + [random.PRNGKey(0)]
    raw = [tuple(int(x) for x in jnp.ravel(random.key_data(k))) for k in keys]
    assert len(set(raw)) == len(raw)  # per-chunk subkeys distinct, parent unused


@pytest.mark.parametrize("batch_size", [None, 10, 64])
def test_chunked_draws_unchunked_passthrough(batch_size: int | None) -> None:
    """batch_size None or >= the sample count calls predict_fn once with the parent key."""
    calls: list[Array] = []

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        calls.append(key)
        return jnp.asarray(post["x"])

    posterior = {"x": jnp.arange(10.0)[:, None]}
    parent = random.PRNGKey(0)
    out = _chunked_draws(parent, predict_fn, posterior, batch_size)
    assert out.shape == (10, 1)
    assert len(calls) == 1
    assert jnp.array_equal(random.key_data(calls[0]), random.key_data(parent))


@pytest.mark.parametrize("num_samples", [1, 3, 4, 5, 12])
def test_forecast_chunked_shapes_and_finite(num_samples: int) -> None:
    # Sweep num_samples around batch_size b=4: {1, b-1, b, b+1, 3b}.
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, num_samples)
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=4)
    assert fc.shape == (num_samples, 6, 1)
    assert bool(jnp.all(jnp.isfinite(fc)))


def test_forecast_chunked_close_to_unchunked() -> None:
    # Chunking changes the PRNG layout, so draws differ; the sample means still
    # agree within Monte Carlo error (same distribution).
    model, data, guide, params = _fit_data()
    n = 400
    post = draw_posterior(random.PRNGKey(2), guide, params, n)
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
    model, data, guide, params = _fit_data()
    covariates = empty_covariates(36)
    # Pre-build posteriors OUTSIDE the counted block (draw/JIT would compile).
    posteriors = [draw_posterior(random.PRNGKey(2), guide, params, n) for n in (5, 8, 12)]
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


# --- P6: host offloading (issue #64) ------------------------------------------


def _cpu() -> jax.Device:
    return jax.devices("cpu")[0]


def test_resolve_device_none_passthrough() -> None:
    assert _resolve_device(None) is None


def test_resolve_device_accepts_platform_string_and_device() -> None:
    cpu = _cpu()
    assert _resolve_device("cpu") == cpu
    assert _resolve_device(cpu) is cpu


def test_chunked_draws_device_commits_result() -> None:
    """With ``device`` set the stitched draws are committed there, values unchanged."""

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    plain = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    committed = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4, _cpu())
    assert isinstance(committed, jax.Array)  # a jax.Device target stays a jax.Array
    assert committed.devices() == {_cpu()}
    assert jnp.array_equal(plain, committed)


def test_chunked_draws_transfers_each_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory contract: every chunk is committed to the device, not just the result.

    CPU CI cannot observe accelerator memory, so the per-chunk ``device_put``
    calls are the testable proxy for "each chunk leaves the accelerator before
    the next is drawn".
    """
    transfers: list[tuple[int, ...]] = []
    real_device_put = jax.device_put

    def spy_device_put(x: Array, device: jax.Device) -> Array:
        transfers.append(tuple(x.shape))
        return real_device_put(x, device)

    monkeypatch.setattr(jax, "device_put", spy_device_put)
    posterior = {"x": jnp.arange(10.0)[:, None]}
    _chunked_draws(
        random.PRNGKey(0), lambda _key, post: jnp.asarray(post["x"]), posterior, 4, _cpu()
    )
    assert transfers == [(4, 1), (4, 1), (4, 1)]  # one transfer per chunk


def test_forecast_device_bitwise_matches_no_device() -> None:
    # device is a placement knob, never a draws knob.
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    hosted = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="cpu"
    )
    assert isinstance(hosted, jax.Array)
    assert hosted.devices() == {_cpu()}
    assert jnp.array_equal(plain, hosted)


def test_predict_in_sample_device_bitwise_matches_no_device() -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4)
    hosted = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4, device="cpu"
    )
    assert isinstance(hosted, jax.Array)
    assert hosted.devices() == {_cpu()}
    assert jnp.array_equal(plain, hosted)


def test_forecast_device_string_alias() -> None:
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    by_string = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="cpu"
    )
    by_device = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device=_cpu()
    )
    assert jnp.array_equal(by_string, by_device)
    assert isinstance(by_string, jax.Array)
    assert isinstance(by_device, jax.Array)
    assert by_string.devices() == by_device.devices() == {_cpu()}


def test_forecast_unchunked_device_commits_result() -> None:
    # The unchunked passthrough honors device too (single transfer of the result).
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    hosted = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), device="cpu")
    assert isinstance(hosted, jax.Array)
    assert hosted.devices() == {_cpu()}
    assert jnp.array_equal(plain, hosted)


@pytest.mark.parametrize("device", ["cpu", "host"])
def test_single_compile_while_chunking_with_device(count_compilations, device: str) -> None:
    """Off-accelerator transfers must not break the fixed-shape single-compile invariant."""
    model, data, guide, params = _fit_data()
    covariates = empty_covariates(36)
    posteriors = [draw_posterior(random.PRNGKey(2), guide, params, n) for n in (5, 8, 12)]
    jax.block_until_ready(posteriors)

    def _sweep() -> None:
        for post in posteriors:
            jax.block_until_ready(
                forecast(
                    random.PRNGKey(3), model, post, data, covariates, batch_size=4, device=device
                )
            )

    _predict.clear_cache()  # ty: ignore[unresolved-attribute]
    _sweep()  # warm-up: compiles the single fixed-shape forecast kernel
    assert _predict._cache_size() == 1  # ty: ignore[unresolved-attribute]

    with count_compilations() as tally:
        _sweep()
    assert tally.count == 0
    assert _predict._cache_size() == 1  # ty: ignore[unresolved-attribute]


# --- P7: backend-free host offloading (``device="host"``) -----------------------


def _fail_devices_for(platform: str) -> "object":
    """Build a ``jax.devices`` stand-in whose ``platform`` backend is missing."""
    real_devices = jax.devices

    def fake_devices(backend: str | None = None) -> list[jax.Device]:
        if backend == platform:
            msg = f"Unknown backend {platform}. Available backends are ['cuda']"
            raise RuntimeError(msg)
        return real_devices(backend)

    return fake_devices


def test_resolve_device_host_passthrough() -> None:
    assert _resolve_device("host") == "host"


def test_resolve_device_missing_cpu_falls_back_to_host_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``device="cpu"`` degrades gracefully when only an accelerator backend exists.

    ``numpyro.set_platform("cuda")`` sets ``jax_platforms`` and leaves the CPU
    backend uninitialized; ``"cpu"`` must then resolve to the backend-free host
    path instead of crashing (the regression behind the GPU ``to_datatree``
    failure).
    """
    monkeypatch.setattr(jax, "devices", _fail_devices_for("cpu"))
    with pytest.warns(UserWarning, match="falls? back to device='host'"):
        resolved = _resolve_device("cpu")
    assert resolved == "host"


def test_resolve_device_missing_platform_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jax, "devices", _fail_devices_for("tpu"))
    with pytest.raises(ValueError, match="platform 'tpu' is not initialized"):
        _resolve_device("tpu")


def test_host_memory_kind_prefers_pinned_host() -> None:
    """The CPU backend exposes ``"pinned_host"``, which must win over ``"unpinned_host"``."""
    assert _host_memory_kind(jax.devices()[0]) == "pinned_host"


class _DeviceOnlyMemory:
    """A memory stand-in reporting the ``"device"`` kind."""

    kind = "device"


def test_host_memory_kind_raises_without_a_host_kind() -> None:
    """A device that addresses no host memory kind must raise, not silently degrade.

    ``jax.Device`` is a C-extension (nanobind) type with no Python-accessible
    constructor, so the stub is a ``MagicMock(spec=...)`` rather than a
    subclass: ``spec`` makes ``isinstance(stub, jax.Device)`` true, which is
    what the beartype-checked ``device: jax.Device`` parameter requires. Each
    call builds a fresh mock (a distinct, freshly hashed object) so
    ``_host_memory_kind``'s ``lru_cache`` cannot serve a stale answer cached
    under another test's device.
    """
    stub_device = mock.MagicMock(spec=jax.devices()[0])
    stub_device.addressable_memories.return_value = [_DeviceOnlyMemory()]

    with pytest.raises(RuntimeError, match="exposes no host memory kind"):
        _host_memory_kind(stub_device)


def test_leaf_view_stages_host_committed_leaves_as_numpy() -> None:
    """``_leaf_view`` is the host-gather switch: only host-committed leaves become NumPy.

    A device-resident array must come back as the *same object* (no copy, no
    round-trip), a host-committed one as ``np.ndarray`` so downstream indexing
    gathers on the host instead of pulling the whole leaf into device memory,
    and a NumPy input must pass straight through.
    """
    x = jnp.arange(6.0).reshape(3, 2)
    hosted = jax.device_put(x, _host_sharding(x))

    assert _leaf_view(x) is x  # device-resident: untouched, not even copied
    viewed = _leaf_view(hosted)
    assert isinstance(viewed, np.ndarray)
    assert np.array_equal(viewed, np.asarray(x))
    assert isinstance(_leaf_view(np.asarray(x)), np.ndarray)


def test_leaf_view_passes_tracers_through() -> None:
    """A tracer has no ``sharding``; ``_leaf_view`` must not reach for one.

    Without the passthrough this raises ``AttributeError`` under ``vmap``,
    which would mask the actionable ``VectorizedMetricError`` that
    ``backtest_vectorized`` raises for host-staging metrics.
    """
    seen: list[bool] = []

    def probe(row: Array) -> Array:
        viewed = _leaf_view(row)
        seen.append(viewed is row)
        return jnp.sum(jnp.asarray(viewed))

    out = jax.vmap(probe)(jnp.ones((3, 4)))
    assert seen == [True]  # the tracer itself, not a NumPy copy
    assert np.array_equal(np.asarray(out), np.full((3,), 4.0))


def test_device_view_moves_host_committed_leaves_to_device_memory() -> None:
    """``_device_view`` is the metric-kernel-safe counterpart to ``_leaf_view``.

    A host-committed leaf must come back as a device-resident ``jax.Array``
    with the same values (so a fused metric kernel never mixes memory kinds),
    a device-resident leaf must come back unchanged in value, and a NumPy
    input must come back as a jax ``Array``.
    """
    x = jnp.arange(6.0).reshape(3, 2)
    hosted = jax.device_put(x, _host_sharding(x))

    viewed = _device_view(hosted)
    assert isinstance(viewed, jax.Array)
    assert viewed.sharding.memory_kind == "device"
    assert jnp.array_equal(viewed, x)

    same = _device_view(x)
    assert isinstance(same, jax.Array)
    assert jnp.array_equal(same, x)

    from_numpy = _device_view(np.asarray(x))
    assert isinstance(from_numpy, jax.Array)
    assert jnp.array_equal(from_numpy, x)


def test_device_view_passes_tracers_through() -> None:
    """A tracer has no ``sharding``; ``_device_view`` must not reach for one."""
    seen: list[bool] = []

    def probe(row: Array) -> Array:
        viewed = _device_view(row)
        seen.append(viewed is row)
        return jnp.sum(jnp.asarray(viewed))

    out = jax.vmap(probe)(jnp.ones((3, 4)))
    assert seen == [True]  # the tracer itself, not a converted copy
    assert np.array_equal(np.asarray(out), np.full((3,), 4.0))


def test_chunked_draws_gathers_host_posterior_chunks_on_the_host() -> None:
    """A host-committed posterior must be indexed as NumPy, not as a jax Array.

    This is the correctness contract of the host path on an accelerator:
    indexing a host-committed ``jax.Array`` with a device-resident index array
    raises (``memory_space of all inputs ... must be the same``), so
    ``_chunked_draws`` stages every leaf through ``_leaf_view`` first, which
    keeps the gather on the host. Spying on the chunks actually handed to
    ``predict_fn`` is what pins that: drop the ``_leaf_view`` call and these
    leaves come back as ``jax.Array``.
    """
    chunk_types: list[type] = []

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        chunk_types.append(type(post["x"]))
        return jnp.asarray(post["x"]) * 2.0

    x = jnp.arange(10.0)[:, None]
    posterior = {"x": jax.device_put(x, _host_sharding(x))}

    _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    assert len(chunk_types) == 3
    assert all(issubclass(t, np.ndarray) for t in chunk_types), chunk_types

    # The single-shot passthrough stages the posterior the same way.
    chunk_types.clear()
    _chunked_draws(random.PRNGKey(0), predict_fn, posterior, None)
    assert [issubclass(t, np.ndarray) for t in chunk_types] == [True]


def test_chunked_draws_host_is_host_resident_and_matches_values() -> None:
    """``device="host"`` changes only where draws live, never what they are.

    Ten samples in chunks of four also exercises the overdraw slice: the third
    chunk wraps around, and the stitched result must be cut back to ten rows
    that match the ``device=None`` result element for element.
    """

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    plain = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    hosted = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4, "host")
    assert_host_resident(hosted)
    assert hosted.shape == (10, 1)  # the 12 drawn rows are cut back to 10
    assert np.array_equal(np.asarray(plain), np.asarray(hosted))


def test_chunked_draws_host_transfers_each_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory contract of the host path: every chunk is committed to host memory as drawn."""
    transfers: list[tuple[int, ...]] = []
    real_device_put = jax.device_put

    def spy_device_put(x: Array, device: object = None, **kwargs: object) -> Array:
        if isinstance(device, jax.sharding.SingleDeviceSharding) and device.memory_kind in (
            "pinned_host",
            "unpinned_host",
        ):
            transfers.append(tuple(np.shape(x)))
        return real_device_put(x, device, **kwargs)

    monkeypatch.setattr(jax, "device_put", spy_device_put)
    posterior = {"x": jnp.arange(10.0)[:, None]}
    _chunked_draws(
        random.PRNGKey(0), lambda _key, post: jnp.asarray(post["x"]), posterior, 4, "host"
    )
    # One host copy per chunk, plus the single commit of the stitched result.
    assert transfers == [(4, 1), (4, 1), (4, 1), (10, 1)]


def test_draw_chunked_host_is_host_resident_and_matches_values() -> None:
    """The shared draw driver's host path: same values, host memory, correct overdraw cut.

    ``_draw_chunked`` is the loop behind ``draw_posterior`` and the Pathfinder
    samplers, so its host contract is asserted directly on a deterministic
    ``draw_fn``: for one ``(rng_key, batch_size)``, ``device="host"`` must
    reproduce the ``device=None`` draws bit for bit while every leaf ends up
    committed to host memory. Seven samples in chunks of three overdraw by two
    rows, which the final slice must discard.
    """

    def draw_fn(key: Array, n: int) -> dict[str, Array]:
        return {"a": random.normal(key, (n, 2)), "b": random.uniform(key, (n,))}

    key = random.PRNGKey(11)
    plain = _draw_chunked(key, draw_fn, 7, batch_size=3, device=None, stage="test drawing")
    hosted = _draw_chunked(key, draw_fn, 7, batch_size=3, device="host", stage="test drawing")

    assert_host_resident(hosted)
    assert set(hosted) == {"a", "b"}
    assert hosted["a"].shape == (7, 2)  # nine drawn rows cut back to seven
    assert hosted["b"].shape == (7,)
    for name, leaf in plain.items():
        assert np.array_equal(np.asarray(leaf), np.asarray(hosted[name]))


def test_forecast_host_bitwise_matches_cpu_and_default() -> None:
    # "host" is a placement knob, never a draws knob.
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    cpu = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="cpu"
    )
    hosted = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="host"
    )
    assert_host_resident(hosted)
    assert np.array_equal(np.asarray(plain), np.asarray(hosted))
    assert np.array_equal(np.asarray(cpu), np.asarray(hosted))


def test_predict_in_sample_host_bitwise_matches_default() -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4)
    hosted = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4, device="host"
    )
    assert_host_resident(hosted)
    assert np.array_equal(np.asarray(plain), np.asarray(hosted))


def test_predictive_oom_reports_batch_size() -> None:
    """A device OOM in the predictive chunk loop is re-raised with the lever."""

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        msg = "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3800000000 bytes."
        raise RuntimeError(msg)

    posterior = {"x": jnp.arange(10.0)[:, None]}
    with pytest.raises(
        DeviceMemoryError, match=r"predictive sampling.*lower batch_size \(currently 4\)"
    ) as excinfo:
        _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    assert "RESOURCE_EXHAUSTED" in str(excinfo.value.__cause__)


def test_predictive_non_oom_errors_propagate_unchanged() -> None:
    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        msg = "boom"
        raise ValueError(msg)

    posterior = {"x": jnp.arange(10.0)[:, None]}
    with pytest.raises(ValueError, match="boom"):
        _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)


def test_forecast_numpy_posterior_bitwise_matches_jax() -> None:
    """A host-offloaded (NumPy) posterior streams back through the same draws."""
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    post_np = {name: np.asarray(leaf) for name, leaf in post.items()}
    from_jax = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    from_np = forecast(random.PRNGKey(3), model, post_np, data, empty_covariates(36), batch_size=3)
    assert np.array_equal(np.asarray(from_jax), np.asarray(from_np))


def test_predict_in_sample_numpy_posterior_bitwise_matches_jax() -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    post_np = {name: np.asarray(leaf) for name, leaf in post.items()}
    from_jax = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4
    )
    from_np = predict_in_sample(
        random.PRNGKey(3), model, post_np, empty_covariates(30), batch_size=4
    )
    assert np.array_equal(np.asarray(from_jax), np.asarray(from_np))


def test_forecast_unchunked_host_is_host_resident() -> None:
    # The unchunked passthrough honors "host" too (a single commit of the result).
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    hosted = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), device="host")
    assert_host_resident(hosted)
    assert np.array_equal(np.asarray(plain), np.asarray(hosted))
