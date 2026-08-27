"""Tests for producing draws (``numpyro_forecast.predictive``) and the offload helpers."""

import os
import subprocess
import sys
import warnings
from collections.abc import Callable, Mapping
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    assert_host_resident,
    assert_numpy_host,
    assert_pinned_host_resident,
    commit_host,
    empty_covariates,
    fail_devices_for,
    rw_model,
    svi_guide_params,
)
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal

import numpyro_forecast
from numpyro_forecast._offload import (
    _WARNING_SKIP_PREFIXES,
    _device_view,
    _draw_chunked,
    _host_memory_kind,
    _host_sharding,
    _is_cpu_committed,
    _is_host_resident,
    _leaf_view,
    _oom_advice,
    _resolve_device,
    _stitch_chunks,
    _transfer,
)
from numpyro_forecast.exceptions import DeviceMemoryError
from numpyro_forecast.predictive import (
    _chunk_indices,
    _chunked_draws,
    _predict,
    _sample_axis_size,
    draw_posterior,
    forecast,
    predict_in_sample,
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
    output flows straight into `forecast()`, the compatibility invariant that
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
    # One transfer per chunk, then the stitched result: CPU-committed chunks are
    # stitched through NumPy and committed back with a zero-copy device_put, so
    # the last entry is a commit of host memory, not a fourth accelerator transfer.
    assert transfers == [(4, 1), (4, 1), (4, 1), (10, 1)]


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


# --- P7: host offloading (``device="host"``: CPU device, NumPy fallback, explicit pinned)


@pytest.mark.parametrize("device", ["host", "cpu"])
def test_resolve_device_host_resolves_to_the_cpu_device(device: str) -> None:
    """``"host"`` (and ``"cpu"``) resolve to the CPU backend device when it exists.

    That device is pageable host memory: it has no pinned-pool cap, which is
    what makes ``device="host"`` viable for a 50K-series posterior on a GPU host.
    """
    assert _resolve_device(device) == jax.devices("cpu")[0]


_FALLBACK_WARNING = r"falls back to device='host'.*NumPy.*set_platform\('cuda,cpu'\)"


def test_resolve_device_host_without_cpu_backend_is_numpy_and_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a CPU backend ``"host"`` is the backend-free NumPy path, with no warning.

    ``numpyro.set_platform("cuda")`` sets ``jax_platforms`` and leaves the CPU
    backend uninitialized. A CUDA client offers no pageable ``jax.Array``
    container, so ``"host"`` copies each chunk with ``jax.device_get`` instead
    (PR #65): fully functional, no pinned-pool cap, hence nothing to warn about.
    """
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = _resolve_device("host")
    assert resolved == "numpy"
    assert isinstance(_transfer(jnp.arange(4.0), resolved), np.ndarray)


def test_resolve_device_cpu_without_cpu_backend_falls_back_to_numpy_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``"cpu"`` request that cannot be honored warns and takes the NumPy path."""
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    with pytest.warns(UserWarning, match=_FALLBACK_WARNING):
        resolved = _resolve_device("cpu")
    assert resolved == "numpy"


@pytest.mark.parametrize("sentinel", ["pinned_host", "numpy"])
def test_resolved_sentinels_pass_through_silently(sentinel: str) -> None:
    """Resolved sentinels (also explicit requests) are idempotent and never warn.

    ``to_datatree`` resolves once and hands the result to both predictive
    drivers, which resolve again; a non-idempotent sentinel would re-warn or
    re-resolve differently.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_device(sentinel) == sentinel


def test_is_host_resident_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_is_host_resident`` is the one switch behind ``_leaf_view``/``_device_view``.

    A pinned leaf is host-resident on every backend. A CPU-committed leaf is
    host-resident only when the default backend is an accelerator: on a
    CPU-only machine every array lives on a CPU device, and staging them would
    be pure overhead. An uncommitted leaf is never host-resident.
    """
    x = jnp.arange(6.0).reshape(3, 2)
    pinned = commit_host(x, "pinned")
    on_cpu = commit_host(x, "cpu")

    assert _is_cpu_committed(on_cpu)
    assert not _is_cpu_committed(x)
    assert _is_host_resident(pinned)
    assert not _is_host_resident(on_cpu)  # real CPU backend: nothing to keep off
    assert not _is_host_resident(x)

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _is_host_resident(pinned)
    assert _is_host_resident(on_cpu)
    assert not _is_host_resident(x)


def test_leaf_view_stages_cpu_committed_leaves_as_numpy_under_accelerator_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On an accelerator host a CPU-committed posterior must be gathered in NumPy.

    Indexing a CPU-committed leaf with an uncommitted index array runs the
    gather on the CPU and returns a CPU-committed chunk, which would then drag
    the jitted predictive onto the CPU; a NumPy chunk is placed on the default
    device instead. The view must share the buffer (no copy of a 14 GiB site).
    """
    x = jnp.arange(6.0).reshape(3, 2)
    on_cpu = commit_host(x, "cpu")
    assert _leaf_view(on_cpu) is on_cpu  # real CPU backend: untouched

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    viewed = _leaf_view(on_cpu)
    assert isinstance(viewed, np.ndarray)
    assert np.array_equal(viewed, np.asarray(x))
    assert viewed.__array_interface__["data"][0] == on_cpu.unsafe_buffer_pointer()


def test_device_view_returns_uncommitted_copy_for_cpu_committed_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU-committed metric operand comes back as an *uncommitted* default-device array.

    Uncommitted is deliberate: it follows ``jax.default_device`` and whichever
    other operand is committed, whereas committing to ``jax.devices()[0]`` would
    raise against a truth committed to another accelerator.
    """
    x = jnp.arange(6.0).reshape(3, 2)
    on_cpu = commit_host(x, "cpu")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    viewed = _device_view(on_cpu)
    assert isinstance(viewed, jax.Array)
    assert not viewed.committed
    assert jnp.array_equal(viewed, x)


@pytest.mark.parametrize(
    "chunk_shape", [(8,), (4_096, 32)], ids=["scalar-site-96-bytes", "half-MiB-chunks"]
)
def test_stitch_chunks_on_cpu_committed_chunks_is_zero_copy_and_stays_on_cpu(
    chunk_shape: tuple[int, ...],
) -> None:
    """Stitching CPU-committed chunks stays on the CPU device and shares the staged buffer.

    The host peak is chunks plus one stitched copy; a silent alignment-triggered
    copy in the final ``device_put`` would double it, so the result must alias
    the NumPy staging array. jax's CPU client aliases a NumPy buffer only when
    it is 64-byte aligned, which neither small malloc blocks (a scalar site's
    draws) nor glibc's mmap'd large blocks guarantee, so the staging buffer
    must be allocated aligned rather than left to the allocator. ``chunks`` is
    cleared so the per-chunk buffers can be released before the next site is
    stitched.
    """
    cpu = jax.devices("cpu")[0]
    rows = chunk_shape[0]
    num_samples = 3 * rows - 2  # cut an overdraw so the slice path is covered too
    chunks: list[Array | np.ndarray] = [
        commit_host(jnp.arange(float(np.prod(chunk_shape))).reshape(chunk_shape) + 10.0 * i, "cpu")
        for i in range(3)
    ]
    expected = np.concatenate([np.asarray(c) for c in chunks], axis=0)[:num_samples]
    staged: list[np.ndarray] = []
    real_device_put = jax.device_put

    def spy_device_put(x: object, device: object = None, **kwargs: object) -> Array:
        if isinstance(x, np.ndarray):
            staged.append(x)
        return real_device_put(x, device, **kwargs)

    with mock.patch.object(jax, "device_put", spy_device_put):
        stitched = _stitch_chunks(chunks, num_samples, cpu)

    assert isinstance(stitched, jax.Array)
    assert stitched.committed
    assert stitched.devices() == {cpu}
    assert stitched.shape == (num_samples, *chunk_shape[1:])
    assert np.array_equal(np.asarray(stitched), expected)
    assert chunks == []
    assert len(staged) == 1
    assert staged[0].ctypes.data % 64 == 0
    assert stitched.unsafe_buffer_pointer() == staged[0].ctypes.data


def test_oom_advice_names_pinned_pool_for_host_memory_errors() -> None:
    """A pinned-pool OOM must not be reported as an accelerator OOM.

    Only an explicit ``device="pinned_host"`` (or the caller's own pinned
    arrays) uses that pool, so the host branch names the cap, points at
    ``device="host"`` (pageable memory), and skips the device budget line.
    """
    msg = (
        "RESOURCE_EXHAUSTED: Out of host memory while trying to allocate 15200000000 bytes "
        "with allocator xla_gpu_host_bfc on device 0."
    )
    with pytest.raises(DeviceMemoryError) as excinfo:
        with _oom_advice("posterior drawing", 250):
            raise RuntimeError(msg)
    text = str(excinfo.value)
    assert "pinned host memory" in text
    assert "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB" in text
    assert "device='host'" in text
    assert "set_platform" not in text
    assert "batch_size (currently 250)" in text
    assert "device memory budget" not in text
    assert "RESOURCE_EXHAUSTED" in str(excinfo.value.__cause__)


def test_host_committed_mixing_semantics() -> None:
    """Pin the documented mixing rule for CPU-committed results on CPU-only CI.

    Mixed with an *uncommitted* array an op runs on the committed CPU device
    and returns a CPU-committed array; mixed with an array committed to another
    device it raises. Two forced host devices stand in for CPU plus accelerator.
    """
    script = """
import jax, jax.numpy as jnp
d0, d1 = jax.devices("cpu")[:2]
a = jax.device_put(jnp.arange(3.0), d0)
mixed = a + jnp.arange(3.0)
assert mixed.committed and mixed.devices() == {d0}, (mixed.committed, mixed.devices())
b = jax.device_put(jnp.arange(3.0), d1)
try:
    a + b
except ValueError:
    print("RAISED")
else:
    print("NO_RAISE")
"""
    result = subprocess.run(  # noqa: S603 - our own interpreter and script, no user input
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"XLA_FLAGS": "--xla_force_host_platform_device_count=2", "PATH": ""},
    )
    assert result.stdout.strip() == "RAISED"


def test_resolve_device_missing_platform_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jax, "devices", fail_devices_for("tpu"))
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


def test_leaf_view_stages_pinned_leaves_as_numpy() -> None:
    """``_leaf_view`` is the host-gather switch: a pinned (fallback) leaf becomes NumPy.

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


def test_chunked_draws_gathers_pinned_posterior_chunks_on_the_host() -> None:
    """A pinned (fallback) posterior must be indexed as NumPy, not as a jax Array.

    Indexing a pinned ``jax.Array`` with a device-resident index array raises
    (``memory_space of all inputs ... must be the same``), so ``_chunked_draws``
    stages every leaf through ``_leaf_view`` first, which keeps the gather on
    the host. Spying on the chunks actually handed to ``predict_fn`` is what
    pins that: drop the ``_leaf_view`` call and these leaves come back as
    ``jax.Array``.
    """
    chunk_types: list[type] = []

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        chunk_types.append(type(post["x"]))
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": commit_host(jnp.arange(10.0)[:, None], "pinned")}

    _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    assert len(chunk_types) == 3
    assert all(issubclass(t, np.ndarray) for t in chunk_types), chunk_types

    # The single-shot passthrough stages the posterior the same way.
    chunk_types.clear()
    _chunked_draws(random.PRNGKey(0), predict_fn, posterior, None)
    assert [issubclass(t, np.ndarray) for t in chunk_types] == [True]


def test_chunked_draws_gathers_cpu_committed_posterior_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On an accelerator host a CPU-committed posterior is gathered in NumPy too.

    Without the staging the gather (committed leaf, uncommitted index) and then
    the jitted predictive would silently run on the CPU.
    """
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    chunk_types: list[type] = []

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        chunk_types.append(type(post["x"]))
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": commit_host(jnp.arange(10.0)[:, None], "cpu")}
    _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    assert len(chunk_types) == 3
    assert all(issubclass(t, np.ndarray) for t in chunk_types), chunk_types


def test_chunked_draws_cpu_device_is_host_resident_and_matches_values() -> None:
    """The resolved host target changes only where draws live, never what they are.

    ``_chunked_draws`` takes the *resolved* device (``forecast`` and
    ``predict_in_sample`` resolve ``"host"`` first), so the CPU device is passed
    directly. Ten samples in chunks of four also exercises the overdraw slice:
    the third chunk wraps around, and the stitched result must be cut back to
    ten rows that match the ``device=None`` result element for element.
    """

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    plain = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    hosted = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4, jax.devices("cpu")[0])
    assert_host_resident(hosted)
    assert hosted.shape == (10, 1)  # the 12 drawn rows are cut back to 10
    assert np.array_equal(np.asarray(plain), np.asarray(hosted))


def test_chunked_draws_numpy_sentinel_returns_numpy_and_matches_values() -> None:
    """The resolved ``"numpy"`` sentinel inside ``_chunked_draws`` is the backend-free path."""

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    plain = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    hosted = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4, "numpy")
    assert isinstance(hosted, np.ndarray)
    assert hosted.shape == (10, 1)  # the 12 drawn rows are cut back to 10
    assert np.array_equal(np.asarray(plain), hosted)


def test_chunked_draws_numpy_transfers_each_chunk_via_device_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory contract of the NumPy path: one ``device_get`` per chunk, nothing pinned."""
    gets: list[tuple[int, ...]] = []
    puts: list[object] = []
    real_device_get = jax.device_get
    real_device_put = jax.device_put

    def spy_device_get(x: Array) -> np.ndarray:
        gets.append(tuple(np.shape(x)))
        return real_device_get(x)

    def spy_device_put(x: Array, device: object = None, **kwargs: object) -> Array:
        puts.append(device)
        return real_device_put(x, device, **kwargs)

    monkeypatch.setattr(jax, "device_get", spy_device_get)
    monkeypatch.setattr(jax, "device_put", spy_device_put)
    posterior = {"x": jnp.arange(10.0)[:, None]}
    _chunked_draws(
        random.PRNGKey(0), lambda _key, post: jnp.asarray(post["x"]), posterior, 4, "numpy"
    )
    assert gets == [(4, 1), (4, 1), (4, 1)]  # np.concatenate is not a transfer
    assert not any(_is_cpu_target(d) or _is_pinned_target(d) for d in puts)


def test_chunked_draws_pinned_sentinel_is_pinned_and_matches_values() -> None:
    """The resolved ``"pinned_host"`` sentinel inside ``_chunked_draws`` is the pinned fallback."""

    def predict_fn(key: Array, post: Mapping[str, "Array | np.ndarray"]) -> Array:
        return jnp.asarray(post["x"]) * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    plain = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    hosted = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4, "pinned_host")
    assert_pinned_host_resident(hosted)
    assert hosted.shape == (10, 1)
    assert np.array_equal(np.asarray(plain), np.asarray(hosted))


def _is_cpu_target(device: object) -> bool:
    """Whether a ``jax.device_put`` target is the CPU device or a sharding on it."""
    if isinstance(device, jax.Device):
        return device.platform == "cpu"
    if isinstance(device, jax.sharding.SingleDeviceSharding):
        return all(d.platform == "cpu" for d in device.device_set)
    return False


def _is_pinned_target(device: object) -> bool:
    """Whether a ``jax.device_put`` target is a pinned/unpinned host memory sharding."""
    return isinstance(device, jax.sharding.SingleDeviceSharding) and device.memory_kind in (
        "pinned_host",
        "unpinned_host",
    )


@pytest.mark.parametrize(
    ("device", "is_target"),
    [
        pytest.param("cpu_device", _is_cpu_target, id="cpu-device"),
        pytest.param("pinned_host", _is_pinned_target, id="pinned-sentinel"),
    ],
)
def test_chunked_draws_host_transfers_each_chunk(
    monkeypatch: pytest.MonkeyPatch, device: str, is_target: Callable[[object], bool]
) -> None:
    """Memory contract of both host paths: every chunk is moved off the accelerator as drawn."""
    transfers: list[tuple[int, ...]] = []
    real_device_put = jax.device_put
    target: jax.Device | str = jax.devices("cpu")[0] if device == "cpu_device" else device

    def spy_device_put(x: Array, device: object = None, **kwargs: object) -> Array:
        if is_target(device):
            transfers.append(tuple(np.shape(x)))
        return real_device_put(x, device, **kwargs)

    monkeypatch.setattr(jax, "device_put", spy_device_put)
    posterior = {"x": jnp.arange(10.0)[:, None]}
    _chunked_draws(
        random.PRNGKey(0),
        lambda _key, post: jnp.asarray(post["x"]),
        posterior,
        4,
        target,
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


def test_draw_chunked_numpy_returns_numpy_and_matches_values() -> None:
    """The shared draw driver's NumPy path: same values, NumPy leaves, correct overdraw cut."""

    def draw_fn(key: Array, n: int) -> dict[str, Array]:
        return {"a": random.normal(key, (n, 2)), "b": random.uniform(key, (n,))}

    key = random.PRNGKey(11)
    plain = _draw_chunked(key, draw_fn, 7, batch_size=3, device=None, stage="test drawing")
    hosted = _draw_chunked(key, draw_fn, 7, batch_size=3, device="numpy", stage="test drawing")
    assert_numpy_host(hosted)
    assert hosted["a"].shape == (7, 2)
    assert hosted["b"].shape == (7,)
    for name, leaf in plain.items():
        assert np.array_equal(np.asarray(leaf), hosted[name])


def test_draw_posterior_host_without_cpu_backend_is_numpy_and_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public backend-free contract: no CPU backend means NumPy leaves, no warning, same draws."""
    _model, _data, guide, params = _fit_data()
    plain = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)

    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hosted = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4, device="host")
    assert_numpy_host(hosted)
    for name, leaf in plain.items():
        assert np.array_equal(np.asarray(leaf), hosted[name])


def test_forecast_host_without_cpu_backend_returns_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hosted = forecast(
            random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="host"
        )
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)


def test_predict_in_sample_host_without_cpu_backend_returns_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4)
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hosted = predict_in_sample(
            random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4, device="host"
        )
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)


def test_forecast_explicit_pinned_host_stays_pinned() -> None:
    """``device="pinned_host"`` is the only way onto the pinned pool, and it is honored."""
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    pinned = forecast(
        random.PRNGKey(3),
        model,
        post,
        data,
        empty_covariates(36),
        batch_size=3,
        device="pinned_host",
    )
    assert_pinned_host_resident(pinned)
    assert np.array_equal(np.asarray(plain), np.asarray(pinned))


def test_forecast_explicit_numpy_device_returns_numpy() -> None:
    """``device="numpy"`` forces the backend-free NumPy path even with a CPU backend present."""
    model, data, guide, params = _fit_data()
    post = draw_posterior(random.PRNGKey(2), guide, params, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    hosted = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="numpy"
    )
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)


def test_cpu_fallback_warning_attributes_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``"cpu"`` fallback warning points at user code, not at the jaxtyping wrappers.

    Every package function is wrapped by the jaxtyping import hook, so a plain
    ``stacklevel`` lands in ``jaxtyping/_decorator.py`` and Python's per-location
    deduplication never fires; the warning must skip the package and the
    wrapper frames so it is attributed to (and deduplicated per) the call site.
    """
    _model, _data, guide, params = _fit_data()
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        hosted = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4, device="cpu")
    matching = [r for r in records if "falls back to device='host'" in str(r.message)]
    assert [r.filename for r in matching] == [__file__]
    assert_numpy_host(hosted)
    # The package prefix must be exactly the package directory: one level too high
    # would skip the whole checkout (tests included) and misattribute the warning.
    assert _WARNING_SKIP_PREFIXES[0] == os.path.dirname(numpyro_forecast.__file__)


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
