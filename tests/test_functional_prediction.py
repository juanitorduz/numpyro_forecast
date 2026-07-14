"""Tests for functional predictive sampling (``functional.prediction``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import empty_covariates, rw_body
from jax import random

from numpyro_forecast.functional import (
    SVIFit,
    draw_posterior,
    fit_svi,
    forecast,
    forecasting_model,
    predict_in_sample,
)
from numpyro_forecast.functional.prediction import (
    _chunk_indices,
    _chunked_draws,
    _predict,
    _resolve_device,
    _sample_axis_size,
)
from numpyro_forecast.typing import Array, ForecastModel


def _fit_data(t: int = 30, num_steps: int = 40) -> tuple[ForecastModel, Array, SVIFit]:
    model = forecasting_model(rw_body)
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

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        calls.append((key, post["x"]))
        return post["x"] * 2.0

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

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        calls.append(key)
        return post["x"]

    posterior = {"x": jnp.arange(10.0)[:, None]}
    parent = random.PRNGKey(0)
    out = _chunked_draws(parent, predict_fn, posterior, batch_size)
    assert out.shape == (10, 1)
    assert len(calls) == 1
    assert jnp.array_equal(random.key_data(calls[0]), random.key_data(parent))


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

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        return post["x"] * 2.0

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
    _chunked_draws(random.PRNGKey(0), lambda _key, post: post["x"], posterior, 4, _cpu())
    assert transfers == [(4, 1), (4, 1), (4, 1)]  # one transfer per chunk


def test_forecast_device_bitwise_matches_no_device() -> None:
    # device is a placement knob, never a draws knob.
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    hosted = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="cpu"
    )
    assert isinstance(hosted, jax.Array)
    assert hosted.devices() == {_cpu()}
    assert jnp.array_equal(plain, hosted)


def test_predict_in_sample_device_bitwise_matches_no_device() -> None:
    model, _data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    plain = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4)
    hosted = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4, device="cpu"
    )
    assert isinstance(hosted, jax.Array)
    assert hosted.devices() == {_cpu()}
    assert jnp.array_equal(plain, hosted)


def test_forecast_device_string_alias() -> None:
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
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
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    hosted = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), device="cpu")
    assert isinstance(hosted, jax.Array)
    assert hosted.devices() == {_cpu()}
    assert jnp.array_equal(plain, hosted)


@pytest.mark.parametrize("device", ["cpu", "host"])
def test_single_compile_while_chunking_with_device(count_compilations, device: str) -> None:
    """Off-accelerator transfers must not break the fixed-shape single-compile invariant."""
    model, data, fit = _fit_data()
    covariates = empty_covariates(36)
    posteriors = [draw_posterior(random.PRNGKey(2), fit, n) for n in (5, 8, 12)]
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


def test_chunked_draws_host_returns_numpy_and_matches_values() -> None:
    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        return post["x"] * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    plain = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    hosted = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4, "host")
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)


def test_chunked_draws_host_transfers_each_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory contract of the host path: every chunk is copied off-device via device_get."""
    transfers: list[tuple[int, ...]] = []
    real_device_get = jax.device_get

    def spy_device_get(x: Array) -> "np.ndarray":
        transfers.append(tuple(x.shape))
        return real_device_get(x)

    monkeypatch.setattr(jax, "device_get", spy_device_get)
    posterior = {"x": jnp.arange(10.0)[:, None]}
    _chunked_draws(random.PRNGKey(0), lambda _key, post: post["x"], posterior, 4, "host")
    assert transfers == [(4, 1), (4, 1), (4, 1)]  # one host copy per chunk


def test_forecast_host_bitwise_matches_cpu_and_default() -> None:
    # "host" is a placement/representation knob, never a draws knob.
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3)
    cpu = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="cpu"
    )
    hosted = forecast(
        random.PRNGKey(3), model, post, data, empty_covariates(36), batch_size=3, device="host"
    )
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)
    assert np.array_equal(np.asarray(cpu), hosted)


def test_predict_in_sample_host_bitwise_matches_default() -> None:
    model, _data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    plain = predict_in_sample(random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4)
    hosted = predict_in_sample(
        random.PRNGKey(3), model, post, empty_covariates(30), batch_size=4, device="host"
    )
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)


def test_forecast_unchunked_host_returns_numpy() -> None:
    # The unchunked passthrough honors "host" too (single device_get of the result).
    model, data, fit = _fit_data()
    post = draw_posterior(random.PRNGKey(2), fit, 10)
    plain = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36))
    hosted = forecast(random.PRNGKey(3), model, post, data, empty_covariates(36), device="host")
    assert isinstance(hosted, np.ndarray)
    assert np.array_equal(np.asarray(plain), hosted)
