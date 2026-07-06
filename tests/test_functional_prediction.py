"""Tests for functional predictive sampling (``functional.prediction``)."""

import jax
import jax.numpy as jnp
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
from numpyro_forecast.functional.prediction import _chunked_draws, _pad_posterior, _predict
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


def test_chunked_draws_pads_splits_and_truncates() -> None:
    """The chunk driver feeds fixed-size chunks, distinct subkeys, and truncates the pad."""
    calls: list[tuple[Array, tuple[int, ...]]] = []

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        calls.append((key, post["x"].shape))
        return post["x"] * 2.0

    posterior = {"x": jnp.arange(10.0)[:, None]}
    out = _chunked_draws(random.PRNGKey(0), predict_fn, posterior, 4)
    # 10 samples pad to 12 = 3 chunks of 4; the result is cut back to 10.
    assert out.shape == (10, 1)
    assert jnp.allclose(out, posterior["x"] * 2.0)
    assert [shape for _, shape in calls] == [(4, 1), (4, 1), (4, 1)]
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
