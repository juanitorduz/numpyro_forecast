"""Tests for drawing posterior samples from fits (``functional.posterior``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import mcmc_fit, svi_fit
from jax import random

from numpyro_forecast.functional import MCMCFit, draw_posterior
from numpyro_forecast.functional.posterior import _jitted_sample_posterior
from numpyro_forecast.typing import Array


def test_draw_posterior_svi_leading_sample_axis() -> None:
    fit = svi_fit(t=30)
    post = draw_posterior(random.PRNGKey(2), fit, 8)
    assert post["drift"].shape == (8, 30, 1)


def test_draw_posterior_rejects_non_positive() -> None:
    fit = svi_fit(t=30, num_steps=20)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        draw_posterior(random.PRNGKey(2), fit, 0)


def test_draw_posterior_mcmc_leading_sample_axis() -> None:
    fit = mcmc_fit(t=20)
    post = draw_posterior(random.PRNGKey(2), fit, 7)
    assert post["drift"].shape == (7, 20, 1)


def test_draw_posterior_rejects_unsupported_fit_type() -> None:
    # The singledispatch fallback rejects anything that is not a known fit.
    with pytest.raises(NotImplementedError, match="does not support"):
        draw_posterior(random.PRNGKey(0), object(), 4)


def test_draw_posterior_mcmc_thins_without_replacement() -> None:
    # Fewer draws requested than the chain holds: thin on an evenly spaced grid,
    # which is deterministic, strictly increasing, and duplicate-free.
    fit = MCMCFit(samples={"x": jnp.arange(10.0)[:, None]})
    post = draw_posterior(random.PRNGKey(0), fit, 5)
    values = post["x"][:, 0]
    assert post["x"].shape == (5, 1)
    assert len(set(values.tolist())) == 5
    assert bool(jnp.all(jnp.diff(values) > 0))


def test_draw_posterior_mcmc_equal_returns_every_draw() -> None:
    # Requesting exactly the chain length returns the draws unchanged, in order.
    fit = MCMCFit(samples={"x": jnp.arange(6.0)[:, None]})
    post = draw_posterior(random.PRNGKey(0), fit, 6)
    assert jnp.array_equal(post["x"][:, 0], jnp.arange(6.0))


def test_draw_posterior_mcmc_oversample_resamples_with_replacement() -> None:
    # More draws requested than available: resample with replacement, drawing
    # only from the existing posterior values.
    fit = MCMCFit(samples={"x": jnp.arange(4.0)[:, None]})
    post = draw_posterior(random.PRNGKey(0), fit, 16)
    assert post["x"].shape == (16, 1)
    assert set(post["x"][:, 0].tolist()) <= set(jnp.arange(4.0).tolist())


# --- Chunked drawing with host offload (GPU OOM in draw_posterior) --------------


def _trees_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def test_draw_posterior_chunked_shape_finite_and_truncated() -> None:
    # 10 draws in chunks of 4: the last chunk overdraws and is cut back to 10.
    fit = svi_fit(t=30)
    post = draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4)
    assert post["drift"].shape == (10, 30, 1)
    assert bool(jnp.all(jnp.isfinite(post["drift"])))


def test_draw_posterior_batch_ge_samples_bitwise_matches_default() -> None:
    # At or above the draw count the single-shot path runs: same key, same draws.
    fit = svi_fit(t=30)
    plain = draw_posterior(random.PRNGKey(2), fit, 8)
    batched = draw_posterior(random.PRNGKey(2), fit, 8, batch_size=8)
    oversized = draw_posterior(random.PRNGKey(2), fit, 8, batch_size=64)
    assert _trees_equal(plain, batched)
    assert _trees_equal(plain, oversized)


def test_draw_posterior_chunked_deterministic_given_key_and_batch() -> None:
    fit = svi_fit(t=30)
    first = draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4)
    second = draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4)
    assert _trees_equal(first, second)


def test_draw_posterior_host_returns_numpy_bitwise_match() -> None:
    # device is a placement/representation knob, never a draws knob.
    fit = svi_fit(t=30)
    plain = draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4)
    hosted = draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4, device="host")
    assert all(isinstance(leaf, np.ndarray) for leaf in hosted.values())
    assert _trees_equal(plain, hosted)


def test_draw_posterior_chunked_calls_impl_per_fixed_size_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory contract: the guide is sampled ceil(n/b) times with exactly b draws each.

    Fixed-size chunks keep the guide sampling at a single compiled shape, and
    per-chunk transfers (tested via the host path elsewhere) bound accelerator
    memory by one chunk of the posterior instead of all of it.
    """
    import numpyro_forecast.functional.posterior as posterior_mod

    calls: list[tuple[int, Array]] = []
    real_impl = posterior_mod._draw_posterior_impl

    def spy_impl(fit: object, num_samples: int, rng_key: Array) -> dict[str, Array]:
        calls.append((num_samples, rng_key))
        return real_impl(fit, num_samples, rng_key)

    monkeypatch.setattr(posterior_mod, "_draw_posterior_impl", spy_impl)
    fit = svi_fit(t=30)
    draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4)
    assert [n for n, _ in calls] == [4, 4, 4]  # ceil(10 / 4) chunks, fixed size
    raw = [tuple(int(x) for x in jnp.ravel(random.key_data(k))) for _, k in calls]
    assert len(set(raw)) == len(raw)  # one distinct subkey per chunk


def test_draw_posterior_chunked_single_compile(count_compilations) -> None:
    """The jitted guide sampling compiles once per (guide, chunk shape) and is reused.

    Every chunk shares the fixed ``batch_size`` shape, and the jitted
    ``sample_posterior`` is cached per guide instance, so a repeat of the same
    chunked draw (and any later draw with the same batch size, regardless of
    ``num_samples``) must compile nothing.
    """
    fit = svi_fit(t=30)
    jax.block_until_ready(draw_posterior(random.PRNGKey(2), fit, 10, batch_size=4))  # warm-up

    with count_compilations() as tally:
        jax.block_until_ready(draw_posterior(random.PRNGKey(3), fit, 10, batch_size=4))
    assert tally.count == 0

    # A different num_samples with the same batch size reuses the sampling
    # executable (the chunk shape, not the total, keys the compilation); only
    # trivial stitching kernels differ, so the jitted sampler stays at one entry.
    jax.block_until_ready(draw_posterior(random.PRNGKey(4), fit, 7, batch_size=4))
    sample = _jitted_sample_posterior(fit.guide)
    assert sample._cache_size() == 1  # ty: ignore[unresolved-attribute]


def test_draw_posterior_mcmc_batch_size_is_a_draws_noop() -> None:
    # MCMC thins deterministically from materialized draws: chunked selection
    # would duplicate them, so batch_size is bypassed and draws are unchanged.
    fit = MCMCFit(samples={"x": jnp.arange(10.0)[:, None]})
    plain = draw_posterior(random.PRNGKey(0), fit, 5)
    batched = draw_posterior(random.PRNGKey(0), fit, 5, batch_size=2)
    assert _trees_equal(plain, batched)


def test_draw_posterior_mcmc_host_returns_numpy() -> None:
    fit = MCMCFit(samples={"x": jnp.arange(10.0)[:, None]})
    hosted = draw_posterior(random.PRNGKey(0), fit, 5, device="host")
    assert isinstance(hosted["x"], np.ndarray)
    assert np.array_equal(hosted["x"][:, 0], np.asarray([0.0, 2.0, 4.0, 7.0, 9.0]))


def test_draw_posterior_rejects_non_positive_batch_size() -> None:
    fit = MCMCFit(samples={"x": jnp.arange(10.0)[:, None]})
    with pytest.raises(ValueError, match="batch_size must be positive"):
        draw_posterior(random.PRNGKey(0), fit, 5, batch_size=0)
