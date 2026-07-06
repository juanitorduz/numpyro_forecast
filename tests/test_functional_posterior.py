"""Tests for drawing posterior samples from fits (``functional.posterior``)."""

import jax.numpy as jnp
import pytest
from conftest import mcmc_fit, svi_fit
from jax import random

from numpyro_forecast.functional import MCMCFit, draw_posterior


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
