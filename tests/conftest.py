"""Shared fixtures for numpyro_forecast tests."""

import jax.numpy as jnp
import pytest
from jax import Array, random


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
def sample_panel() -> Array:
    """Short synthetic panel series shaped ``(time, n_series)``."""
    t = jnp.linspace(0, 4 * jnp.pi, 60)[:, None]
    return jnp.sin(t) + 0.1 * random.normal(random.PRNGKey(1), (60, 4))


@pytest.fixture
def fast_svi() -> dict[str, int]:
    """Minimal SVI settings for fast tests."""
    return {"num_steps": 50}


@pytest.fixture
def fast_mcmc() -> dict[str, int]:
    """Minimal MCMC settings for fast tests."""
    return {"num_warmup": 50, "num_samples": 50, "num_chains": 1}
