"""Tests for the empirical CRPS implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from jaxtyping import TypeCheckError

from numpyro_forecast.metrics import crps_empirical


def _brute_force_crps(pred: Array, truth: Array) -> Array:
    """Reference O(n^2) CRPS: E|X-y| - 0.5 E|X-X'|."""
    term1 = jnp.abs(pred - truth).mean(axis=0)
    pairwise = jnp.abs(pred[:, None] - pred[None, :]).mean(axis=(0, 1))
    return term1 - 0.5 * pairwise


def test_crps_matches_brute_force(rng_key: Array) -> None:
    pred = random.normal(rng_key, (200, 3, 4))
    truth = random.normal(random.PRNGKey(7), (3, 4))
    got = crps_empirical(pred, truth)
    expected = _brute_force_crps(pred, truth)
    assert got.shape == (3, 4)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_crps_large_sample_no_int_overflow(rng_key: Array) -> None:
    # With n past ~46k both the rank weight ``i * (n - i)`` and the ``n ** 2``
    # normalization overflow int32 if computed as integers, which silently
    # corrupts (or raises in) the CRPS. Compare against an overflow-free float64
    # NumPy reference built from the same sorted-sample identity.
    n = 100_001
    pred = random.normal(rng_key, (n, 1))
    truth = jnp.array([0.3])
    got = crps_empirical(pred, truth)

    pred_np = np.asarray(pred, dtype=np.float64)
    truth_np = np.asarray(truth, dtype=np.float64)
    pred_sorted = np.sort(pred_np, axis=0)
    diff = pred_sorted[1:] - pred_sorted[:-1]
    i = np.arange(1, n, dtype=np.float64)
    weight = (i * (n - i))[:, None]
    absolute_error = np.abs(pred_np - truth_np).mean(axis=0)
    reference = absolute_error - (diff * weight).sum(axis=0) / n**2

    assert bool(jnp.all(got >= 0.0))
    assert jnp.allclose(got, jnp.asarray(reference), atol=1e-4)


def test_crps_deterministic_prediction_is_absolute_error() -> None:
    # All samples identical -> the dispersion term vanishes -> CRPS = |c - y|.
    pred = jnp.full((50, 2), 3.0)
    truth = jnp.array([1.0, 5.0])
    got = crps_empirical(pred, truth)
    assert jnp.allclose(got, jnp.array([2.0, 2.0]), atol=1e-6)


def test_crps_is_nonnegative(rng_key: Array) -> None:
    pred = random.normal(rng_key, (100, 5))
    truth = random.normal(random.PRNGKey(1), (5,))
    assert bool(jnp.all(crps_empirical(pred, truth) >= -1e-6))


def test_crps_shape_mismatch_raises() -> None:
    # The jaxtyping/beartype import hook enforces the shared ``*batch`` axis
    # between ``pred`` and ``truth`` at call time, before the manual check.
    with pytest.raises(TypeCheckError):
        crps_empirical(jnp.zeros((10, 3)), jnp.zeros((4,)))


def test_crps_needs_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2 samples"):
        crps_empirical(jnp.zeros((1, 3)), jnp.zeros((3,)))
