"""Tests for the seasonal feature builders."""

import jax.numpy as jnp
import numpy as np

from numpyro_forecast.features import fourier_features, periodic_repeat


def test_fourier_features_shape_and_values() -> None:
    feats = fourier_features(duration=12, period=12.0, num_terms=3)
    assert feats.shape == (12, 6)
    # First column is sin(2*pi*1*t/12); at t=0 it is 0, at t=3 it is 1.
    assert jnp.allclose(feats[0, 0], 0.0, atol=1e-6)
    assert jnp.allclose(feats[3, 0], 1.0, atol=1e-6)


def test_fourier_features_memoizes_identical_calls() -> None:
    # The design matrix is cached per (duration, period, num_terms) tuple, so an
    # identical call is served from the cache (same object), while different
    # arguments produce a distinct array.
    first = fourier_features(duration=12, period=12.0, num_terms=3)
    assert fourier_features(duration=12, period=12.0, num_terms=3) is first
    assert fourier_features(duration=12, period=12.0, num_terms=2) is not first


def test_periodic_repeat_tiles_pattern() -> None:
    season = jnp.array([1.0, 2.0, 3.0])
    repeated = periodic_repeat(season, 7)
    assert jnp.allclose(repeated, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))


def test_periodic_repeat_along_axis() -> None:
    season = jnp.arange(6.0).reshape(2, 3)  # period 3 along axis -1
    repeated = periodic_repeat(season, 5, axis=-1)
    assert repeated.shape == (2, 5)
    assert jnp.allclose(repeated[0], jnp.array([0.0, 1.0, 2.0, 0.0, 1.0]))


def test_periodic_repeat_accepts_array_like() -> None:
    # ArrayLike inputs (here a NumPy array, not a traced jax.Array) are accepted
    # and converted internally, so callers need no explicit cast.
    repeated = periodic_repeat(np.array([1.0, 2.0, 3.0]), 7)
    assert repeated.shape == (7,)
    assert jnp.allclose(repeated, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))
