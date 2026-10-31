"""Tests for the seasonal feature builders."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from conftest import svi_forecast_fn
from jax import Array, random

from numpyro_forecast import Horizon, backtest, predict
from numpyro_forecast.features import fourier_features, periodic_repeat


def _expected_fourier(duration: int, period: float, num_terms: int) -> np.ndarray:
    """Build the reference Fourier design matrix in float64 NumPy."""
    angles = 2.0 * np.pi * np.arange(1, num_terms + 1)[None, :] * np.arange(duration)[:, None]
    return np.concatenate([np.sin(angles / period), np.cos(angles / period)], axis=-1)


def test_fourier_features_shape_and_values() -> None:
    feats = fourier_features(duration=12, period=12.0, num_terms=3)
    assert feats.shape == (12, 6)
    # First column is sin(2*pi*1*t/12); at t=0 it is 0, at t=3 it is 1.
    assert jnp.allclose(feats[0, 0], 0.0, atol=1e-6)
    assert jnp.allclose(feats[3, 0], 1.0, atol=1e-6)


def test_fourier_features_first_called_inside_jit_stays_usable() -> None:
    # The memo is process-global, so this test uses an argument tuple that no other
    # test builds: an eager first call elsewhere would hide a value cached under a trace.
    duration, period, num_terms = 23, 6.5, 2
    expected = _expected_fourier(duration, period, num_terms)
    traced = jax.jit(lambda x: x + fourier_features(duration, period, num_terms))(jnp.zeros(()))
    # The tolerance covers the float32 cast of the result, not a numerical claim.
    np.testing.assert_allclose(traced, expected, atol=1e-5)
    # A later eager call and a retrace (new input shape) reuse the memo entry.
    np.testing.assert_allclose(fourier_features(duration, period, num_terms), expected, atol=1e-5)
    retraced = jax.jit(lambda x: x + fourier_features(duration, period, num_terms))(
        jnp.zeros((1, 1))
    )
    np.testing.assert_allclose(retraced, expected, atol=1e-5)


def test_fourier_features_follows_x64_flag() -> None:
    duration, period, num_terms = 19, 6.5, 3
    expected = _expected_fourier(duration, period, num_terms)
    with jax.enable_x64(False):
        assert fourier_features(duration, period, num_terms).dtype == jnp.float32
    with jax.enable_x64(True):
        feats = fourier_features(duration, period, num_terms)
        assert feats.dtype == jnp.float64
        np.testing.assert_allclose(feats, expected, rtol=0.0, atol=1e-12)
    with jax.enable_x64(False):
        assert fourier_features(duration, period, num_terms).dtype == jnp.float32


def test_backtest_with_fourier_features_built_from_the_horizon() -> None:
    # A model that builds its seasonal design from ``h.duration`` first calls
    # ``fourier_features`` under a trace (``svi.run``, the jitted ``forecast``). The
    # second window trains on the duration the first window forecast over, so it
    # reuses a memo entry created inside a trace. The period is unique to this test.
    def model(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        beta = jnp.asarray(numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([2]).to_event(1)))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        seasonal = fourier_features(h.duration, 5.5, 1) @ beta
        predict(h, dist.Normal(0.0, sigma), seasonal[:, None])

    data = jnp.sin(2.0 * jnp.pi * jnp.arange(28.0) / 5.5)[:, None]
    results = backtest(
        random.PRNGKey(0),
        lambda: model,
        data,
        jnp.zeros((28, 0)),
        forecast_fn=svi_forecast_fn(num_steps=5),
        min_train_window=14,
        test_window=7,
        stride=7,
        num_samples=10,
    )
    assert [(r.t1, r.t2) for r in results] == [(14, 21), (21, 28)]
    assert all(np.isfinite(v) for r in results for v in r.metrics.values())


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
