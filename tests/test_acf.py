"""Tests for the batched ACF/PACF diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from jaxtyping import TypeCheckError

from numpyro_forecast.acf import acf, pacf


def _np_acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Reference float64 ACF from direct lagged sums (biased normalization)."""
    y = np.asarray(y, dtype=np.float64)
    y_centered = y - y.mean()
    denominator = np.sum(y_centered**2)
    return np.stack(
        [
            np.sum(y_centered[lag:] * y_centered[: y.size - lag]) / denominator
            for lag in range(max_lag + 1)
        ]
    )


def _np_pacf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Reference float64 PACF via the branched Durbin-Levinson recursion."""
    rho = _np_acf(y, max_lag)
    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0
    phi_prev = np.zeros(max_lag + 1)
    for k in range(1, max_lag + 1):
        if k == 1:
            phi_kk = rho[1]
        else:
            numerator = rho[k] - np.sum(phi_prev[1:k] * rho[k - 1 : 0 : -1])
            denominator = 1.0 - np.sum(phi_prev[1:k] * rho[1:k])
            phi_kk = numerator / denominator
        phi_new = phi_prev.copy()
        phi_new[k] = phi_kk
        phi_new[1:k] = phi_prev[1:k] - phi_kk * phi_prev[k - 1 : 0 : -1]
        phi_prev = phi_new
        pacf_values[k] = phi_kk
    return pacf_values


def _simulate_ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    """Simulate a float64 AR(1) series with unit-variance innovations."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def test_acf_matches_numpy_oracle(rng_key: Array) -> None:
    y = random.normal(rng_key, (200,))
    got = acf(y, max_lag=20)
    expected = _np_acf(np.asarray(y), 20)
    assert got.shape == (21,)
    assert jnp.allclose(got, jnp.asarray(expected), atol=1e-5)


def test_pacf_matches_numpy_oracle(rng_key: Array) -> None:
    y = random.normal(rng_key, (200,))
    got = pacf(y, max_lag=20)
    expected = _np_pacf(np.asarray(y), 20)
    assert got.shape == (21,)
    assert jnp.allclose(got, jnp.asarray(expected), atol=1e-4)


def test_lag_zero_is_one(rng_key: Array) -> None:
    y = random.normal(rng_key, (2, 3, 80))
    assert jnp.allclose(acf(y, max_lag=5)[..., 0], 1.0, atol=1e-6)
    assert jnp.allclose(pacf(y, max_lag=5)[..., 0], 1.0, atol=1e-6)


def test_acf_ar1_decays_like_phi_pow_lag() -> None:
    y = _simulate_ar1(phi=0.7, n=5_000)
    got = acf(jnp.asarray(y), max_lag=2)
    assert jnp.allclose(got[1], 0.7, atol=0.05)
    assert jnp.allclose(got[2], 0.49, atol=0.05)


def test_pacf_ar1_cuts_off_after_lag_one() -> None:
    y = _simulate_ar1(phi=0.7, n=5_000)
    got = pacf(jnp.asarray(y), max_lag=5)
    assert jnp.allclose(got[1], 0.7, atol=0.05)
    assert float(jnp.max(jnp.abs(got[2:6]))) < 0.06


def test_white_noise_acf_near_zero(rng_key: Array) -> None:
    n = 2_000
    y = random.normal(rng_key, (n,))
    got = acf(y, max_lag=10)
    assert float(jnp.max(jnp.abs(got[1:]))) < 3.0 / np.sqrt(n)


def test_batch_matches_per_series_loop(rng_key: Array) -> None:
    y = random.normal(rng_key, (3, 4, 128))
    for fn in (acf, pacf):
        batched = fn(y, max_lag=10)
        assert batched.shape == (3, 4, 11)
        looped = jnp.stack(
            [jnp.stack([fn(y[i, j], max_lag=10) for j in range(4)]) for i in range(3)]
        )
        assert jnp.allclose(batched, looped, atol=1e-6)


def test_vmap_consistency(rng_key: Array) -> None:
    y = random.normal(rng_key, (5, 64))
    for fn in (acf, pacf):
        direct = fn(y, max_lag=7)
        mapped = jax.vmap(lambda s, fn=fn: fn(s, max_lag=7))(y)
        assert jnp.allclose(direct, mapped, atol=1e-6)


def test_output_shapes(sample_univariate: Array, sample_hierarchical: Array) -> None:
    assert acf(sample_univariate[:, 0], max_lag=20).shape == (21,)
    assert pacf(sample_univariate[:, 0], max_lag=20).shape == (21,)
    assert acf(sample_hierarchical[..., 0], max_lag=20).shape == (3, 21)
    assert pacf(sample_hierarchical[..., 0], max_lag=20).shape == (3, 21)


def test_public_wrappers_are_jittable(rng_key: Array) -> None:
    y = random.normal(rng_key, (100,))
    for fn in (acf, pacf):
        jitted = jax.jit(fn, static_argnames=("max_lag",))
        assert jnp.allclose(jitted(y, max_lag=8), fn(y, max_lag=8), atol=1e-6)


@pytest.mark.parametrize("max_lag", [0, -1, 30, 33])
def test_rejects_max_lag_out_of_range(max_lag: int) -> None:
    y = jnp.zeros((30,))
    with pytest.raises(ValueError, match="max_lag"):
        acf(y, max_lag=max_lag)
    with pytest.raises(ValueError, match="max_lag"):
        pacf(y, max_lag=max_lag)


def test_constant_series_is_nan_in_eager_mode() -> None:
    # Under jit the mean rounds (sum * (1/n)), leaving a nonzero constant
    # residual whose normalized autocorrelation is meaningless but finite; the
    # documented NaN contract holds for eager execution only.
    y = jnp.ones((30,))
    with jax.disable_jit():
        assert bool(jnp.all(jnp.isnan(acf(y, max_lag=5))))
        assert bool(jnp.all(jnp.isnan(pacf(y, max_lag=5))))


def test_wrong_input_typecheck_error() -> None:
    with pytest.raises(TypeCheckError):
        acf(jnp.array(1.0), max_lag=1)
    with pytest.raises(TypeCheckError):
        pacf(jnp.arange(10), max_lag=3)


def test_dtype_preserved(rng_key: Array) -> None:
    y = random.normal(rng_key, (50,), dtype=jnp.float32)
    assert acf(y, max_lag=5).dtype == jnp.float32
    assert pacf(y, max_lag=5).dtype == jnp.float32
