"""Tests for the batched ACF/PACF diagnostics."""

import contextlib
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import commit_host
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


@pytest.mark.parametrize("eager", [False, True], ids=["jit", "eager"])
@pytest.mark.parametrize("time", [10, 30])
@pytest.mark.parametrize("level", [0.0, 1.0, 3.3, 5.0, -7.25, 10_000.0])
def test_constant_series_is_nan(level: float, time: int, eager: bool) -> None:
    # A zero-variance series has no autocorrelation: every lag is a 0 / 0 in
    # the default (jitted) call and under jax.disable_jit() alike, whatever the
    # level, so it can never read as a persistent AR(1)-like spectrum.
    y = jnp.full((time,), level)
    mode = jax.disable_jit() if eager else contextlib.nullcontext()
    with mode:
        acf_values = np.asarray(acf(y, max_lag=3))
        pacf_values = np.asarray(pacf(y, max_lag=3))
    assert np.isnan(acf_values).all(), acf_values
    assert np.isnan(pacf_values).all(), pacf_values


def test_constant_rows_of_a_panel_are_nan(rng_key: Array) -> None:
    # Every constant row is NaN regardless of its level, through batching and
    # through jax.vmap, and the NaN rows do not leak into the non-degenerate
    # row of the same batch.
    noise = random.normal(rng_key, (30,))
    panel = jnp.stack([jnp.zeros(30), jnp.full(30, 1.0), jnp.full(30, 5.0), noise])
    for fn in (acf, pacf):
        got = np.asarray(fn(panel, max_lag=2))
        mapped = np.asarray(jax.vmap(lambda s, fn=fn: fn(s, max_lag=2))(panel))
        assert np.isnan(got[:3]).all(), got
        assert np.isnan(mapped[:3]).all(), mapped
        np.testing.assert_allclose(got[3], np.asarray(fn(noise, max_lag=2)), atol=1e-6)
        np.testing.assert_allclose(mapped[3], got[3], atol=1e-6)


@pytest.mark.parametrize(("level", "bump"), [(0.0, 1.0), (5.0, 1.0), (5.0, 1e-3), (10_000.0, 1.0)])
def test_near_constant_series_is_finite(level: float, bump: float) -> None:
    # Guard against over-rejection (passes before and after the constant
    # series fix): one changed sample makes the variance nonzero, so the
    # spectrum is finite and close to the float64 oracle, also when the bump
    # is small compared with the level (a variance threshold would give NaN).
    y = jnp.full((30,), level).at[10].add(bump)
    acf_values = acf(y, max_lag=5)
    pacf_values = pacf(y, max_lag=5)
    assert bool(jnp.all(jnp.isfinite(acf_values)))
    assert bool(jnp.all(jnp.isfinite(pacf_values)))
    assert jnp.allclose(acf_values, jnp.asarray(_np_acf(np.asarray(y), 5)), atol=1e-3)
    assert jnp.allclose(pacf_values, jnp.asarray(_np_pacf(np.asarray(y), 5)), atol=1e-3)


def test_wrong_input_typecheck_error() -> None:
    with pytest.raises(TypeCheckError):
        acf(jnp.array(1.0), max_lag=1)
    # Integer counts are accepted (#163); a boolean series is still rejected.
    with pytest.raises(TypeCheckError):
        pacf(jnp.arange(10) > 4, max_lag=3)


_COUNTS = np.array([200, 3, 0, 2, 5, 1, 0, 4, 2, 1, 3, 7, 0, 2, 6])
"""A count series whose first sample is far above the rest.

`_acf` shifts the series by its first sample, so in ``uint8`` the shift
``y - y[..., :1]`` wraps for every later sample.
"""


@pytest.mark.parametrize(
    ("dtype", "container"),
    [
        (np.uint8, "jax"),
        (np.int16, "jax"),
        (np.int32, "jax"),
        (np.uint32, "jax"),
        (np.int64, "numpy"),
    ],
    ids=["uint8", "int16", "int32", "uint32", "numpy-int64"],
)
@pytest.mark.parametrize(("fn", "oracle"), [(acf, _np_acf), (pacf, _np_pacf)], ids=["acf", "pacf"])
def test_integer_counts_match_float_oracle(
    fn: Callable[..., Array],
    oracle: Callable[[np.ndarray, int], np.ndarray],
    dtype: type[np.integer],
    container: str,
) -> None:
    """Integer counts (a Poisson ``predict()`` draw) give the float spectrum (#163)."""
    counts = _COUNTS.astype(dtype)
    y = jnp.asarray(counts) if container == "jax" else counts
    got = fn(y, max_lag=4)
    assert got.dtype == jnp.result_type(float)
    expected = fn(jnp.asarray(_COUNTS, jnp.float32), max_lag=4)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    np.testing.assert_allclose(got, oracle(_COUNTS, 4), atol=1e-4)


def test_integer_batch_is_vmappable_and_jittable() -> None:
    panel = jnp.stack([jnp.asarray(_COUNTS, jnp.int32), jnp.asarray(_COUNTS[::-1], jnp.int32)])
    for fn in (acf, pacf):
        expected = fn(panel.astype(jnp.float32), max_lag=3)
        mapped = jax.vmap(lambda s, fn=fn: fn(s, max_lag=3))(panel)
        jitted = jax.jit(fn, static_argnames=("max_lag",))(panel, max_lag=3)
        np.testing.assert_allclose(mapped, expected, atol=1e-6)
        np.testing.assert_allclose(jitted, expected, atol=1e-6)


def test_dtype_preserved(rng_key: Array) -> None:
    y = random.normal(rng_key, (50,), dtype=jnp.float32)
    assert acf(y, max_lag=5).dtype == jnp.float32
    assert pacf(y, max_lag=5).dtype == jnp.float32


@pytest.mark.parametrize("kind", ["pinned", "cpu"])
def test_host_committed_series_stays_committed_and_matches_device_series(
    monkeypatch: pytest.MonkeyPatch, rng_key: Array, kind: str
) -> None:
    """A host-committed series (``device="host"`` draws) is scored where it lives.

    The ``cpu`` case simulates an accelerator default backend: the integer
    cast happens on the CPU and the result stays committed there, so a large
    host-offloaded panel is never moved to the accelerator.
    """
    y = random.normal(rng_key, (2, 40))
    counts = jnp.asarray(_COUNTS, jnp.int32)
    y_host, counts_host = commit_host(y, kind), commit_host(counts, kind)
    if kind == "cpu":
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for fn in (acf, pacf):
        np.testing.assert_allclose(fn(y_host, max_lag=4), fn(y, max_lag=4))
        got = fn(counts_host, max_lag=4)
        np.testing.assert_allclose(got, fn(counts.astype(jnp.float32), max_lag=4))
        if kind == "cpu":
            assert got.committed
            assert got.devices() == {jax.devices("cpu")[0]}
