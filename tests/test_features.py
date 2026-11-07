"""Tests for the seasonal feature builders."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import svi_forecast_fn
from jax import Array, random
from jaxtyping import TypeCheckError

from numpyro_forecast import Horizon, backtest, predict
from numpyro_forecast.features import fourier_features, periodic_repeat
from numpyro_forecast.typing import IntScalar, RealScalar


def _expected_fourier(duration: int, period: float, num_terms: int) -> np.ndarray:
    """Build the reference Fourier design matrix in float64 NumPy."""
    angles = 2.0 * np.pi * np.arange(1, num_terms + 1)[None, :] * np.arange(duration)[:, None]
    return np.concatenate([np.sin(angles / period), np.cos(angles / period)], axis=-1)


def test_fourier_features_shape_and_values() -> None:
    feats = fourier_features(duration=12, period=12.0, num_terms=3)
    assert feats.shape == (12, 6)
    assert jnp.allclose(feats[0, 0], 0.0, atol=1e-6)
    assert jnp.allclose(feats[3, 0], 1.0, atol=1e-6)


@pytest.mark.parametrize("duration", [10, np.int64(10), np.int32(10)], ids=repr)
@pytest.mark.parametrize(
    "period", [7, 7.0, np.int64(7), np.float64(7.0), np.float32(7.0)], ids=repr
)
@pytest.mark.parametrize("num_terms", [2, np.int64(2)], ids=repr)
def test_fourier_features_accepts_python_and_numpy_scalars(
    duration: IntScalar, period: RealScalar, num_terms: IntScalar
) -> None:
    reference = fourier_features(10, 7.0, 2)
    np.testing.assert_array_equal(fourier_features(duration, period, num_terms), reference)
    np.testing.assert_allclose(reference, _expected_fourier(10, 7.0, 2), atol=1e-6)


@pytest.mark.parametrize(
    ("duration", "period", "num_terms"),
    [
        pytest.param(10.0, 7.0, 2, id="float-duration"),
        pytest.param(np.float64(10.0), 7.0, 2, id="numpy-float-duration"),
        pytest.param(10, 7.0, 2.5, id="float-num-terms"),
        pytest.param(10, "7", 2, id="str-period"),
        pytest.param(jnp.asarray(10), 7.0, 2, id="jax-duration"),
        pytest.param(10, jnp.asarray(7.0), 2, id="jax-period"),
    ],
)
def test_fourier_features_rejects_non_integral_counts_and_non_scalars(
    duration: object, period: object, num_terms: object
) -> None:
    with pytest.raises(TypeCheckError):
        fourier_features(duration, period, num_terms)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        pytest.param(fourier_features, (10.7, 7.0, 2), id="fourier-duration"),
        pytest.param(fourier_features, (10, 7.0, 2.0), id="fourier-num-terms"),
        pytest.param(periodic_repeat, (jnp.arange(3.0), 7.5), id="periodic-duration"),
    ],
)
def test_float_counts_raise_with_the_runtime_checks_off(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Array], args: tuple[object, ...]
) -> None:
    monkeypatch.setattr(jaxtyping.config, "jaxtyping_disable", True)
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        fn(*args)


def test_fourier_features_rejects_a_traced_duration() -> None:
    with pytest.raises(TypeCheckError):
        jax.jit(lambda d: fourier_features(d, 7.0, 2))(10)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        pytest.param((-3, 7.0, 2), "duration must be non-negative", id="negative-duration"),
        pytest.param((10, 0, 2), "period must be positive and finite", id="zero-period"),
        pytest.param((10, -7.0, 2), "period must be positive and finite", id="negative-period"),
        pytest.param((10, float("nan"), 2), "period must be positive and finite", id="nan-period"),
        pytest.param((10, float("inf"), 2), "period must be positive and finite", id="inf-period"),
        pytest.param((10, np.float32("nan"), 2), "period must be positive", id="numpy-nan-period"),
        pytest.param((10, 7.0, -1), "num_terms must be non-negative", id="negative-num-terms"),
        pytest.param((10, 7.0, np.int64(-1)), "num_terms must be non-negative", id="numpy-terms"),
    ],
)
def test_fourier_features_rejects_degenerate_values(
    args: tuple[IntScalar, RealScalar, IntScalar], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        fourier_features(*args)


@pytest.mark.parametrize(
    ("duration", "num_terms", "shape"), [(0, 1, (0, 2)), (10, 0, (10, 0))], ids=repr
)
def test_fourier_features_accepts_an_empty_window_or_no_harmonics(
    duration: int, num_terms: int, shape: tuple[int, int]
) -> None:
    assert fourier_features(duration, 7.0, num_terms).shape == shape


def test_periodic_repeat_rejects_a_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration must be non-negative"):
        periodic_repeat(jnp.array([1.0, 2.0, 3.0]), np.int64(-1))


def test_periodic_repeat_accepts_a_zero_duration() -> None:
    assert periodic_repeat(jnp.array([1.0, 2.0, 3.0]), 0).shape == (0,)


def test_fourier_features_with_numpy_scalars_under_jit_and_vmap() -> None:
    coefs = jnp.arange(12.0).reshape(3, 4)

    @jax.jit
    @jax.vmap
    def seasonal(beta: Array) -> Array:
        return fourier_features(np.int64(29), np.int64(9), np.int64(2)) @ beta

    expected = (_expected_fourier(29, 9.0, 2) @ np.asarray(coefs).T).T
    np.testing.assert_allclose(seasonal(coefs), expected, rtol=1e-5, atol=1e-5)


def test_fourier_features_first_called_inside_jit_stays_usable() -> None:
    duration, period, num_terms = 23, 6.5, 2
    expected = _expected_fourier(duration, period, num_terms)
    traced = jax.jit(lambda x: x + fourier_features(duration, period, num_terms))(jnp.zeros(()))
    np.testing.assert_allclose(traced, expected, atol=1e-5)
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
    season = jnp.arange(6.0).reshape(2, 3)
    repeated = periodic_repeat(season, 5, axis=-1)
    assert repeated.shape == (2, 5)
    assert jnp.allclose(repeated[0], jnp.array([0.0, 1.0, 2.0, 0.0, 1.0]))


def test_periodic_repeat_accepts_array_like() -> None:
    repeated = periodic_repeat(np.array([1.0, 2.0, 3.0]), 7)
    assert repeated.shape == (7,)
    assert jnp.allclose(repeated, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))


@pytest.mark.parametrize("duration", [7, np.int64(7), np.uint8(7)], ids=repr)
def test_periodic_repeat_accepts_python_and_numpy_integers(duration: IntScalar) -> None:
    repeated = periodic_repeat(jnp.array([1.0, 2.0, 3.0]), duration)
    np.testing.assert_array_equal(repeated, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))


@pytest.mark.parametrize("duration", [7.0, np.float64(7.0), jnp.asarray(7)], ids=repr)
def test_periodic_repeat_rejects_a_non_integral_or_array_duration(duration: object) -> None:
    with pytest.raises(TypeCheckError):
        periodic_repeat(jnp.array([1.0, 2.0, 3.0]), duration)  # ty: ignore[invalid-argument-type]


def test_periodic_repeat_with_a_numpy_duration_under_jit_and_vmap() -> None:
    seasons = jnp.arange(6.0).reshape(2, 3)
    repeated = jax.jit(jax.vmap(lambda season: periodic_repeat(season, np.int64(5))))(seasons)
    np.testing.assert_array_equal(repeated, periodic_repeat(seasons, 5, axis=-1))
