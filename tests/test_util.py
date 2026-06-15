"""Tests for array/distribution/seasonality utilities."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from numpyro_forecast.util import (
    concat_future,
    fourier_features,
    periodic_repeat,
    prefix_condition,
    shift_loc,
    slice_time,
    zero_data_like,
)


def test_zero_data_like_extends_to_covariate_duration() -> None:
    data = jnp.ones((3, 10, 2))
    covariates = jnp.ones((3, 17, 5))
    zero = zero_data_like(data, covariates)
    assert zero.shape == (3, 17, 2)
    assert bool(jnp.all(zero == 0))


def test_concat_future_default_time_axis() -> None:
    prefix = jnp.ones((4, 2))
    suffix = jnp.zeros((3, 2))
    out = concat_future(prefix, suffix)
    assert out.shape == (7, 2)


def test_shift_loc_normal() -> None:
    noise = dist.Normal(loc=jnp.zeros((5, 1)), scale=2.0)
    prediction = jnp.arange(5.0)[:, None]
    shifted = shift_loc(noise, prediction)
    assert isinstance(shifted, dist.Normal)
    assert jnp.allclose(shifted.mean, prediction)
    assert jnp.allclose(jnp.broadcast_to(shifted.scale, (5, 1)), 2.0)


def test_shift_loc_student_t_keeps_df() -> None:
    noise = dist.StudentT(df=4.0, loc=jnp.zeros((3, 1)), scale=1.0)
    shifted = shift_loc(noise, jnp.ones((3, 1)))
    assert isinstance(shifted, dist.StudentT)
    assert jnp.allclose(shifted.df, 4.0)
    assert jnp.allclose(shifted.loc, 1.0)


def test_shift_loc_unsupported_raises() -> None:
    with pytest.raises(NotImplementedError, match="shift_loc"):
        shift_loc(dist.Poisson(rate=1.0), jnp.zeros(()))


def test_slice_time_prefix_and_suffix() -> None:
    loc = jnp.arange(6.0)[:, None]  # (6, 1)
    d = dist.Normal(loc=loc, scale=1.0)
    prefix = slice_time(d, slice(None, 4))
    suffix = slice_time(d, slice(4, None))
    assert isinstance(prefix, dist.Normal)
    assert isinstance(suffix, dist.Normal)
    assert prefix.batch_shape == (4, 1)
    assert suffix.batch_shape == (2, 1)
    assert jnp.allclose(prefix.loc, loc[:4])
    assert jnp.allclose(suffix.loc, loc[4:])


def test_prefix_condition_iid_returns_future_slice() -> None:
    loc = jnp.arange(6.0)[:, None]
    d = dist.Normal(loc=loc, scale=1.0)
    data = jnp.zeros((4, 1))  # t = 4
    future = prefix_condition(d, data)
    assert isinstance(future, dist.Normal)
    assert future.batch_shape == (2, 1)
    assert jnp.allclose(future.loc, loc[4:])


def test_fourier_features_shape_and_values() -> None:
    feats = fourier_features(duration=12, period=12.0, num_terms=3)
    assert feats.shape == (12, 6)
    # First column is sin(2*pi*1*t/12); at t=0 it is 0, at t=3 it is 1.
    assert jnp.allclose(feats[0, 0], 0.0, atol=1e-6)
    assert jnp.allclose(feats[3, 0], 1.0, atol=1e-6)


def test_periodic_repeat_tiles_pattern() -> None:
    season = jnp.array([1.0, 2.0, 3.0])
    repeated = periodic_repeat(season, 7)
    assert jnp.allclose(repeated, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))


def test_periodic_repeat_along_axis() -> None:
    season = jnp.arange(6.0).reshape(2, 3)  # period 3 along axis -1
    repeated = periodic_repeat(season, 5, axis=-1)
    assert repeated.shape == (2, 5)
    assert jnp.allclose(repeated[0], jnp.array([0.0, 1.0, 2.0, 0.0, 1.0]))
