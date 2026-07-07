"""Tests for the time-axis array shaping helpers."""

import jax.numpy as jnp

from numpyro_forecast.arrays import concat_future, zero_data_like


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
