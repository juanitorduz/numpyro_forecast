"""Tests for the time-axis array shaping helpers."""

import jax.numpy as jnp

from numpyro_forecast.arrays import concat_future, pad_future, zero_data_like


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


def test_pad_future_appends_rows_with_value() -> None:
    gate = jnp.ones((2, 4, 3), dtype=bool)
    padded = pad_future(gate, 5)
    assert padded.shape == (2, 9, 3)
    assert padded.dtype == gate.dtype
    assert bool(jnp.all(padded[:, :4]))
    assert not bool(jnp.any(padded[:, 4:]))
    ones = pad_future(jnp.zeros((4, 1)), 2, value=1.0)
    assert bool(jnp.all(ones[4:] == 1.0))
    assert pad_future(gate, 0).shape == gate.shape
