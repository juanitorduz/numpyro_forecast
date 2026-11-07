"""Tests for the time-axis array shaping helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from numpyro_forecast.arrays import concat_future, pad_future, zero_data_like
from numpyro_forecast.typing import IntScalar


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


def test_pad_future_accepts_an_int_fill_value() -> None:
    x = jnp.zeros((4, 1))
    np.testing.assert_array_equal(pad_future(x, 3, value=1), pad_future(x, 3, value=1.0))
    gate = jnp.zeros((2, 4, 3), dtype=bool)
    assert bool(jnp.all(pad_future(gate, 2, value=1)[:, 4:]))


@pytest.mark.parametrize(
    ("future", "value"),
    [(np.int64(3), 1.0), (np.uint8(3), np.float32(1.0)), (3, np.int64(1)), (np.int32(3), 1)],
    ids=["numpy-rows", "numpy-rows-and-float32-value", "numpy-int-value", "int32-rows-int-value"],
)
def test_pad_future_accepts_numpy_scalars(future: object, value: object) -> None:
    x = jnp.zeros((4, 1))
    np.testing.assert_array_equal(pad_future(x, future, value=value), pad_future(x, 3, value=1.0))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("future", [3.0, np.float64(3.0), jnp.asarray(3)], ids=repr)
def test_pad_future_rejects_a_float_or_array_row_count(future: object) -> None:
    with pytest.raises(TypeCheckError):
        pad_future(jnp.zeros((4, 1)), future)  # ty: ignore[invalid-argument-type]


def test_pad_future_keeps_integer_fills_exact() -> None:
    with jax.enable_x64(True):
        wide = pad_future(jnp.zeros((1, 1), dtype=jnp.int64), 1, value=2**53 + 1)
        assert int(wide[1, 0]) == 2**53 + 1
    unsigned = pad_future(jnp.zeros((1, 1), dtype=jnp.uint8), 1, value=255)
    assert int(unsigned[1, 0]) == 255


@pytest.mark.parametrize("future", [-1, np.int64(-1)], ids=repr)
def test_pad_future_rejects_a_negative_row_count(future: IntScalar) -> None:
    with pytest.raises(ValueError, match="future must be non-negative"):
        pad_future(jnp.zeros((4, 1)), future)
