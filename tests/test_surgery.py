"""Tests for the distribution-surgery dispatchers and the elementwise registry."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from numpyro_forecast import surgery
from numpyro_forecast.surgery import (
    _ELEMENTWISE_FAMILIES,
    prefix_condition,
    register_elementwise,
    shift_loc,
    slice_time,
)


@pytest.fixture
def restore_elementwise_registry():
    """Snapshot and restore the module-global elementwise registry sets.

    Tests that register throwaway families must not leak them into the
    process-wide registry (which other tests inspect).
    """
    families = set(surgery._ELEMENTWISE_FAMILIES)
    checked = set(surgery._ELEMENTWISE_CHECKED)
    try:
        yield
    finally:
        surgery._ELEMENTWISE_FAMILIES.clear()
        surgery._ELEMENTWISE_FAMILIES.update(families)
        surgery._ELEMENTWISE_CHECKED.clear()
        surgery._ELEMENTWISE_CHECKED.update(checked)


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


def test_shift_loc_independent_shifts_base() -> None:
    base = dist.Normal(loc=jnp.zeros((4, 2)), scale=1.0)
    noise = dist.Independent(base, reinterpreted_batch_ndims=1)
    prediction = jnp.arange(8.0).reshape(4, 2)
    shifted = shift_loc(noise, prediction)
    assert isinstance(shifted, dist.Independent)
    assert shifted.reinterpreted_batch_ndims == 1
    assert isinstance(shifted.base_dist, dist.Normal)
    assert jnp.allclose(shifted.base_dist.loc, prediction)


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


def test_slice_time_independent_slices_base() -> None:
    loc = jnp.arange(6.0)[:, None]  # (6, 1)
    noise = dist.Independent(dist.Normal(loc=loc, scale=1.0), reinterpreted_batch_ndims=1)
    sliced = slice_time(noise, slice(None, 4))
    assert isinstance(sliced, dist.Independent)
    assert sliced.reinterpreted_batch_ndims == 1
    assert isinstance(sliced.base_dist, dist.Normal)
    assert sliced.base_dist.batch_shape == (4, 1)
    assert jnp.allclose(sliced.base_dist.loc, loc[:4])


def test_slice_time_mvn_batched_layout_supported() -> None:
    """A ``(*batch, time)`` MVN (obs == 1) slices along the time (event) axis.

    Regression for W1.2: a batched loc such as ``(5, time)`` used to be rejected;
    the time step count is now read from the covariance, so batched surgery works.
    """
    mvn = dist.MultivariateNormal(
        loc=jnp.zeros((5, 4)), covariance_matrix=jnp.broadcast_to(jnp.eye(4), (5, 4, 4))
    )
    sliced = slice_time(mvn, slice(None, 2))
    assert isinstance(sliced, dist.MultivariateNormal)
    assert sliced.loc.shape == (5, 2)


def test_prefix_condition_iid_returns_future_slice() -> None:
    loc = jnp.arange(6.0)[:, None]
    d = dist.Normal(loc=loc, scale=1.0)
    data = jnp.zeros((4, 1))  # t = 4
    future = prefix_condition(d, data)
    assert isinstance(future, dist.Normal)
    assert future.batch_shape == (2, 1)
    assert jnp.allclose(future.loc, loc[4:])


# --- P6: elementwise registry ------------------------------------------------

_SHIFT_LOC_FAMILIES = [
    (dist.Normal, {"loc": jnp.zeros((5, 1)), "scale": jnp.ones((5, 1))}),
    (
        dist.StudentT,
        {"df": jnp.full((5, 1), 4.0), "loc": jnp.zeros((5, 1)), "scale": jnp.ones((5, 1))},
    ),
    (dist.Laplace, {"loc": jnp.zeros((5, 1)), "scale": jnp.ones((5, 1))}),
    (dist.Cauchy, {"loc": jnp.zeros((5, 1)), "scale": jnp.ones((5, 1))}),
    (dist.Gumbel, {"loc": jnp.zeros((5, 1)), "scale": jnp.ones((5, 1))}),
    (
        dist.AsymmetricLaplace,
        {"loc": jnp.zeros((5, 1)), "scale": jnp.ones((5, 1)), "asymmetry": jnp.ones((5, 1))},
    ),
]


@pytest.mark.parametrize("cls, kwargs", _SHIFT_LOC_FAMILIES)
def test_shift_loc_adds_location(cls, kwargs) -> None:
    prediction = jnp.arange(5.0)[:, None]
    shifted = shift_loc(cls(**kwargs), prediction)
    assert isinstance(shifted, cls)
    assert jnp.allclose(shifted.loc, prediction)


def test_shift_loc_independent_laplace() -> None:
    base = dist.Laplace(loc=jnp.zeros((4, 2)), scale=1.0)
    noise = dist.Independent(base, reinterpreted_batch_ndims=1)
    prediction = jnp.arange(8.0).reshape(4, 2)
    shifted = shift_loc(noise, prediction)
    assert isinstance(shifted, dist.Independent)
    assert isinstance(shifted.base_dist, dist.Laplace)
    assert jnp.allclose(shifted.base_dist.loc, prediction)


_ELEMENTWISE_INSTANCES = [
    dist.Normal(loc=jnp.arange(6.0)[:, None], scale=1.0),
    dist.StudentT(df=4.0, loc=jnp.arange(6.0)[:, None], scale=1.0),
    dist.Laplace(loc=jnp.arange(6.0)[:, None], scale=1.0),
    dist.Cauchy(loc=jnp.arange(6.0)[:, None], scale=1.0),
    dist.Gumbel(loc=jnp.arange(6.0)[:, None], scale=1.0),
    dist.Poisson(rate=jnp.arange(1.0, 7.0)[:, None]),
    dist.NegativeBinomial2(mean=jnp.arange(1.0, 7.0)[:, None], concentration=2.0),
]


@pytest.mark.parametrize("noise", _ELEMENTWISE_INSTANCES)
def test_slice_time_matches_manual_param_slice(noise) -> None:
    sliced = slice_time(noise, slice(2, 5))
    assert type(sliced) is type(noise)
    assert sliced.batch_shape == (3, 1)
    for name in type(noise).arg_constraints:
        expected = jnp.broadcast_to(getattr(noise, name), (6, 1))[2:5, :]
        assert jnp.allclose(getattr(sliced, name), expected)


def test_poisson_prefix_condition_is_future_marginal() -> None:
    rate = jnp.arange(1.0, 7.0)[:, None]
    d = dist.Poisson(rate=rate)
    data = jnp.zeros((4, 1))
    future = prefix_condition(d, data)
    assert isinstance(future, dist.Poisson)
    assert future.batch_shape == (2, 1)
    assert jnp.allclose(future.rate, rate[4:])


def test_slice_time_dirichlet_actionable_error() -> None:
    # Dirichlet has a non-empty event_shape and is not elementwise.
    d = dist.Dirichlet(concentration=jnp.ones((5, 3)))
    with pytest.raises(NotImplementedError, match="register_elementwise"):
        slice_time(d, slice(None, 3))


def test_register_elementwise_decorator_and_slice(restore_elementwise_registry) -> None:
    @register_elementwise
    class _MyExponential(dist.Exponential):
        pass

    assert _MyExponential in _ELEMENTWISE_FAMILIES
    d = _MyExponential(rate=jnp.arange(1.0, 7.0)[:, None])
    sliced = slice_time(d, slice(1, 4))
    assert isinstance(sliced, _MyExponential)
    assert sliced.batch_shape == (3, 1)


def test_elementwise_membership_is_not_inherited() -> None:
    # A subclass of a registered family is not itself registered (exact type).
    class _SubNormal(dist.Normal):
        pass

    assert _SubNormal not in _ELEMENTWISE_FAMILIES
    with pytest.raises(NotImplementedError, match="register_elementwise"):
        slice_time(_SubNormal(loc=jnp.zeros((5, 1)), scale=1.0), slice(None, 3))


def test_register_structural_fake_fails_check(restore_elementwise_registry) -> None:
    """The generic elementwise path rejects a mis-declared family.

    ``_VectorFamily`` is a *direct* ``Distribution`` subclass, so ``slice_time``
    dispatches to the generic elementwise handler (not a dedicated one). Its
    non-empty ``event_shape`` makes ``_check_elementwise`` reject it rather than
    silently produce a wrong slice.
    """

    @register_elementwise
    class _VectorFamily(dist.Distribution):
        def __init__(self, loc: jnp.ndarray) -> None:
            self.loc = loc
            super().__init__(batch_shape=jnp.shape(loc)[:-1], event_shape=jnp.shape(loc)[-1:])

    d = _VectorFamily(jnp.zeros((5, 2)))  # event_shape == (2,)
    with pytest.raises(NotImplementedError, match="event_shape"):
        slice_time(d, slice(None, 3))


def test_mvn_subclass_dispatches_to_mvn_handler() -> None:
    """A ``MultivariateNormal`` subclass routes to the dedicated MVN handler (MRO).

    ``singledispatch`` picks the ``MultivariateNormal`` ``slice_time`` handler for
    any subclass, regardless of the elementwise registry, so the time (event) axis
    is sliced correctly instead of falling to the generic elementwise path.
    """

    class _MyMVN(dist.MultivariateNormal):
        pass

    d = _MyMVN(
        loc=jnp.zeros((5, 3)),  # batch=5, time(event)=3, obs=1
        covariance_matrix=jnp.broadcast_to(jnp.eye(3), (5, 3, 3)),
    )
    sliced = slice_time(d, slice(None, 2))
    assert isinstance(sliced, dist.MultivariateNormal)
    assert sliced.loc.shape == (5, 2)
