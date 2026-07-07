"""Registry-consistency invariants for the distribution-surgery dispatchers (I4)."""

import jax.numpy as jnp
import numpyro.distributions as dist

from numpyro_forecast.surgery import (
    _ELEMENTWISE_FAMILIES,
    prefix_condition,
    shift_loc,
    slice_time,
)

# One representative elementwise instance per registered family, for validation.
_REPRESENTATIVES: dict[type, dist.Distribution] = {
    dist.Normal: dist.Normal(loc=jnp.zeros((4, 1)), scale=1.0),
    dist.StudentT: dist.StudentT(df=4.0, loc=jnp.zeros((4, 1)), scale=1.0),
    dist.Laplace: dist.Laplace(loc=jnp.zeros((4, 1)), scale=1.0),
    dist.Cauchy: dist.Cauchy(loc=jnp.zeros((4, 1)), scale=1.0),
    dist.Gumbel: dist.Gumbel(loc=jnp.zeros((4, 1)), scale=1.0),
    dist.AsymmetricLaplace: dist.AsymmetricLaplace(
        loc=jnp.zeros((4, 1)), scale=1.0, asymmetry=1.0
    ),
    dist.Poisson: dist.Poisson(rate=jnp.ones((4, 1))),
    dist.NegativeBinomial2: dist.NegativeBinomial2(mean=jnp.ones((4, 1)), concentration=2.0),
}


def test_registry_consistency() -> None:
    """Invariant I4: slice_time/prefix_condition dispatch sets agree; correlated
    families register all of shift_loc/slice_time/prefix_condition.
    """
    slice_types = set(slice_time.registry)
    prefix_types = set(prefix_condition.registry)
    # Elementwise families flow through the default; the only explicit dispatch
    # shared by both slicers is Independent (plus the object default).
    assert slice_types == prefix_types, (
        f"slice_time and prefix_condition disagree on dispatched types: "
        f"{slice_types ^ prefix_types}"
    )

    # Correlated families (any explicit slice/prefix dispatch beyond the
    # structural Independent/object) must register all three surgeries so a
    # model can center, split, and condition them.
    shift_types = set(shift_loc.registry)
    structural = {object, dist.Independent}
    for cls in (slice_types | prefix_types) - structural:
        assert cls in shift_types and cls in slice_types and cls in prefix_types, (
            f"{cls.__name__} must register shift_loc, slice_time, and "
            "prefix_condition together (all-three-or-none rule)."
        )


def test_every_elementwise_family_slices_and_conditions() -> None:
    for cls in _ELEMENTWISE_FAMILIES:
        assert cls in _REPRESENTATIVES, f"add a representative instance for {cls.__name__}"
        d = _REPRESENTATIVES[cls]
        sliced = slice_time(d, slice(1, 3))
        assert type(sliced) is cls
        future = prefix_condition(d, jnp.zeros((2, 1)))
        assert type(future) is cls
        assert future.batch_shape == (2, 1)
