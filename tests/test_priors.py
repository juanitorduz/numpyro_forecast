"""Tests for the shrinkage prior helpers in `numpyro_forecast.priors`."""

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jaxtyping import TypeCheckError

from numpyro_forecast.priors import minnesota_prior

P, K = 3, 2


def test_minnesota_prior_shapes() -> None:
    loc, scale = minnesota_prior(P, K, tightness=0.2)
    assert loc.shape == (P, K, K)
    assert scale.shape == (P, K, K)


def test_minnesota_prior_loc_is_own_lag_mean_on_the_first_lag_only() -> None:
    loc, _ = minnesota_prior(P, K, tightness=0.2, own_lag_mean=0.9)
    assert jnp.allclose(loc[0], 0.9 * jnp.eye(K))
    assert jnp.array_equal(loc[1:], jnp.zeros((P - 1, K, K)))


def test_minnesota_prior_default_loc_is_a_random_walk() -> None:
    loc, _ = minnesota_prior(P, K, tightness=0.2)
    assert jnp.array_equal(loc[0], jnp.eye(K))


def test_minnesota_prior_harmonic_decay_on_own_lags() -> None:
    _, scale = minnesota_prior(P, K, tightness=0.2, decay="harmonic")
    own = jnp.stack([jnp.diagonal(scale[lag]) for lag in range(P)])
    expected = 0.2 / jnp.arange(1, P + 1)
    assert jnp.allclose(own, jnp.broadcast_to(expected[:, None], own.shape))


def test_minnesota_prior_geometric_decay_on_own_lags() -> None:
    _, scale = minnesota_prior(P, K, tightness=0.2, decay="geometric")
    own = jnp.stack([jnp.diagonal(scale[lag]) for lag in range(P)])
    expected = 0.2 / jnp.arange(1, P + 1) ** 2
    assert jnp.allclose(own, jnp.broadcast_to(expected[:, None], own.shape))


def test_minnesota_prior_cross_lags_are_shrunk_by_cross_shrinkage() -> None:
    _, scale = minnesota_prior(P, K, tightness=0.2, cross_shrinkage=0.25)
    off = ~jnp.eye(K, dtype=bool)
    for lag in range(P):
        own = scale[lag, 0, 0]
        assert jnp.allclose(scale[lag][off], 0.25 * own)


def test_minnesota_prior_is_linear_in_tightness() -> None:
    _, scale_1 = minnesota_prior(P, K, tightness=0.1)
    _, scale_5 = minnesota_prior(P, K, tightness=0.5)
    assert jnp.allclose(scale_5, 5.0 * scale_1)


def test_minnesota_prior_accepts_a_traced_tightness() -> None:
    def scale_of(lam: jax.Array) -> jax.Array:
        return minnesota_prior(P, K, tightness=lam)[1]

    traced = jax.jit(scale_of)(jnp.asarray(0.3))
    eager = minnesota_prior(P, K, tightness=0.3)[1]
    assert jnp.allclose(traced, eager)


def test_minnesota_prior_accepts_integer_own_lag_mean() -> None:
    loc, _ = minnesota_prior(P, K, tightness=0.2, own_lag_mean=0)
    assert jnp.array_equal(loc, jnp.zeros((P, K, K)))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: minnesota_prior(0, K, tightness=0.2), "n_lags"),
        (lambda: minnesota_prior(P, 0, tightness=0.2), "n_obs"),
        (lambda: minnesota_prior(P, K, tightness=0.2, cross_shrinkage=1.5), "cross_shrinkage"),
        (lambda: minnesota_prior(P, K, tightness=0.0), "tightness"),
    ],
    ids=["n_lags", "n_obs", "cross_shrinkage", "tightness"],
)
def test_minnesota_prior_validation(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_minnesota_prior_plugs_into_a_normal_event() -> None:
    loc, scale = minnesota_prior(P, K, tightness=0.2, own_lag_mean=0.0)
    prior = dist.Normal(loc, scale).to_event(3)
    assert prior.event_shape == (P, K, K)
    assert prior.batch_shape == ()


def test_minnesota_prior_rejects_unknown_decay() -> None:
    with pytest.raises(TypeCheckError):
        minnesota_prior(P, K, tightness=0.2, decay=cast(Literal["harmonic"], "linear"))
