"""Tests for kernel resolution, run-config validation, and fit_mcmc (roadmap §3)."""

import jax.numpy as jnp
import pytest
from conftest import RandomWalkModel, empty_covariates
from jax import random
from numpyro.infer import AIES, ESS, HMC, HMCECS, NUTS, SA, BarkerMH, HMCGibbs

from numpyro_forecast.functional import (
    MCMCFit,
    draw_posterior,
    fit_mcmc,
    forecast,
    resolve_kernel,
)


def test_resolve_none_returns_nuts() -> None:
    kernel = resolve_kernel(None, RandomWalkModel(), None)
    assert isinstance(kernel, NUTS)


def test_resolve_none_forwards_kwargs() -> None:
    kernel = resolve_kernel(None, RandomWalkModel(), {"target_accept_prob": 0.9})
    assert isinstance(kernel, NUTS)
    assert kernel._target_accept_prob == 0.9


def test_resolve_class_is_called() -> None:
    kernel = resolve_kernel(HMC, RandomWalkModel(), None)
    assert isinstance(kernel, HMC)


def test_resolve_instance_is_identity() -> None:
    instance = NUTS(RandomWalkModel())
    assert resolve_kernel(instance, RandomWalkModel(), None) is instance


def test_resolve_instance_with_kwargs_raises() -> None:
    instance = NUTS(RandomWalkModel())
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_kernel(instance, RandomWalkModel(), {"target_accept_prob": 0.9})


def test_resolve_unknown_type_raises() -> None:
    with pytest.raises(TypeError, match="does not support"):
        resolve_kernel(42, RandomWalkModel(), None)  # ty: ignore[invalid-argument-type]


def test_aies_single_chain_rejected() -> None:
    data = jnp.zeros((15, 1))
    covariates = empty_covariates(15)
    with pytest.raises(ValueError, match="ensemble sampler"):
        fit_mcmc(
            random.PRNGKey(0),
            RandomWalkModel(),
            data,
            covariates,
            kernel=AIES,
            num_warmup=5,
            num_samples=5,
            num_chains=1,
            chain_method="vectorized",
        )


def test_aies_non_vectorized_rejected() -> None:
    data = jnp.zeros((15, 1))
    covariates = empty_covariates(15)
    with pytest.raises(ValueError, match="ensemble sampler"):
        fit_mcmc(
            random.PRNGKey(0),
            RandomWalkModel(),
            data,
            covariates,
            kernel=ESS,
            num_warmup=5,
            num_samples=5,
            num_chains=4,
            chain_method="sequential",
        )


@pytest.mark.parametrize("kernel", [NUTS, HMC, BarkerMH, SA])
def test_numpyro_kernels_fit_and_forecast(kernel) -> None:
    data = jnp.zeros((15, 1))
    covariates = empty_covariates(15)
    fit = fit_mcmc(
        random.PRNGKey(0),
        RandomWalkModel(),
        data,
        covariates,
        kernel=kernel,
        num_warmup=10,
        num_samples=10,
    )
    assert isinstance(fit, MCMCFit)
    assert fit.num_chains == 1
    samples = draw_posterior(random.PRNGKey(1), fit, 10)
    fc = forecast(random.PRNGKey(2), RandomWalkModel(), samples, data, empty_covariates(18))
    assert fc.shape == (10, 3, 1)
    assert jnp.all(jnp.isfinite(fc))


def test_aies_vectorized_smoke() -> None:
    data = jnp.zeros((12, 1))
    covariates = empty_covariates(12)
    fit = fit_mcmc(
        random.PRNGKey(0),
        RandomWalkModel(),
        data,
        covariates,
        kernel=AIES,
        num_warmup=10,
        num_samples=10,
        num_chains=4,
        chain_method="vectorized",
    )
    assert fit.num_chains == 4
    assert fit.samples["sigma"].shape[0] == 40


def test_mcmcfit_stores_num_chains() -> None:
    data = jnp.zeros((12, 1))
    covariates = empty_covariates(12)
    fit = fit_mcmc(
        random.PRNGKey(0),
        RandomWalkModel(),
        data,
        covariates,
        num_warmup=5,
        num_samples=6,
        num_chains=2,
        chain_method="vectorized",
    )
    assert fit.num_chains == 2
    # Samples are stored flattened (group_by_chain=False): 2 chains * 6 draws.
    assert fit.samples["sigma"].shape[0] == 12


def test_hmcgibbs_instance_smoke() -> None:
    model = RandomWalkModel()

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        return {"sigma": hmc_sites.get("sigma", jnp.array(0.1))}

    inner = NUTS(model)
    kernel = HMCGibbs(inner, gibbs_fn=gibbs_fn, gibbs_sites=["sigma"])
    data = jnp.zeros((12, 1))
    covariates = empty_covariates(12)
    fit = fit_mcmc(
        random.PRNGKey(0),
        model,
        data,
        covariates,
        kernel=kernel,
        num_warmup=8,
        num_samples=8,
    )
    assert isinstance(fit, MCMCFit)


def test_hmcecs_instance_resolves() -> None:
    # HMCECS structurally needs a subsampling model, so this smoke exercises the
    # instance path of resolve_kernel rather than a full fit.
    model = RandomWalkModel()
    inner = NUTS(model)
    kernel = HMCECS(inner, num_blocks=2)
    assert resolve_kernel(kernel, model, None) is kernel
