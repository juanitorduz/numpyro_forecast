"""Tests for functional MCMC fitting (``functional.mcmc``)."""

import jax.numpy as jnp
import pytest
from conftest import empty_covariates, mcmc_fit, rw_body
from jax import random

from numpyro_forecast.functional import MCMCFit, fit_mcmc, forecasting_model


def test_fit_mcmc_returns_populated_fit() -> None:
    fit = mcmc_fit(t=20, num_samples=20)
    assert isinstance(fit, MCMCFit)
    assert "drift_scale" in fit.samples
    assert fit.samples["drift"].shape[0] == 20


def test_fit_mcmc_rejects_unequal_duration() -> None:
    model = forecasting_model(rw_body)
    data = jnp.zeros((20, 1))
    with pytest.raises(ValueError, match="equal duration"):
        fit_mcmc(
            random.PRNGKey(0),
            model,
            data,
            empty_covariates(15),
            num_warmup=5,
            num_samples=5,
        )
