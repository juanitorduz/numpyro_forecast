"""Tests for functional SVI fitting (``functional.svi``)."""

import jax.numpy as jnp
import pytest
from conftest import empty_covariates, rw_body, svi_fit
from jax import random

from numpyro_forecast.functional import SVIFit, fit_svi, forecasting_model


def test_fit_svi_returns_populated_fit() -> None:
    fit = svi_fit(t=30, num_steps=40)
    assert isinstance(fit, SVIFit)
    assert fit.losses.shape == (40,)
    assert any("drift_scale" in name for name in fit.params)


def test_fit_svi_rejects_unequal_duration() -> None:
    model = forecasting_model(rw_body)
    data = jnp.zeros((30, 1))
    with pytest.raises(ValueError, match="equal duration"):
        fit_svi(random.PRNGKey(0), model, data, empty_covariates(25), num_steps=10)
