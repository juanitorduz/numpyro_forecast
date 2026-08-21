"""Tests for guide resolution and guide-flavored posterior draws (roadmap §2).

This file is deleted alongside ``fit_svi`` in a later task, so ``RandomWalkModel``
(formerly shared via ``conftest.py``, which is now functional-only) is kept here
as a local, file-scoped copy rather than threaded through the shared fixture file.
"""

import functools

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import as_autoguide, empty_covariates
from jax import Array, random
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoGuide,
    AutoMultivariateNormal,
    AutoNormal,
)

from numpyro_forecast.exceptions import GuideResolutionError
from numpyro_forecast.forecaster import ForecastingModel
from numpyro_forecast.functional import (
    draw_posterior,
    fit_svi,
    forecast,
    resolve_guide,
)


class RandomWalkModel(ForecastingModel):
    """Local-level random walk with Normal observation noise (local test copy)."""

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series("drift", lambda: dist.Normal(0.0, drift_scale))
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


def _handwritten_guide(covariates, data=None) -> None:
    """A mean-field hand-written guide for RandomWalkModel (in-sample only)."""
    t = covariates.shape[-2]
    drift_scale_loc = numpyro.param("drift_scale_loc", -1.0)
    sigma_loc = numpyro.param("sigma_loc", -1.0)
    numpyro.sample("drift_scale", dist.LogNormal(drift_scale_loc, 0.1))
    numpyro.sample("sigma", dist.LogNormal(sigma_loc, 0.1))
    drift_loc = numpyro.param("drift_loc", jnp.zeros((t, 1)))
    with numpyro.plate("time", t, dim=-2):
        numpyro.sample("drift", dist.Normal(drift_loc, 0.1))


def test_resolve_none_returns_autonormal() -> None:
    guide = resolve_guide(None, RandomWalkModel())
    assert isinstance(guide, AutoNormal)


def test_resolve_instance_is_identity() -> None:
    model = RandomWalkModel()
    instance = AutoNormal(model)
    assert resolve_guide(instance, model) is instance


def test_resolve_subclass_is_called() -> None:
    guide = resolve_guide(AutoMultivariateNormal, RandomWalkModel())
    assert isinstance(guide, AutoMultivariateNormal)


def test_resolve_partial_factory_is_called() -> None:
    factory = functools.partial(AutoNormal, init_scale=0.05)
    guide = resolve_guide(factory, RandomWalkModel())
    assert isinstance(guide, AutoNormal)


def test_resolve_handwritten_is_identity() -> None:
    guide = resolve_guide(_handwritten_guide, RandomWalkModel())
    assert guide is _handwritten_guide


def test_resolve_lambda_factory_rejected() -> None:
    # A factory-shaped lambda `model -> guide` is ambiguous and rejected.
    with pytest.raises(GuideResolutionError, match=r"ambiguous|factory"):
        resolve_guide(lambda model: AutoNormal(model), RandomWalkModel())


def test_resolve_unknown_type_rejected() -> None:
    with pytest.raises(GuideResolutionError, match="does not support"):
        resolve_guide(42, RandomWalkModel())  # ty: ignore[invalid-argument-type]


def test_autodelta_draw_has_tiled_sample_axis() -> None:
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(
        random.PRNGKey(0), RandomWalkModel(), data, covariates, guide=AutoDelta, num_steps=10
    )
    samples = draw_posterior(random.PRNGKey(1), as_autoguide(fit.guide), fit.params, 7)
    # AutoDelta is a point estimate tiled to a leading sample axis: the latent
    # values are identical across the axis (dispatch by type, not by shape).
    assert samples["sigma"].shape[0] == 7
    assert jnp.allclose(samples["sigma"][0], samples["sigma"][1])


def test_handwritten_guide_end_to_end() -> None:
    # draw_posterior is guide-based only (AutoGuide); a hand-written guide draws
    # with one direct Predictive call instead (the guide is not vmappable/jittable
    # through the shared jitted-sample-posterior path).
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(
        random.PRNGKey(0),
        RandomWalkModel(),
        data,
        covariates,
        guide=_handwritten_guide,
        num_steps=10,
    )
    predictive = Predictive(fit.guide, params=fit.params, num_samples=8)
    samples = predictive(random.PRNGKey(1), covariates, data)
    assert samples["drift"].shape == (8, 20, 1)
    forecast_covariates = empty_covariates(25)
    fc = forecast(random.PRNGKey(2), RandomWalkModel(), samples, data, forecast_covariates)
    assert fc.shape[0] == 8
    assert jnp.all(jnp.isfinite(fc))


@pytest.mark.parametrize(
    ("guide", "flavor"),
    [
        (None, "AutoNormal"),
        (AutoMultivariateNormal, "AutoMVN"),
        (AutoDelta, "AutoDelta"),
        (_handwritten_guide, "handwritten"),
    ],
)
def test_guide_never_sees_future_sites(guide, flavor) -> None:
    """Invariant I1: no resolved guide contains a ``*_future`` sample site."""
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(random.PRNGKey(0), RandomWalkModel(), data, covariates, guide=guide, num_steps=5)
    resolved = fit.guide
    if isinstance(resolved, AutoGuide):
        tr = trace(seed(resolved, random.PRNGKey(1))).get_trace(covariates, data)
    else:
        tr = trace(seed(resolved, random.PRNGKey(1))).get_trace(covariates, data)
    future_sites = [
        name for name, site in tr.items() if name.endswith("_future") and site["type"] == "sample"
    ]
    assert not future_sites, f"{flavor} guide leaked future sites: {future_sites}"


def test_autonormal_forecast_shape_contract() -> None:
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(random.PRNGKey(0), RandomWalkModel(), data, covariates, num_steps=10)
    samples = draw_posterior(random.PRNGKey(1), as_autoguide(fit.guide), fit.params, 16)
    fc = forecast(random.PRNGKey(2), RandomWalkModel(), samples, data, empty_covariates(25))
    assert fc.shape == (16, 5, 1)


def test_bare_automvn_class_fits_and_forecasts() -> None:
    """Acceptance: bare AutoMultivariateNormal class fits and forecasts."""
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(
        random.PRNGKey(0),
        RandomWalkModel(),
        data,
        covariates,
        guide=AutoMultivariateNormal,
        num_steps=10,
    )
    samples = draw_posterior(random.PRNGKey(1), as_autoguide(fit.guide), fit.params, 12)
    fc = forecast(random.PRNGKey(2), RandomWalkModel(), samples, data, empty_covariates(24))
    assert fc.shape == (12, 4, 1)
    assert jnp.all(jnp.isfinite(fc))


def test_autodelta_forecast_varies_over_tiled_latents() -> None:
    """Acceptance: AutoDelta forecast has a sample axis with varying values."""
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(
        random.PRNGKey(0), RandomWalkModel(), data, covariates, guide=AutoDelta, num_steps=10
    )
    samples = draw_posterior(random.PRNGKey(1), as_autoguide(fit.guide), fit.params, 32)
    fc = forecast(random.PRNGKey(2), RandomWalkModel(), samples, data, empty_covariates(25))
    assert fc.shape == (32, 5, 1)
    # Observation noise makes forecasts vary across the sample axis even though
    # the latents were tiled point estimates.
    assert float(fc.std(axis=0).mean()) > 0.0
