"""Tests for the ArviZ DataTree export (roadmap §5)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import RandomWalkModel, empty_covariates
from jax import Array, random

from numpyro_forecast.convert import add_forecast, to_datatree, to_inferencedata
from numpyro_forecast.functional import draw_posterior, fit_mcmc, fit_svi, forecast

arviz_base = pytest.importorskip("arviz_base")
arviz_stats = pytest.importorskip("arviz_stats")


def _series(n: int = 18) -> Array:
    return jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (n, 1)), axis=-2)


def _mcmc_fit(num_chains: int = 2):  # type: ignore[no-untyped-def]
    data = _series()
    covariates = empty_covariates(data.shape[-2])
    fit = fit_mcmc(
        random.PRNGKey(1),
        RandomWalkModel(),
        data,
        covariates,
        num_warmup=50,
        num_samples=50,
        num_chains=num_chains,
    )
    return fit, data, covariates


def test_to_datatree_mcmc_groups_and_dims() -> None:
    fit, data, covariates = _mcmc_fit()
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    assert set(tree.children) == {
        "posterior",
        "posterior_predictive",
        "observed_data",
        "constant_data",
    }
    post = tree["posterior"]
    assert post.sizes["chain"] == 2
    assert post.sizes["draw"] == 50
    pp = tree["posterior_predictive"]
    assert pp.sizes["time"] == data.shape[-2]
    assert pp.sizes["obs_dim"] == 1
    assert tree.attrs["inference_library"] == "numpyro"
    assert tree.attrs["creation_library"] == "numpyro_forecast"


def test_to_datatree_observed_and_constant_roundtrip() -> None:
    fit, data, covariates = _mcmc_fit()
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    obs = np.asarray(tree["observed_data"]["obs"])
    np.testing.assert_allclose(obs, np.asarray(data), rtol=1e-6)
    const = np.asarray(tree["constant_data"]["covariates"])
    assert const.shape == (data.shape[-2], 0)


def test_mcmc_chain_reshape_roundtrip() -> None:
    """The (num_samples,...) -> (chain, draw, ...) reshape matches group_by_chain."""
    fit, data, covariates = _mcmc_fit(num_chains=2)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    reshaped = np.asarray(tree["posterior"]["sigma"])  # (chain, draw)
    flat = np.asarray(fit.samples["sigma"])  # (num_samples,)
    # get_samples(group_by_chain=False) concatenates chains in order, so a plain
    # reshape recovers the per-chain layout.
    np.testing.assert_allclose(reshaped.reshape(-1), flat, rtol=1e-6)
    assert reshaped.shape == (2, flat.shape[0] // 2)


def test_rhat_runs_warning_free_on_two_chain_posterior() -> None:
    """Acceptance: az rhat runs warning-free on a 2-chain posterior."""
    fit, data, covariates = _mcmc_fit(num_chains=2)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rhat = arviz_stats.rhat(tree["posterior"].dataset)
    assert "sigma" in rhat


def test_to_datatree_svi_has_variational_metadata() -> None:
    data = _series()
    covariates = empty_covariates(data.shape[-2])
    svi = fit_svi(random.PRNGKey(1), RandomWalkModel(), data, covariates, num_steps=60)
    tree = to_datatree(
        random.PRNGKey(2),
        svi,
        RandomWalkModel(),
        data,
        covariates,
        num_predictive_samples=40,
    )
    assert tree["posterior"].sizes["chain"] == 1
    assert tree["posterior"].sizes["draw"] == 40
    assert tree["posterior"].dataset.attrs.get("variational") is True


def test_to_datatree_time_coord_threading() -> None:
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    custom_time = list(range(100, 100 + n))
    tree = to_datatree(
        random.PRNGKey(2), fit, RandomWalkModel(), data, covariates, time_coord=custom_time
    )
    np.testing.assert_array_equal(tree["observed_data"].coords["time"].values, custom_time)
    np.testing.assert_array_equal(tree["posterior_predictive"].coords["time"].values, custom_time)


def test_add_forecast_groups_and_time_continuation() -> None:
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 5)
    post = draw_posterior(random.PRNGKey(3), fit, 100)
    fc = forecast(random.PRNGKey(4), RandomWalkModel(), post, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    out = add_forecast(tree, fc, future_covariates[n:])
    assert "predictions" in out.children
    assert "predictions_constant_data" in out.children
    times = out["predictions"].coords["time"].values
    np.testing.assert_array_equal(times, np.arange(n, n + 5))
    # The original tree is untouched.
    assert "predictions" not in tree.children


def test_add_forecast_explicit_time_coord() -> None:
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 3)
    post = draw_posterior(random.PRNGKey(3), fit, 60)
    fc = forecast(random.PRNGKey(4), RandomWalkModel(), post, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    out = add_forecast(tree, fc, future_covariates[n:], time_coord=[900, 901, 902])
    np.testing.assert_array_equal(out["predictions"].coords["time"].values, [900, 901, 902])


def test_all_groups_via_dict_to_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every group must be built through arviz_base.dict_to_dataset (normative rule)."""
    calls = {"count": 0}
    real = arviz_base.dict_to_dataset

    def spy(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(arviz_base, "dict_to_dataset", spy)
    fit, data, covariates = _mcmc_fit()
    to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    # posterior + posterior_predictive + observed_data + constant_data.
    assert calls["count"] == 4


def test_to_inferencedata_shim_raises_on_datatree_only_arviz() -> None:
    fit, data, covariates = _mcmc_fit()
    with pytest.warns(FutureWarning), pytest.raises(ImportError, match="to_datatree"):
        to_inferencedata(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
