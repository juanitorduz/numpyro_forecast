"""Tests for the ArviZ DataTree export (roadmap §5)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import RandomWalkModel, empty_covariates
from jax import Array, random

from numpyro_forecast.convert import add_forecast_groups, to_datatree
from numpyro_forecast.functional import draw_posterior, fit_mcmc, fit_svi, forecast
from numpyro_forecast.typing import ForecastModel

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


def test_add_forecast_groups_adds_groups_and_time_continuation() -> None:
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 5)
    post = draw_posterior(random.PRNGKey(3), fit, 100)
    fc = forecast(random.PRNGKey(4), RandomWalkModel(), post, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    out = add_forecast_groups(tree, fc, future_covariates[n:])
    assert "predictions" in out.children
    assert "predictions_constant_data" in out.children
    times = out["predictions"].coords["time"].values
    np.testing.assert_array_equal(times, np.arange(n, n + 5))
    # The original tree is untouched.
    assert "predictions" not in tree.children


def test_add_forecast_groups_explicit_time_coord() -> None:
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 3)
    post = draw_posterior(random.PRNGKey(3), fit, 60)
    fc = forecast(random.PRNGKey(4), RandomWalkModel(), post, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    out = add_forecast_groups(tree, fc, future_covariates[n:], time_coord=[900, 901, 902])
    np.testing.assert_array_equal(out["predictions"].coords["time"].values, [900, 901, 902])


def test_add_forecast_groups_rejects_wrong_length_time_coord() -> None:
    """A ``time_coord`` whose length differs from the forecast horizon raises by name.

    Without the guard the mismatch flowed into ``dict_to_dataset`` and died deep
    in xarray with a shape message that never mentioned ``time_coord``.
    """
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 3)
    post = draw_posterior(random.PRNGKey(3), fit, 60)
    fc = forecast(random.PRNGKey(4), RandomWalkModel(), post, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    with pytest.raises(ValueError, match="time_coord has length 2"):
        add_forecast_groups(tree, fc, future_covariates[n:], time_coord=[900, 901])


def test_add_forecast_groups_noninteger_time_requires_explicit_coord() -> None:
    """A datetime64 in-sample time coordinate demands an explicit forecast ``time_coord``.

    The default integer continuation would otherwise raise an opaque cast error;
    the frequency of a datetime index is deliberately not guessed.
    """
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    days = np.datetime64("2024-01-01") + np.arange(n)
    future_covariates = empty_covariates(n + 3)
    post = draw_posterior(random.PRNGKey(3), fit, 60)
    fc = forecast(random.PRNGKey(4), RandomWalkModel(), post, data, future_covariates)
    tree = to_datatree(
        random.PRNGKey(2), fit, RandomWalkModel(), data, covariates, time_coord=list(days)
    )
    with pytest.raises(ValueError, match="pass explicit time_coord"):
        add_forecast_groups(tree, fc, future_covariates[n:])
    future_days = days[-1] + 1 + np.arange(3)
    out = add_forecast_groups(tree, fc, future_covariates[n:], time_coord=list(future_days))
    np.testing.assert_array_equal(out["predictions"].coords["time"].values, future_days)


def test_to_datatree_forecast_covariates_add_prediction_groups() -> None:
    """Covariates extending beyond the data attach the forecast groups in one call."""
    fit, data, _ = _mcmc_fit()
    n = data.shape[-2]
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, empty_covariates(n + 5))
    assert "predictions" in tree.children
    assert "predictions_constant_data" in tree.children
    np.testing.assert_array_equal(tree["predictions"].coords["time"].values, np.arange(n, n + 5))
    # constant_data holds only the in-sample slice; the future rows live in
    # predictions_constant_data.
    assert tree["constant_data"].sizes["time"] == n
    assert tree["predictions_constant_data"].sizes["time"] == 5


def test_to_datatree_forecast_keeps_mcmc_chain_structure() -> None:
    """The predictions group preserves the fit's real (chain, draw) layout."""
    fit, data, _ = _mcmc_fit(num_chains=2)
    n = data.shape[-2]
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, empty_covariates(n + 4))
    preds = tree["predictions"]
    assert preds.sizes["chain"] == 2
    assert preds.sizes["draw"] == 50
    assert preds.sizes["time"] == 4


def test_to_datatree_forecast_variational_draw_count() -> None:
    """A variational fit forecasts with the same draws as the in-sample predictive."""
    data = _series()
    n = data.shape[-2]
    svi = fit_svi(random.PRNGKey(1), RandomWalkModel(), data, empty_covariates(n), num_steps=40)
    tree = to_datatree(
        random.PRNGKey(2),
        svi,
        RandomWalkModel(),
        data,
        empty_covariates(n + 3),
        num_predictive_samples=20,
    )
    preds = tree["predictions"]
    assert preds.sizes["chain"] == 1
    assert preds.sizes["draw"] == 20
    assert tree["posterior_predictive"].sizes["draw"] == 20


def test_to_datatree_forecast_full_length_time_coord_splits() -> None:
    """With a horizon, an explicit time_coord covers the full covariates length."""
    fit, data, _ = _mcmc_fit()
    n = data.shape[-2]
    custom_time = list(range(300, 300 + n + 3))
    tree = to_datatree(
        random.PRNGKey(2),
        fit,
        RandomWalkModel(),
        data,
        empty_covariates(n + 3),
        time_coord=custom_time,
    )
    np.testing.assert_array_equal(tree["observed_data"].coords["time"].values, custom_time[:n])
    np.testing.assert_array_equal(tree["predictions"].coords["time"].values, custom_time[n:])


def test_to_datatree_forecast_rejects_insample_length_time_coord() -> None:
    """An in-sample-length time_coord with a horizon present raises by name."""
    fit, data, _ = _mcmc_fit()
    n = data.shape[-2]
    with pytest.raises(ValueError, match="time_coord has length"):
        to_datatree(
            random.PRNGKey(2),
            fit,
            RandomWalkModel(),
            data,
            empty_covariates(n + 3),
            time_coord=list(range(n)),
        )


def test_to_datatree_forecast_groups_via_dict_to_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forecast groups also go through arviz_base.dict_to_dataset (normative rule)."""
    calls = {"count": 0}
    real = arviz_base.dict_to_dataset

    def spy(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(arviz_base, "dict_to_dataset", spy)
    fit, data, _ = _mcmc_fit()
    n = data.shape[-2]
    to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, empty_covariates(n + 2))
    # The four in-sample groups plus predictions and predictions_constant_data.
    assert calls["count"] == 6


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


def test_posterior_time_site_shares_time_coord() -> None:
    """A per-time posterior site listed in posterior_dims shares the tree time coord."""
    fit, data, covariates = _mcmc_fit()
    n = data.shape[-2]
    custom_time = list(range(200, 200 + n))
    tree = to_datatree(
        random.PRNGKey(2),
        fit,
        RandomWalkModel(),
        data,
        covariates,
        time_coord=custom_time,
        posterior_dims={"drift": ["time", "obs_dim"]},
    )
    drift_time = tree["posterior"]["drift"].coords["time"].values
    np.testing.assert_array_equal(drift_time, custom_time)
    np.testing.assert_array_equal(
        drift_time, tree["posterior_predictive"]["obs"].coords["time"].values
    )


def test_posterior_dims_default_is_backward_compatible() -> None:
    """Without posterior_dims, per-time sites keep ArviZ's auto-named dims."""
    fit, data, covariates = _mcmc_fit()
    tree = to_datatree(random.PRNGKey(2), fit, RandomWalkModel(), data, covariates)
    assert "time" not in tree["posterior"]["drift"].dims


def test_to_datatree_splits_rng_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Posterior draws and in-sample predictive must receive distinct subkeys.

    Spies on the ``draw_posterior``/``predict_in_sample`` names as looked up
    inside ``convert``; a variational fit exercises both draws.
    """
    import numpyro_forecast.convert as convert_mod

    captured: dict[str, Array] = {}
    real_draw = convert_mod.draw_posterior
    real_pred = convert_mod.predict_in_sample

    def spy_draw(rng_key: Array, fit: object, num: int) -> object:
        captured["post"] = rng_key
        return real_draw(rng_key, fit, num)

    def spy_pred(
        rng_key: Array, model: ForecastModel, posterior: dict[str, Array], covariates: Array
    ) -> object:
        captured["pred"] = rng_key
        return real_pred(rng_key, model, posterior, covariates)

    monkeypatch.setattr(convert_mod, "draw_posterior", spy_draw)
    monkeypatch.setattr(convert_mod, "predict_in_sample", spy_pred)

    data = _series()
    covariates = empty_covariates(data.shape[-2])
    svi = fit_svi(random.PRNGKey(1), RandomWalkModel(), data, covariates, num_steps=40)
    parent = random.PRNGKey(2)
    to_datatree(parent, svi, RandomWalkModel(), data, covariates, num_predictive_samples=20)

    assert not jnp.array_equal(captured["post"], captured["pred"])
    assert not jnp.array_equal(captured["post"], parent)
    assert not jnp.array_equal(captured["pred"], parent)


def test_to_datatree_deterministic_given_key() -> None:
    """Two calls with the same rng_key produce identical trees (derived split)."""
    data = _series()
    covariates = empty_covariates(data.shape[-2])
    svi = fit_svi(random.PRNGKey(1), RandomWalkModel(), data, covariates, num_steps=40)
    a = to_datatree(
        random.PRNGKey(7), svi, RandomWalkModel(), data, covariates, num_predictive_samples=20
    )
    b = to_datatree(
        random.PRNGKey(7), svi, RandomWalkModel(), data, covariates, num_predictive_samples=20
    )
    np.testing.assert_array_equal(
        np.asarray(a["posterior_predictive"]["obs"]),
        np.asarray(b["posterior_predictive"]["obs"]),
    )
    np.testing.assert_array_equal(
        np.asarray(a["posterior"]["drift"]),
        np.asarray(b["posterior"]["drift"]),
    )
