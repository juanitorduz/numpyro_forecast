"""Tests for the ArviZ DataTree export (roadmap §5)."""

import warnings
from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import xarray
import xarray.testing as xarray_testing
from conftest import (
    as_model,
    empty_covariates,
    fail_devices_for,
    nuts_samples,
    rw_model,
    svi_guide_params,
)
from jax import Array, random
from jax.typing import ArrayLike
from numpyro.infer import Predictive

from numpyro_forecast.convert import (
    _reshape_chains,
    add_forecast_groups,
    predictions_to_datatree,
    to_datatree,
)
from numpyro_forecast.models import Horizon, innovations, predict
from numpyro_forecast.predictive import draw_posterior, forecast
from numpyro_forecast.typing import ForecastModel

arviz_base = pytest.importorskip("arviz_base")
arviz_stats = pytest.importorskip("arviz_stats")


def _model() -> ForecastModel:
    """The functional random-walk model matching conftest's posterior-drawing helpers."""
    return rw_model


def _series(n: int = 18) -> Array:
    return jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (n, 1)), axis=-2)


def _mcmc_posterior(
    num_chains: int = 2, t: int = 18, num_warmup: int = 50, num_samples: int = 50
) -> tuple[dict[str, Array], Array, Array]:
    """An MCMC posterior plus the data/covariates it was fit on.

    ``nuts_samples`` builds ``data`` internally with the exact deterministic
    recipe `_series()` reproduces, so calling both with the same ``t``
    yields the data the posterior was actually fit on. The model never reads
    covariate *values* (only the time-axis length, via ``Horizon``), so the
    ``covariates`` passed to ``to_datatree``/``forecast`` in a given test may
    differ in shape from ``empty_covariates(t)`` used here for fitting.
    """
    posterior = nuts_samples(
        t, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    return posterior, _series(t), empty_covariates(t)


def _svi_posterior(
    num_draws: int, t: int = 18, num_steps: int = 40
) -> tuple[dict[str, Array], Array, Array]:
    """An SVI-drawn posterior plus the data/covariates it was fit on."""
    guide, params = svi_guide_params(t, num_steps=num_steps)
    posterior = cast(
        "dict[str, Array]", draw_posterior(random.PRNGKey(3), guide, params, num_draws)
    )
    return posterior, _series(t), empty_covariates(t)


def test_to_datatree_mcmc_groups_and_dims() -> None:
    posterior, data, covariates = _mcmc_posterior()
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
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
    posterior, data, covariates = _mcmc_posterior()
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    obs = np.asarray(tree["observed_data"]["obs"])
    np.testing.assert_allclose(obs, np.asarray(data), rtol=1e-6)
    const = np.asarray(tree["constant_data"]["covariates"])
    assert const.shape == (data.shape[-2], 0)


def test_mcmc_chain_reshape_roundtrip() -> None:
    """The (num_samples,...) -> (chain, draw, ...) reshape matches group_by_chain."""
    posterior, data, covariates = _mcmc_posterior(num_chains=2)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    reshaped = np.asarray(tree["posterior"]["sigma"])  # (chain, draw)
    flat = np.asarray(posterior["sigma"])  # (num_samples,)
    # get_samples(group_by_chain=False) concatenates chains in order, so a plain
    # reshape recovers the per-chain layout.
    np.testing.assert_allclose(reshaped.reshape(-1), flat, rtol=1e-6)
    assert reshaped.shape == (2, flat.shape[0] // 2)


def test_rhat_runs_warning_free_on_two_chain_posterior() -> None:
    """Acceptance: az rhat runs warning-free on a 2-chain posterior."""
    posterior, data, covariates = _mcmc_posterior(num_chains=2)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rhat = arviz_stats.rhat(tree["posterior"].dataset)
    assert "sigma" in rhat


def test_to_datatree_svi_posterior_single_chain_has_no_variational_attrs() -> None:
    """A variational (SVI-drawn) posterior gets a single pseudo-chain and no dropped attrs.

    The fit-type-derived ``variational``/``is_mcmc`` attrs are gone now that
    ``to_datatree`` never sees a fit object, only a posterior dict.
    """
    posterior, data, covariates = _svi_posterior(num_draws=40)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates)
    assert tree["posterior"].sizes["chain"] == 1
    assert tree["posterior"].sizes["draw"] == 40
    assert "variational" not in tree["posterior"].dataset.attrs
    assert "is_mcmc" not in tree["posterior"].dataset.attrs


def test_to_datatree_time_coord_threading() -> None:
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    custom_time = list(range(100, 100 + n))
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        covariates,
        num_chains=2,
        time_coord=custom_time,
    )
    np.testing.assert_array_equal(tree["observed_data"].coords["time"].values, custom_time)
    np.testing.assert_array_equal(tree["posterior_predictive"].coords["time"].values, custom_time)


def test_add_forecast_groups_adds_groups_and_time_continuation() -> None:
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 5)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    out = add_forecast_groups(tree, fc, future_covariates[n:])
    assert "predictions" in out.children
    assert "predictions_constant_data" in out.children
    times = out["predictions"].coords["time"].values
    np.testing.assert_array_equal(times, np.arange(n, n + 5))
    assert out["predictions"]["obs"].dims == ("chain", "draw", "time", "obs_dim")
    # The original tree is untouched.
    assert "predictions" not in tree.children


def test_add_forecast_groups_explicit_time_coord() -> None:
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 3)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    out = add_forecast_groups(tree, fc, future_covariates[n:], time_coord=[900, 901, 902])
    np.testing.assert_array_equal(out["predictions"].coords["time"].values, [900, 901, 902])


def test_add_forecast_groups_rejects_wrong_length_time_coord() -> None:
    """A ``time_coord`` whose length differs from the forecast horizon raises by name.

    Without the guard the mismatch flowed into ``dict_to_dataset`` and died deep
    in xarray with a shape message that never mentioned ``time_coord``.
    """
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 3)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    with pytest.raises(ValueError, match="time_coord has length 2"):
        add_forecast_groups(tree, fc, future_covariates[n:], time_coord=[900, 901])


def test_add_forecast_groups_noninteger_time_requires_explicit_coord() -> None:
    """A datetime64 in-sample time coordinate demands an explicit forecast ``time_coord``.

    The default integer continuation would otherwise raise an opaque cast error;
    the frequency of a datetime index is deliberately not guessed.
    """
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    days = np.datetime64("2024-01-01") + np.arange(n)
    future_covariates = empty_covariates(n + 3)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        covariates,
        num_chains=2,
        time_coord=list(days),
    )
    with pytest.raises(ValueError, match="pass explicit time_coord"):
        add_forecast_groups(tree, fc, future_covariates[n:])
    future_days = days[-1] + 1 + np.arange(3)
    out = add_forecast_groups(tree, fc, future_covariates[n:], time_coord=list(future_days))
    np.testing.assert_array_equal(out["predictions"].coords["time"].values, future_days)


def test_to_datatree_forecast_covariates_add_prediction_groups() -> None:
    """Covariates extending beyond the data attach the forecast groups in one call."""
    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    tree = to_datatree(
        random.PRNGKey(2), _model(), posterior, data, empty_covariates(n + 5), num_chains=2
    )
    assert "predictions" in tree.children
    assert "predictions_constant_data" in tree.children
    np.testing.assert_array_equal(tree["predictions"].coords["time"].values, np.arange(n, n + 5))
    # constant_data holds only the in-sample slice; the future rows live in
    # predictions_constant_data.
    assert tree["constant_data"].sizes["time"] == n
    assert tree["predictions_constant_data"].sizes["time"] == 5


def test_to_datatree_forecast_keeps_mcmc_chain_structure() -> None:
    """The predictions group preserves the requested (chain, draw) layout."""
    posterior, data, _ = _mcmc_posterior(num_chains=2)
    n = data.shape[-2]
    tree = to_datatree(
        random.PRNGKey(2), _model(), posterior, data, empty_covariates(n + 4), num_chains=2
    )
    preds = tree["predictions"]
    assert preds.sizes["chain"] == 2
    assert preds.sizes["draw"] == 50
    assert preds.sizes["time"] == 4


def test_to_datatree_forecast_variational_draw_count() -> None:
    """A variational posterior forecasts with the same draws as the in-sample predictive."""
    posterior, data, _ = _svi_posterior(num_draws=20)
    n = data.shape[-2]
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, empty_covariates(n + 3))
    preds = tree["predictions"]
    assert preds.sizes["chain"] == 1
    assert preds.sizes["draw"] == 20
    assert tree["posterior_predictive"].sizes["draw"] == 20


def test_to_datatree_forecast_full_length_time_coord_splits() -> None:
    """With a horizon, an explicit time_coord covers the full covariates length."""
    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    custom_time = list(range(300, 300 + n + 3))
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        empty_covariates(n + 3),
        num_chains=2,
        time_coord=custom_time,
    )
    np.testing.assert_array_equal(tree["observed_data"].coords["time"].values, custom_time[:n])
    np.testing.assert_array_equal(tree["predictions"].coords["time"].values, custom_time[n:])


def test_to_datatree_forecast_rejects_insample_length_time_coord() -> None:
    """An in-sample-length time_coord with a horizon present raises by name."""
    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    with pytest.raises(ValueError, match="time_coord has length"):
        to_datatree(
            random.PRNGKey(2),
            _model(),
            posterior,
            data,
            empty_covariates(n + 3),
            num_chains=2,
            time_coord=list(range(n)),
        )


def test_to_datatree_rejects_short_covariates() -> None:
    """Covariates shorter than the data raise a clear error instead of a shape failure."""
    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    with pytest.raises(ValueError, match="covariates cover"):
        to_datatree(
            random.PRNGKey(2), _model(), posterior, data, empty_covariates(n - 3), num_chains=2
        )


@pytest.mark.parametrize(
    ("data", "covariates", "name"),
    [
        (jnp.zeros((6, 1)), jnp.arange(8.0), "covariates"),
        (jnp.zeros(6), jnp.zeros((8, 0)), "data"),
    ],
)
def test_to_datatree_rejects_inputs_without_time_axis(
    data: Array, covariates: Array, name: str
) -> None:
    posterior = {"level": jnp.zeros(4), "drift": jnp.zeros((4, 6, 1))}
    with pytest.raises(ValueError, match=rf"{name} must have time at axis -2"):
        to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates)


def test_to_datatree_rejects_posterior_leaf_without_sample_axis() -> None:
    posterior = {"level": jnp.array(0.0), "drift": jnp.zeros((4, 6, 1))}
    with pytest.raises(ValueError, match=r"posterior sites \['level'\] have no sample axis"):
        to_datatree(random.PRNGKey(2), _model(), posterior, jnp.zeros((6, 1)), empty_covariates(8))


def test_to_datatree_rejects_horizon_sites_before_drawing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #161: a posterior with horizon values raises before any predictive draw.

    The drivers check the posterior too, but `to_datatree()` calls them only
    after it has built the posterior group, and the in-sample draws come first.
    """
    import numpyro_forecast.convert as convert_mod

    def fail(*args: object, **kwargs: object) -> object:
        msg = "to_datatree drew predictive samples before it checked the posterior"
        raise AssertionError(msg)

    monkeypatch.setattr(convert_mod, "predict_in_sample", fail)
    monkeypatch.setattr(convert_mod, "forecast", fail)
    posterior = {
        "drift_scale": jnp.ones(4),
        "sigma": jnp.ones(4),
        "drift": jnp.zeros((4, 6, 1)),
        "drift_future": jnp.zeros((4, 2, 1)),
    }
    with pytest.raises(ValueError, match=r"forecast-horizon sites \['drift_future'\]"):
        to_datatree(random.PRNGKey(2), _model(), posterior, jnp.zeros((6, 1)), empty_covariates(8))


def test_to_datatree_rejects_data_without_observations() -> None:
    posterior = {"level": jnp.zeros(4), "drift": jnp.zeros((4, 0, 1))}
    with pytest.raises(ValueError, match=r"data has no observed time steps \(shape \(0, 1\)\)"):
        to_datatree(random.PRNGKey(2), _model(), posterior, jnp.zeros((0, 1)), empty_covariates(8))


def test_to_datatree_coords_reach_forecast_groups() -> None:
    """User coords propagate to the forecast groups; the forecast time coordinate wins."""
    posterior, data, _ = _mcmc_posterior(num_chains=1)
    n = data.shape[-2]
    labels = ["x0", "x1"]
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        jnp.zeros((n + 3, 2)),
        coords={"covariate_dim": labels},
    )
    np.testing.assert_array_equal(tree["constant_data"].coords["covariate_dim"].values, labels)
    np.testing.assert_array_equal(
        tree["predictions_constant_data"].coords["covariate_dim"].values, labels
    )
    np.testing.assert_array_equal(tree["predictions"].coords["time"].values, np.arange(n, n + 3))


def test_to_datatree_covariate_dims_panel_tensor() -> None:
    """3-D (channel, time, series) covariates keep their layout under covariate_dims."""
    posterior, data, _ = _svi_posterior(num_draws=40)
    n = data.shape[-2]
    labels = ["availability", "discount", "holiday"]
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        jnp.zeros((3, n + 2, 1)),
        coords={"channel": labels},
        covariate_dims=["channel", "time", "series"],
    )
    const = tree["constant_data"]["covariates"]
    assert const.dims == ("channel", "time", "series")
    assert const.sizes == {"channel": 3, "time": n, "series": 1}
    np.testing.assert_array_equal(const.coords["channel"].values, labels)
    future = tree["predictions_constant_data"]["covariates"]
    assert future.dims == ("channel", "time", "series")
    assert future.sizes == {"channel": 3, "time": 2, "series": 1}
    np.testing.assert_array_equal(future.coords["channel"].values, labels)


def test_to_datatree_rejects_wrong_covariate_dims_length() -> None:
    """A covariate_dims length mismatch raises a clear error before any sampling."""
    posterior, data, covariates = _mcmc_posterior()
    with pytest.raises(ValueError, match="covariate_dims"):
        to_datatree(
            random.PRNGKey(2),
            _model(),
            posterior,
            data,
            covariates,
            num_chains=2,
            covariate_dims=["time"],
        )


def test_add_forecast_groups_covariate_dims() -> None:
    """add_forecast_groups stores tensor covariates under the given dim names."""
    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    future_covariates = jnp.zeros((2, n + 5, 1))
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        future_covariates[..., :n, :],
        num_chains=2,
        covariate_dims=["channel", "time", "series"],
    )
    out = add_forecast_groups(
        tree,
        fc,
        future_covariates[..., n:, :],
        covariate_dims=["channel", "time", "series"],
    )
    future = out["predictions_constant_data"]["covariates"]
    assert future.dims == ("channel", "time", "series")
    assert future.sizes == {"channel": 2, "time": 5, "series": 1}


def test_add_forecast_groups_inherits_covariate_dims() -> None:
    """Omitted covariate_dims inherits the names stored on the tree's constant_data.

    Before the cross-check, custom but still 2-D names (here ``feature`` for the
    trailing axis) were silently replaced by the default ``covariate_dim``,
    leaving constant_data and predictions_constant_data disagreeing on axis
    names.
    """
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 5)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        covariates,
        num_chains=2,
        covariate_dims=["time", "feature"],
    )
    out = add_forecast_groups(tree, fc, future_covariates[n:])
    assert out["constant_data"]["covariates"].dims == ("time", "feature")
    assert out["predictions_constant_data"]["covariates"].dims == ("time", "feature")


def test_add_forecast_groups_rejects_mismatched_covariate_dims() -> None:
    """Explicit covariate_dims disagreeing with the tree's stored names raise."""
    from numpyro_forecast.exceptions import CovariateDimsError

    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 5)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        covariates,
        num_chains=2,
        covariate_dims=["time", "feature"],
    )
    with pytest.raises(CovariateDimsError, match="disagree with the names"):
        add_forecast_groups(
            tree, fc, future_covariates[n:], covariate_dims=["time", "covariate_dim"]
        )


def test_add_forecast_groups_inherited_dims_reject_wrong_ndim() -> None:
    """Inherited dims that cannot cover every covariates_future axis raise by name."""
    from numpyro_forecast.exceptions import CovariateDimsError

    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    insample_covariates = jnp.zeros((2, n, 1))
    future_covariates = empty_covariates(n + 5)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        insample_covariates,
        num_chains=2,
        covariate_dims=["channel", "time", "series"],
    )
    with pytest.raises(CovariateDimsError, match="carry 3 axis names"):
        add_forecast_groups(tree, fc, future_covariates[n:])


# --- batched (*batch, time, obs) data (issue #172) --------------------------


def _panel_body(h: Horizon, covariates: Array) -> None:
    """A model whose predictor takes the ``(batch, time, obs)`` layout of ``covariates``."""
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 0.5))
    drift = innovations(h, "drift", dist.Normal(0.0, 0.1))
    predict(h, dist.Normal(0.0, sigma), covariates + jnp.cumsum(drift, axis=-2))


def _panel(
    batch: int, t: int, horizon: int, num_draws: int = 4
) -> tuple[ForecastModel, dict[str, Array], Array, Array]:
    """A batched export setup: ``data`` of shape ``(batch, t, 2)`` with distinct values."""
    data = jnp.arange(batch * t * 2, dtype=jnp.float32).reshape(batch, t, 2)
    covariates = jnp.zeros((batch, t + horizon, 2))
    posterior = {"sigma": jnp.full((num_draws,), 0.3), "drift": jnp.zeros((num_draws, t, 1))}
    return as_model(_panel_body), posterior, data, covariates


@pytest.mark.parametrize(
    ("batch", "t", "horizon"),
    [(3, 5, 0), (3, 5, 2), (5, 5, 0), (5, 5, 2)],
    ids=["batch-ne-time", "batch-ne-time-horizon", "batch-eq-time", "batch-eq-time-horizon"],
)
def test_to_datatree_batched_data_keeps_time_at_axis_minus_two(
    batch: int, t: int, horizon: int
) -> None:
    """``(batch, time, obs)`` data puts the ``time`` dim on axis -2 of every ``obs`` group.

    The fixed ``("time", "obs_dim")`` names used to land on the batch and time
    axes: a batch size other than the time length failed with a size conflict,
    and an equal one exported a tree whose ``sel(time=k)`` returned batch row
    ``k``.
    """
    model, posterior, data, covariates = _panel(batch, t, horizon)
    tree = to_datatree(
        random.PRNGKey(1),
        model,
        posterior,
        data,
        covariates,
        covariate_dims=["batch", "time", "covariate_dim"],
    )
    dims = ("obs_batch_0", "time", "obs_dim")
    observed = tree["observed_data"]["obs"]
    assert observed.dims == dims
    for k in range(t):
        np.testing.assert_array_equal(observed.sel(time=k).values, np.asarray(data)[..., k, :])
    assert tree["posterior_predictive"]["obs"].dims == ("chain", "draw", *dims)
    if horizon:
        predictions = tree["predictions"]["obs"]
        assert predictions.dims == ("chain", "draw", *dims)
        np.testing.assert_array_equal(predictions.coords["time"].values, np.arange(t, t + horizon))


def test_to_datatree_obs_dims_name_every_data_axis() -> None:
    """``obs_dims`` names the axes of all three ``obs`` groups, which share the user coords."""
    model, posterior, data, covariates = _panel(3, 5, 2)
    origins = ["SFO", "OAK", "SJC"]
    tree = to_datatree(
        random.PRNGKey(1),
        model,
        posterior,
        data,
        covariates,
        coords={"origin": origins},
        covariate_dims=["origin", "time", "covariate_dim"],
        obs_dims=["origin", "time", "destin"],
    )
    for group in ("observed_data", "posterior_predictive", "predictions"):
        obs = tree[group]["obs"]
        assert obs.dims[-3:] == ("origin", "time", "destin")
        np.testing.assert_array_equal(obs.coords["origin"].values, origins)
    np.testing.assert_array_equal(
        tree["observed_data"]["obs"].sel(origin="OAK").values, np.asarray(data)[1]
    )


_LEVEL_COVARIATE_DIMS = ["covariate", "time", "obs_dim"]
_LEVEL_COORDS = {"covariate": ["sales", "availability"], "obs_dim": ["s1", "s2"]}


def _level_model(covariates: Array, data: Array | None = None) -> None:
    """``obs = level + covariates[1]`` over two series, for ``(covariate, time, k)`` covariates.

    The near-zero observation noise makes every draw equal its own posterior
    ``level`` plus the covariate, so a forecast group can be checked against
    the posterior draw by draw. ``k`` is ``2`` or ``1`` (one column broadcast
    over both series).
    """
    h = Horizon.from_data(covariates, data)
    level = jnp.asarray(numpyro.sample("level", dist.Normal(0.0, 1.0)))
    predict(h, dist.Normal(0.0, 1e-6), level + covariates[1] + jnp.zeros(2))


def _level_tree(
    posterior: Mapping[str, ArrayLike],
    covariates: Array,
    covariate_dims: list[str] = _LEVEL_COVARIATE_DIMS,
    coords: Mapping[str, list[str]] = _LEVEL_COORDS,
    num_chains: int = 4,
) -> xarray.DataTree:
    """Export `_level_model` on ``(4, 2)`` data with the given chains and labels."""
    return to_datatree(
        random.PRNGKey(0),
        _level_model,
        posterior,
        jnp.ones((4, 2)),
        covariates,
        num_chains=num_chains,
        covariate_dims=covariate_dims,
        coords=coords,
    )


@pytest.mark.parametrize(
    ("obs_dims", "match"),
    [
        (["time", "obs_dim"], r"obs_dims has 2 names \['time', 'obs_dim'\] but data has 3"),
        (["origin", "date", "destin"], r"obs_dims must name the time axis \(axis -2\) 'time'"),
        (["obs_dim", "time", "obs_dim"], r"obs_dims repeats a name"),
        (["chain", "time", "obs_dim"], r"obs_dims uses the sample dimension names \['chain'\]"),
    ],
)
def test_to_datatree_rejects_malformed_obs_dims(obs_dims: list[str], match: str) -> None:
    model, posterior, data, covariates = _panel(3, 5, 0)
    with pytest.raises(ValueError, match=match):
        to_datatree(
            random.PRNGKey(1),
            model,
            posterior,
            data,
            covariates,
            covariate_dims=["origin", "time", "covariate_dim"],
            obs_dims=obs_dims,
        )


def test_to_datatree_rejects_obs_draws_with_more_axes_than_data() -> None:
    """A model whose ``obs`` site has more axes than ``data`` raises instead of mislabeling.

    With a batch size equal to the time length, the auto-named extra axis used
    to export without an error, with ``time`` on the batch axis.
    """
    model, posterior, _, covariates = _panel(5, 5, 0)
    with pytest.raises(
        ValueError, match=r"the in-sample predictive draws have shape \(5, 5, 2\) per draw"
    ):
        to_datatree(
            random.PRNGKey(1),
            model,
            posterior,
            jnp.zeros((5, 2)),
            covariates,
            covariate_dims=["batch", "time", "covariate_dim"],
        )


def _shared_level_model(covariates: Array, data: Array | None = None) -> None:
    """A level shared by every series, with a ``(duration, 1)`` predictor and no series plate."""
    h = Horizon.from_data(covariates, data)
    level = numpyro.sample("level", dist.Normal(0.0, 1.0))
    predict(h, dist.Normal(0.0, 1.0), jnp.full((h.duration, 1), level))


def test_to_datatree_rejects_in_sample_draws_narrower_than_data() -> None:
    """Issue #259: one in-sample column for three observed series raises.

    `~~numpyro_forecast.predictive.predict_in_sample()` calls the model with
    ``data=None``, so a predictor shared across series without a series plate
    draws one column. The export wrote it next to the three observed columns
    without an error.
    """
    with pytest.raises(
        ValueError,
        match=r"the in-sample predictive draws have shape \(5, 1\) per draw, but data has "
        r"shape \(5, 3\)",
    ):
        to_datatree(
            random.PRNGKey(1),
            _shared_level_model,
            {"level": jnp.zeros(4)},
            jnp.zeros((5, 3)),
            empty_covariates(5),
        )


def test_to_datatree_drops_an_obs_key_from_the_posterior() -> None:
    """Issue #260: an ``"obs"`` key is neither a posterior variable nor a pin on the draws.

    Merged from a ``Predictive`` call with ``data``, it used to export the data
    once per draw in the ``posterior`` group and as every in-sample draw.
    """
    t = 6
    covariates = empty_covariates(t)
    data = 0.1 * random.normal(random.PRNGKey(0), (t, 1))
    posterior = {"drift_scale": jnp.ones(4), "sigma": jnp.ones(4), "drift": jnp.zeros((4, t, 1))}
    observed = Predictive(rw_model, posterior_samples=posterior)(
        random.PRNGKey(1), covariates, data
    )
    tree = to_datatree(random.PRNGKey(2), rw_model, {**posterior, **observed}, data, covariates)
    assert set(tree["posterior"].data_vars) == set(posterior)
    in_sample = np.asarray(tree["posterior_predictive"]["obs"])
    assert not np.allclose(in_sample, np.asarray(data))


@pytest.mark.parametrize(
    ("obs_dims", "coords"),
    [(None, None), (["origin", "time", "destin"], {"origin": ["SFO", "OAK", "SJC"]})],
    ids=["default-names", "named-and-labeled"],
)
def test_add_forecast_groups_lays_out_batched_forecast_like_to_datatree(
    obs_dims: list[str] | None, coords: dict[str, list[str]] | None
) -> None:
    """A ``(sample, batch, future, obs)`` forecast gets the forecast groups of `to_datatree()`.

    Both forecast groups take the dimension names and labels of their in-sample
    counterparts, so label-based selection works the same whichever route
    attached the forecast.
    """
    model, posterior, data, covariates = _panel(3, 5, 2)
    n = data.shape[-2]
    covariate_dims = ["origin", "time", "covariate_dim"]
    one_call = to_datatree(
        random.PRNGKey(1),
        model,
        posterior,
        data,
        covariates,
        coords=coords,
        covariate_dims=covariate_dims,
        obs_dims=obs_dims,
    )
    tree = to_datatree(
        random.PRNGKey(1),
        model,
        posterior,
        data,
        covariates[:, :n],
        coords=coords,
        covariate_dims=covariate_dims,
        obs_dims=obs_dims,
    )
    fc = forecast(random.PRNGKey(4), model, posterior, data, covariates)
    out = add_forecast_groups(tree, fc, covariates[:, n:])
    for group, name in (("predictions", "obs"), ("predictions_constant_data", "covariates")):
        expected = one_call[group][name]
        actual = out[group][name]
        assert actual.dims == expected.dims
        for dim in expected.dims:
            np.testing.assert_array_equal(actual.coords[dim].values, expected.coords[dim].values)
    np.testing.assert_array_equal(np.asarray(out["predictions"]["obs"])[0], np.asarray(fc))


def test_add_forecast_groups_rejects_forecast_layout_mismatch() -> None:
    """A forecast with other axes than the tree's stored ``obs`` raises by name."""
    model, posterior, data, covariates = _panel(3, 5, 2)
    n = data.shape[-2]
    tree = to_datatree(
        random.PRNGKey(1),
        model,
        posterior,
        data,
        covariates[:, :n],
        covariate_dims=["origin", "time", "covariate_dim"],
    )
    fc = forecast(random.PRNGKey(4), model, posterior, data, covariates)
    with pytest.raises(ValueError, match=r"forecast_samples have shape \(2, 2\) per draw"):
        add_forecast_groups(tree, fc[:, 0], covariates[:, n:])


def test_add_forecast_groups_squeezed_observed_data_takes_default_names() -> None:
    """An ``observed_data`` squeezed to one series does not block the forecast groups.

    The example notebooks select one series with ``isel(obs_dim=0)`` before
    plotting. Its stored ``obs`` has no time axis at -2 to inherit, so the
    forecast takes the default names.
    """
    posterior, data, covariates = _svi_posterior(num_draws=10)
    n = data.shape[-2]
    future_covariates = empty_covariates(n + 3)
    fc = forecast(random.PRNGKey(4), _model(), posterior, data, future_covariates)
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates)
    tree["observed_data"] = tree["observed_data"].dataset.isel(obs_dim=0)
    out = add_forecast_groups(tree, fc, future_covariates[n:])
    assert out["predictions"]["obs"].dims == ("chain", "draw", "time", "obs_dim")


@pytest.mark.parametrize(
    ("covariate_dims", "coords", "width"),
    [
        pytest.param(_LEVEL_COVARIATE_DIMS, _LEVEL_COORDS, 2, id="named-labels"),
        # The in-sample labels of a time axis named "step" must not reach the forecast.
        pytest.param(["covariate", "step", "obs_dim"], _LEVEL_COORDS, 2, id="time-named-step"),
        # One covariate column for both series: 'obs_dim' has 2 labels on obs, 1 on covariates.
        pytest.param(
            _LEVEL_COVARIATE_DIMS, {"covariate": ["sales", "availability"]}, 1, id="broadcast"
        ),
    ],
)
def test_add_forecast_groups_matches_to_datatree_horizon_route(
    covariate_dims: list[str], coords: dict[str, list[str]], width: int
) -> None:
    """Both forecast routes build the same groups: chain split, draw order and labels.

    Regression for #176: ``add_forecast_groups`` stored the draws as one
    pseudo-chain and dropped the user coordinates, so arithmetic with the
    4-chain posterior silently kept 5 of 20 draws and the labels fell back to
    integers.
    """
    posterior = {"level": jnp.arange(20, dtype=jnp.float32)}
    covariates = jnp.ones((2, 6, width))
    one_call = _level_tree(posterior, covariates, covariate_dims, coords)
    fc = forecast(random.PRNGKey(1), _level_model, posterior, jnp.ones((4, 2)), covariates)
    insample = _level_tree(posterior, covariates[:, :4], covariate_dims, coords)
    step_by_step = add_forecast_groups(insample, fc, covariates[:, 4:])
    for group in ("predictions", "predictions_constant_data"):
        expected = one_call[group].dataset
        xarray_testing.assert_allclose(step_by_step[group].dataset, expected, atol=1e-4)
        for name, coord in expected.coords.items():
            assert step_by_step[group].dataset.coords[name].dtype == coord.dtype, name


@pytest.mark.parametrize(
    ("selection", "sizes"),
    [
        pytest.param({"draw": slice(None, None, 2)}, {"chain": 2, "draw": 5}, id="thinned"),
        pytest.param(
            {"chain": 1, "draw": slice(None, None, 2)}, {"chain": 1, "draw": 5}, id="one-chain"
        ),
    ],
)
def test_add_forecast_groups_keeps_the_posterior_draw_labels(
    selection: dict[str, int | slice], sizes: dict[str, int]
) -> None:
    """The forecast draws take the tree's chain and draw labels, so they align with the posterior.

    A tree thinned with ``isel`` keeps the draw labels ``0, 2, 4, ...``, also
    when one selected chain leaves a posterior with no ``chain`` dimension; a
    forecast drawn from that subset must line up with it label by label.
    """
    posterior = {"level": jnp.arange(20, dtype=jnp.float32)}
    covariates = jnp.ones((2, 6, 2))
    subset = _level_tree(posterior, covariates[:, :4], num_chains=2).isel(selection)
    # A C-order flatten of (chain, draw) is the order of stack(sample=("chain", "draw")).
    subset_level = np.asarray(subset["posterior"].dataset["level"]).reshape(-1)
    fc = forecast(
        random.PRNGKey(1), _level_model, {"level": subset_level}, jnp.ones((4, 2)), covariates
    )
    out = add_forecast_groups(subset, fc, covariates[:, 4:])
    residual = out["predictions"].dataset["obs"] - out["posterior"].dataset["level"]
    assert residual.sizes == {**sizes, "time": 2, "obs_dim": 2}
    np.testing.assert_allclose(residual.values, 1.0, atol=1e-4)


@pytest.mark.parametrize(
    ("forecast_shape", "future_steps", "match"),
    [
        ((10, 2, 2), 2, r"forecast_samples has 10 draws but the tree's posterior group has 20"),
        ((20, 2, 2), 1, r"covariates_future has length 1 along the time axis \(-2\) but"),
        ((20, 2, 3), 2, r"forecast_samples has size 3 along 'obs_dim' but the tree's"),
    ],
)
def test_add_forecast_groups_rejects_draws_off_the_tree_layout(
    forecast_shape: tuple[int, int, int], future_steps: int, match: str
) -> None:
    """Draws that cannot take the tree's sample layout or labels raise by name."""
    covariates = jnp.ones((2, 4 + future_steps, 2))
    tree = _level_tree({"level": jnp.arange(20, dtype=jnp.float32)}, covariates[:, :4])
    with pytest.raises(ValueError, match=match):
        add_forecast_groups(tree, np.zeros(forecast_shape), covariates[:, 4:])


def test_to_datatree_forecast_groups_via_dict_to_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forecast groups also go through arviz_base.dict_to_dataset (normative rule)."""
    calls = {"count": 0}
    real = arviz_base.dict_to_dataset

    def spy(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(arviz_base, "dict_to_dataset", spy)
    posterior, data, _ = _mcmc_posterior()
    n = data.shape[-2]
    to_datatree(
        random.PRNGKey(2), _model(), posterior, data, empty_covariates(n + 2), num_chains=2
    )
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
    posterior, data, covariates = _mcmc_posterior()
    to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    # posterior + posterior_predictive + observed_data + constant_data.
    assert calls["count"] == 4


def test_posterior_time_site_shares_time_coord() -> None:
    """A per-time posterior site listed in posterior_dims shares the tree time coord."""
    posterior, data, covariates = _mcmc_posterior()
    n = data.shape[-2]
    custom_time = list(range(200, 200 + n))
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        covariates,
        num_chains=2,
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
    posterior, data, covariates = _mcmc_posterior()
    tree = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=2)
    assert "time" not in tree["posterior"]["drift"].dims


def test_to_datatree_rejects_non_divisible_num_chains() -> None:
    """A posterior sample count not evenly divisible by num_chains raises by name."""
    posterior = nuts_samples(18, num_warmup=20, num_samples=20)
    data = _series(18)
    covariates = empty_covariates(18)
    with pytest.raises(
        ValueError, match=r"posterior sample count 20 is not evenly divisible by num_chains 3"
    ):
        to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, num_chains=3)


@pytest.mark.parametrize("num_chains", [0, -1, -2])
def test_to_datatree_rejects_non_positive_num_chains(num_chains: int) -> None:
    """``num_chains < 1`` raises by name instead of a bare ZeroDivisionError or reshape error."""
    posterior = {"level": jnp.zeros(6), "drift": jnp.zeros((6, 18, 1))}
    with pytest.raises(
        ValueError, match=rf"num_chains must be a positive integer, got {num_chains}"
    ):
        to_datatree(
            random.PRNGKey(2),
            _model(),
            posterior,
            _series(18),
            empty_covariates(18),
            num_chains=num_chains,
        )


@pytest.mark.parametrize("num_chains", [0, -3, True, False])
def test_reshape_chains_rejects_non_positive_num_chains(num_chains: int) -> None:
    with pytest.raises(ValueError, match="num_chains must be a positive integer"):
        _reshape_chains(np.zeros((6, 2)), num_chains)


def test_to_datatree_accepts_numpy_posterior() -> None:
    """A NumPy posterior dict (host-offloaded draws) passes straight through."""
    posterior, data, covariates = _mcmc_posterior(num_chains=1)
    numpy_posterior = {name: np.asarray(value) for name, value in posterior.items()}
    tree = to_datatree(random.PRNGKey(2), _model(), numpy_posterior, data, covariates)
    assert tree["posterior"].sizes["chain"] == 1
    assert tree["posterior"].sizes["draw"] == posterior["sigma"].shape[0]


def test_to_datatree_splits_rng_key_for_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-sample predictive and forecast draws receive distinct subkeys.

    Spies on the ``predict_in_sample``/``forecast`` names as looked up inside
    ``convert``; a horizon exercises both draws.
    """
    import numpyro_forecast.convert as convert_mod

    captured: dict[str, Array] = {}
    real_pred = convert_mod.predict_in_sample
    real_forecast = convert_mod.forecast

    def spy_pred(
        rng_key: Array,
        model: ForecastModel,
        posterior: dict[str, Array],
        covariates: Array,
        **kwargs: object,
    ) -> object:
        captured["pred"] = rng_key
        return real_pred(rng_key, model, posterior, covariates, **kwargs)  # ty: ignore[invalid-argument-type]

    def spy_forecast(rng_key: Array, *args: object, **kwargs: object) -> object:
        captured["forecast"] = rng_key
        return real_forecast(rng_key, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(convert_mod, "predict_in_sample", spy_pred)
    monkeypatch.setattr(convert_mod, "forecast", spy_forecast)

    posterior, data, _ = _svi_posterior(num_draws=20)
    n = data.shape[-2]
    parent = random.PRNGKey(2)
    to_datatree(parent, _model(), posterior, data, empty_covariates(n + 3))

    assert not jnp.array_equal(captured["pred"], captured["forecast"])
    assert not jnp.array_equal(captured["pred"], parent)
    assert not jnp.array_equal(captured["forecast"], parent)


def test_to_datatree_deterministic_given_key() -> None:
    """Two calls with the same rng_key produce identical trees (derived split)."""
    posterior, data, covariates = _svi_posterior(num_draws=20)
    a = to_datatree(random.PRNGKey(7), _model(), posterior, data, covariates)
    b = to_datatree(random.PRNGKey(7), _model(), posterior, data, covariates)
    np.testing.assert_array_equal(
        np.asarray(a["posterior_predictive"]["obs"]),
        np.asarray(b["posterior_predictive"]["obs"]),
    )
    np.testing.assert_array_equal(
        np.asarray(a["posterior"]["drift"]),
        np.asarray(b["posterior"]["drift"]),
    )


# --- batch_size (issue #64) ----------------------------------------


def _svi_posterior_with_horizon(
    num_draws: int, horizon: int = 5
) -> tuple[dict[str, Array], Array, Array]:
    posterior, data, _ = _svi_posterior(num_draws=num_draws)
    n = data.shape[-2]
    return posterior, data, empty_covariates(n + horizon)


def test_to_datatree_batch_size_groups_and_shapes() -> None:
    """Chunked predictive sampling produces the same tree layout as the default."""
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    n = data.shape[-2]
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        covariates,
        batch_size=4,
    )
    assert set(tree.children) == {
        "posterior",
        "posterior_predictive",
        "observed_data",
        "constant_data",
        "predictions",
        "predictions_constant_data",
    }
    pp = tree["posterior_predictive"]["obs"]
    assert pp.sizes == {"chain": 1, "draw": 10, "time": n, "obs_dim": 1}
    predictions = tree["predictions"]["obs"]
    assert predictions.sizes == {"chain": 1, "draw": 10, "time": 5, "obs_dim": 1}
    assert bool(np.all(np.isfinite(np.asarray(pp))))
    assert bool(np.all(np.isfinite(np.asarray(predictions))))


def test_to_datatree_batch_size_ge_samples_matches_default() -> None:
    """A batch size at or above the draw count hits the passthrough: identical tree."""
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    default = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates)
    batched = to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, batch_size=64)
    for group in default.children:
        xarray_testing.assert_equal(default[group].dataset, batched[group].dataset)


def test_to_datatree_batch_size_deterministic_given_key() -> None:
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    trees = [
        to_datatree(random.PRNGKey(7), _model(), posterior, data, covariates, batch_size=4)
        for _ in range(2)
    ]
    for group in trees[0].children:
        xarray_testing.assert_equal(trees[0][group].dataset, trees[1][group].dataset)


def test_to_datatree_rejects_non_positive_batch_size() -> None:
    """The batch-size contract is enforced at the public to_datatree surface."""
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, batch_size=0)


@pytest.mark.parametrize("other_device", [None, "host"])
def test_to_datatree_device_matches_cpu(other_device: str | None) -> None:
    """device is a placement knob, never a draws knob: equal trees."""
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    trees = [
        to_datatree(
            random.PRNGKey(7),
            _model(),
            posterior,
            data,
            covariates,
            batch_size=4,
            device=device,
        )
        for device in ("cpu", other_device)
    ]
    for group in trees[0].children:
        xarray_testing.assert_equal(trees[0][group].dataset, trees[1][group].dataset)


@pytest.mark.parametrize("device", ["host", None])
def test_to_datatree_forwards_device(monkeypatch: pytest.MonkeyPatch, device: str | None) -> None:
    """Both predictive calls receive the *resolved* device (default ``"host"``).

    ``to_datatree`` resolves the placement once (``"host"`` becomes the CPU
    backend device) and hands the same value to both drivers.
    """
    import numpyro_forecast.convert as convert_mod

    captured: dict[str, object] = {}
    real_pred = convert_mod.predict_in_sample
    real_forecast = convert_mod.forecast

    def spy_pred(*args: object, **kwargs: object) -> object:
        captured["predict_in_sample"] = kwargs["device"]
        return real_pred(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    def spy_forecast(*args: object, **kwargs: object) -> object:
        captured["forecast"] = kwargs["device"]
        return real_forecast(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(convert_mod, "predict_in_sample", spy_pred)
    monkeypatch.setattr(convert_mod, "forecast", spy_forecast)

    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    if device is None:
        to_datatree(
            random.PRNGKey(2),
            _model(),
            posterior,
            data,
            covariates,
            batch_size=4,
            device=None,
        )
    else:
        # The default must forward "host", so device is deliberately omitted.
        to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, batch_size=4)
    expected = None if device is None else jax.devices("cpu")[0]
    assert captured["predict_in_sample"] == expected
    assert captured["forecast"] == expected


def test_to_datatree_forwards_numpy_sentinel_without_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a CPU backend the default ``"host"`` resolves to ``"numpy"`` for both drivers."""
    import numpyro_forecast.convert as convert_mod

    captured: dict[str, object] = {}
    real_pred = convert_mod.predict_in_sample
    real_forecast = convert_mod.forecast

    def spy_pred(*args: object, **kwargs: object) -> object:
        captured["predict_in_sample"] = kwargs["device"]
        return real_pred(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    def spy_forecast(*args: object, **kwargs: object) -> object:
        captured["forecast"] = kwargs["device"]
        return real_forecast(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(convert_mod, "predict_in_sample", spy_pred)
    monkeypatch.setattr(convert_mod, "forecast", spy_forecast)
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    to_datatree(random.PRNGKey(2), _model(), posterior, data, covariates, batch_size=4)
    assert captured == {"predict_in_sample": "numpy", "forecast": "numpy"}


def test_to_datatree_default_works_without_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the GPU crash: to_datatree must not need the CPU backend.

    ``numpyro.set_platform("cuda")`` restricts ``jax_platforms`` to cuda only,
    so ``jax.devices("cpu")`` raises. The default ``device="host"``
    then takes the backend-free NumPy path (one ``jax.device_get`` per chunk),
    so the export succeeds with no warning.
    """
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tree = to_datatree(random.PRNGKey(7), _model(), posterior, data, covariates, batch_size=4)
    assert "posterior_predictive" in tree.children
    assert "predictions" in tree.children


def test_to_datatree_cpu_without_cpu_backend_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``device="cpu"`` with no CPU backend warns once per export.

    The device is resolved once and the resolved ``"numpy"`` sentinel is handed
    to both predictive drivers, so the unmet request is reported a single time.
    """
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    posterior, data, covariates = _svi_posterior_with_horizon(num_draws=10)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        tree = to_datatree(
            random.PRNGKey(7),
            _model(),
            posterior,
            data,
            covariates,
            batch_size=4,
            device="cpu",
        )
    assert "predictions" in tree.children
    fallback = [r for r in records if "falls back to device='host'" in str(r.message)]
    assert len(fallback) == 1, [str(r.message) for r in records]


def test_to_datatree_batch_size_mcmc_keeps_chain_structure() -> None:
    posterior, data, _covariates = _mcmc_posterior(num_chains=2)
    n = data.shape[-2]
    tree = to_datatree(
        random.PRNGKey(2),
        _model(),
        posterior,
        data,
        empty_covariates(n + 4),
        num_chains=2,
        batch_size=16,
    )
    pp = tree["posterior_predictive"]["obs"]
    assert pp.sizes == {"chain": 2, "draw": 50, "time": n, "obs_dim": 1}
    predictions = tree["predictions"]["obs"]
    assert predictions.sizes == {"chain": 2, "draw": 50, "time": 4, "obs_dim": 1}


def _prediction_draws(
    sample: int = 40, time: int = 12, series: int = 3
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    predictions = rng.normal(1.0, 0.3, size=(sample, time, series))
    x = np.linspace(0.0, 1.0, time)
    labels = [f"s{i}" for i in range(series)]
    return predictions, x, labels


def test_predictions_to_datatree_groups_and_dims() -> None:
    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, labels)
    assert set(tree.children) == {"posterior_predictive", "observed_data", "constant_data"}
    obs = tree["posterior_predictive"]["obs"]
    assert obs.dims == ("chain", "draw", "time", "series")
    assert tree["constant_data"]["t"].dims == ("time", "series")
    np.testing.assert_array_equal(np.asarray(tree["observed_data"].coords["time"]), x)
    assert list(tree["observed_data"].coords["series"].values) == labels
    assert tree.attrs["creation_library"] == "numpyro_forecast"


def test_predictions_to_datatree_sample_axis_becomes_chain_draw() -> None:
    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, labels)
    obs = tree["posterior_predictive"]["obs"]
    assert obs.sizes["chain"] == 1
    assert obs.sizes["draw"] == predictions.shape[0]
    np.testing.assert_allclose(np.asarray(obs)[0], predictions)


def test_predictions_to_datatree_custom_group() -> None:
    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, labels, group="prior_predictive")
    assert set(tree.children) == {"prior_predictive", "observed_data", "constant_data"}
    assert tree["prior_predictive"]["obs"].sizes["draw"] == predictions.shape[0]


def test_predictions_to_datatree_default_observed_is_zeros() -> None:
    # plot_lm requires the observed_data group even with observed_scatter disabled,
    # so the default is a zeros placeholder rather than omitting the group.
    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, labels)
    np.testing.assert_array_equal(
        np.asarray(tree["observed_data"]["obs"]), np.zeros(predictions.shape[1:])
    )


def test_predictions_to_datatree_observed_passthrough() -> None:
    predictions, x, labels = _prediction_draws()
    observed = np.arange(predictions[0].size, dtype=float).reshape(predictions.shape[1:])
    tree = predictions_to_datatree(predictions, x, labels, observed=observed)
    np.testing.assert_allclose(np.asarray(tree["observed_data"]["obs"]), observed)


def test_predictions_to_datatree_non_str_labels() -> None:
    predictions, x, _ = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, [0, 1, 2])
    picked = tree["posterior_predictive"]["obs"].sel(series=2)
    np.testing.assert_allclose(np.asarray(picked)[0], predictions[:, :, 2])


def test_predictions_to_datatree_x_grid_broadcast() -> None:
    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, labels)
    grid = np.asarray(tree["constant_data"]["t"])
    for column in range(len(labels)):
        np.testing.assert_array_equal(grid[:, column], x)


def test_predictions_to_datatree_label_length_mismatch_raises() -> None:
    predictions, x, labels = _prediction_draws()
    with pytest.raises(ValueError, match="series has length"):
        predictions_to_datatree(predictions, x, labels[:-1])


@pytest.mark.parametrize("group", ["observed_data", "constant_data"])
def test_predictions_to_datatree_rejects_reserved_group(group: str) -> None:
    """A reserved group name raises instead of silently overwriting the draws."""
    predictions, x, labels = _prediction_draws()
    with pytest.raises(ValueError, match=rf"group='{group}' is reserved"):
        predictions_to_datatree(predictions, x, labels, group=group)


def test_predictions_to_datatree_accepts_jax_arrays() -> None:
    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(jnp.asarray(predictions), jnp.asarray(x), labels)
    np.testing.assert_allclose(
        np.asarray(tree["posterior_predictive"]["obs"])[0], predictions, rtol=1e-6
    )


def test_predictions_to_datatree_via_dict_to_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """All three groups must be built through arviz_base.dict_to_dataset (normative rule)."""
    calls = {"count": 0}
    real = arviz_base.dict_to_dataset

    def spy(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(arviz_base, "dict_to_dataset", spy)
    predictions, x, labels = _prediction_draws()
    predictions_to_datatree(predictions, x, labels)
    assert calls["count"] == 3


def test_predictions_to_datatree_plot_lm_smoke() -> None:
    """The tree drives az.plot_lm faceting end to end (band artists retrievable)."""
    pytest.importorskip("arviz_plots")
    import matplotlib

    matplotlib.use("Agg")
    import arviz as az

    predictions, x, labels = _prediction_draws()
    tree = predictions_to_datatree(predictions, x, labels)
    pc = az.plot_lm(
        tree,
        y="obs",
        x="t",
        plot_dim="time",
        ci_kind="hdi",
        ci_prob=(0.5, 0.94),
        smooth=False,
        col_wrap=1,
        visuals={"ci_band": {"color": "C0"}, "observed_scatter": False, "pe_line": False},
    )
    band = pc.viz["ci_band"]["t"].sel(series=labels[-1], prob=0.94).item()
    assert band is not None
    assert pc.get_target("t", {"series": labels[0]}) is not None
