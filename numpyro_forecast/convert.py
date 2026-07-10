"""Convert fits into ArviZ-schema :class:`xarray.DataTree` objects.

ArviZ (>= 1.0, the DataTree line) is a hard dependency of this package, so
``arviz_base`` and ``xarray`` are imported at module top like any other
dependency.

The single normative construction rule (enforced by
``tests/test_convert.py::test_all_groups_via_dict_to_dataset``) is that **every**
group is built with :func:`arviz_base.dict_to_dataset`: it owns dim naming,
``sample_dims`` handling, and schema evolution, so hand-rolling
:class:`xarray.Dataset` objects would silently drift from the ArviZ schema. The
groups are then assembled with :meth:`xarray.DataTree.from_dict`.
"""

from collections.abc import Mapping, Sequence
from functools import singledispatch
from typing import Any, cast

import arviz_base
import numpy as np
import xarray
from jax import random
from jaxtyping import Float, Num

from numpyro_forecast.functional import MCMCFit, draw_posterior, forecast, predict_in_sample
from numpyro_forecast.functional._validation import _require_covariates_cover_data
from numpyro_forecast.typing import Array, ForecastModel

_DEFAULT_NUM_PREDICTIVE_SAMPLES = 1_000
"""Default posterior draw count for variational fits when unspecified."""

_SAMPLE_DIMS = ["chain", "draw"]
"""The ArviZ sample dimensions shared by the posterior groups."""

_DEFAULT_COVARIATE_DIMS = ("time", "covariate_dim")
"""Default dimension names for the stored covariates (the 2-D layout)."""


def _resolve_covariate_dims(covariates: Array, covariate_dims: Sequence[str] | None) -> list[str]:
    """Validate and normalize the covariate dimension names.

    Parameters
    ----------
    covariates
        The covariates array the names must match.
    covariate_dims
        One dimension name per covariates axis, or ``None`` for the default
        2-D ``("time", "covariate_dim")`` layout.

    Returns
    -------
    list[str]
        The dimension names to hand to :func:`arviz_base.dict_to_dataset`.

    Raises
    ------
    ValueError
        If the number of names does not match ``covariates.ndim``.
    """
    dims = _DEFAULT_COVARIATE_DIMS if covariate_dims is None else covariate_dims
    ndim = np.asarray(covariates).ndim
    if len(dims) != ndim:
        msg = (
            f"covariate_dims has {len(dims)} names {list(dims)} but covariates "
            f"has {ndim} dimensions; pass one name per axis"
        )
        raise ValueError(msg)
    return list(dims)


@singledispatch
def _posterior_reshape(fit: object, samples: dict[str, Array]) -> dict[str, "np.ndarray"]:
    """Reshape sample-leading draws to ``(chain, draw, ...)`` per fit type.

    The default adds a single pseudo-chain (``leaf[None]``), which is correct for
    the variational fits (SVI, Pathfinder) that have no chain structure. The
    :class:`~numpyro_forecast.functional.mcmc.MCMCFit` override splits the flattened
    draws back into ``fit.num_chains`` chains.

    Parameters
    ----------
    fit
        The fit whose chain structure determines the reshape.
    samples
        Posterior draws with the sample axis leading.

    Returns
    -------
    dict[str, numpy.ndarray]
        Draws reshaped to ``(chain, draw, ...)``.
    """
    return {name: np.asarray(value)[None] for name, value in samples.items()}


@_posterior_reshape.register
def _(fit: MCMCFit, samples: dict[str, Array]) -> dict[str, "np.ndarray"]:
    reshaped: dict[str, np.ndarray] = {}
    for name, value in samples.items():
        array = np.asarray(value)
        reshaped[name] = array.reshape(fit.num_chains, -1, *array.shape[1:])
    return reshaped


def _reshape_predictive(fit: object, predictive: Array) -> "np.ndarray":
    """Apply the fit's chain reshape to a single predictive array."""
    return _posterior_reshape(fit, {"obs": predictive})["obs"]


def _merge_coords(
    time: "np.ndarray", coords: Mapping[str, Sequence[Any]] | None
) -> dict[str, Any]:
    """Merge coordinates with precedence user ``coords`` > generated ``time``."""
    merged: dict[str, Any] = {"time": time}
    if coords is not None:
        merged.update(coords)
    return merged


def _forecast_group_datasets(
    forecast_draws: "np.ndarray",
    covariates_future: Array,
    future_time: "np.ndarray",
    coords: Mapping[str, Sequence[Any]] | None = None,
    covariate_dims: Sequence[str] | None = None,
) -> tuple["xarray.Dataset", "xarray.Dataset"]:
    """Build the ``predictions`` and ``predictions_constant_data`` datasets.

    ``forecast_draws`` must already carry the ``(chain, draw, future, obs)``
    layout; callers own the chain reshape (:func:`to_datatree` applies the fit's
    real chain structure, :func:`add_forecast_groups` a single pseudo-chain).

    ``coords`` are user coordinates to share with the forecast groups, but with
    the precedence inverted relative to :func:`_merge_coords`: ``future_time``
    always wins over a user ``time`` entry, because that entry covers the
    in-sample window (``time_coord`` is the sanctioned route for explicit
    forecast time values). ``covariate_dims`` names the axes of
    ``covariates_future`` (default ``("time", "covariate_dim")``).
    """
    dims = _resolve_covariate_dims(covariates_future, covariate_dims)
    merged_future: dict[str, Any] = dict(coords) if coords is not None else {}
    merged_future["time"] = future_time
    future_coords = cast("dict[Any, Any]", merged_future)
    predictions_ds = arviz_base.dict_to_dataset(
        {"obs": forecast_draws},
        sample_dims=_SAMPLE_DIMS,
        dims={"obs": ["time", "obs_dim"]},
        coords=future_coords,
    )
    predictions_constant_ds = arviz_base.dict_to_dataset(
        {"covariates": np.asarray(covariates_future)},
        sample_dims=[],
        dims={"covariates": dims},
        coords=future_coords,
    )
    return predictions_ds, predictions_constant_ds


def to_datatree(
    rng_key: Array,
    fit: object,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    num_predictive_samples: int | None = None,
    coords: Mapping[str, Sequence[Any]] | None = None,
    time_coord: Sequence[Any] | None = None,
    posterior_dims: Mapping[str, Sequence[str]] | None = None,
    covariate_dims: Sequence[str] | None = None,
) -> "xarray.DataTree":
    r"""Convert a fit into an ArviZ-schema :class:`xarray.DataTree`.

    PRNG: ``rng_key`` is consumed by the in-sample posterior-predictive draws (for
    a variational fit, also the posterior draws) and, when a forecast horizon is
    present, the forecast draws.

    Parameters
    ----------
    rng_key
        PRNG key for the predictive (and variational posterior) draws.
    fit
        A fit from :func:`~numpyro_forecast.functional.mcmc.fit_mcmc`,
        :func:`~numpyro_forecast.functional.svi.fit_svi`, or
        :func:`~numpyro_forecast.contrib.blackjax.fit_pathfinder`.
    model
        The forecasting model that produced ``fit``.
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2``. When ``covariates`` extends beyond
        ``data`` along the time axis (the package-wide shape convention for a
        forecast horizon), the trailing rows are treated as future covariates:
        the returned tree additionally carries ``predictions`` (forecast ``obs``
        draws from :func:`~numpyro_forecast.functional.prediction.forecast`) and
        ``predictions_constant_data`` groups.
    num_predictive_samples
        Number of posterior draws for a variational fit (ignored for
        :class:`~numpyro_forecast.functional.mcmc.MCMCFit`, which uses its own draws).
        The same draws drive the in-sample predictive and the forecast. Defaults
        to ``1_000``.
    coords
        Optional extra coordinates; these take precedence over the generated
        ``time`` coordinate. They also propagate to the forecast groups, where
        the generated forecast ``time`` takes precedence instead (a user
        ``time`` entry covers the in-sample window; use ``time_coord`` for
        explicit forecast time values).
    time_coord
        Optional explicit time coordinate values. Without a forecast horizon it
        covers the in-sample window (defaults to ``range(n_time)``); with a
        horizon it must cover the full ``covariates`` length and is split into
        the in-sample and forecast time coordinates (the default is the integer
        continuation).
    posterior_dims
        Optional mapping from a posterior site name to its non-sample dimension
        names, e.g. ``{"drift": ["time"]}``. Sites listed here share the tree-wide
        ``time`` coordinate; unlisted sites keep ArviZ's auto-named dims. This is
        an explicit opt-in on purpose: inferring time-indexed sites from trace
        shapes is fragile (a coincidental ``n_params == n_time`` would misattribute
        the axis).
    covariate_dims
        Optional dimension names for the stored covariates, one per axis;
        defaults to the 2-D ``("time", "covariate_dim")`` layout. Use this when
        ``covariates`` carries extra batch axes, e.g. a panel tensor shaped
        ``(channel, time, series)`` with ``covariate_dims=["channel", "time",
        "series"]``. The time axis is always ``-2`` (the package-wide
        convention), so its entry should be named ``"time"`` to share the
        tree-wide time coordinate.

    Returns
    -------
    xarray.DataTree
        A tree with ``posterior`` (``(chain, draw, ...)``; a single pseudo-chain
        plus ``variational: True`` attrs for SVI/Pathfinder), ``posterior_predictive``
        (in-sample ``obs``), ``observed_data``, and ``constant_data`` groups. When
        ``covariates`` extends beyond ``data``, also ``predictions`` and
        ``predictions_constant_data`` groups (the forecast keeps an MCMC fit's real
        chain structure).

    Raises
    ------
    ValueError
        If ``covariates`` is shorter than ``data`` along the time axis, if
        ``time_coord`` is given but its length does not match the in-sample
        window plus the forecast horizon, or if ``covariate_dims`` does not
        name every ``covariates`` axis.

    Notes
    -----
    ``rng_key`` is split internally: one subkey drives the posterior draws (for
    variational fits), one the in-sample predictive, and, when a horizon is
    present, a third the forecast. The split is a deterministic derivation
    applied for every fit type, so passing the same key twice never correlates
    the sample sets. For step-by-step control over the forecast draws (e.g. a
    custom ``batch_size``), build the in-sample tree with matching-length
    covariates and attach the horizon with :func:`add_forecast_groups`.
    """
    _require_covariates_cover_data(data, covariates)
    cov_dims = _resolve_covariate_dims(covariates, covariate_dims)
    n_time = data.shape[-2]
    horizon = covariates.shape[-2] - n_time

    if horizon > 0:
        key_post, key_pred, key_forecast = random.split(rng_key, 3)
    else:
        key_post, key_pred = random.split(rng_key)
        key_forecast = None

    is_mcmc = isinstance(fit, MCMCFit)
    if is_mcmc:
        samples: dict[str, Array] = dict(fit.samples)  # type: ignore[attr-defined]
    else:
        num = (
            _DEFAULT_NUM_PREDICTIVE_SAMPLES
            if num_predictive_samples is None
            else num_predictive_samples
        )
        samples = draw_posterior(key_post, fit, num)

    if time_coord is not None:
        time_values = np.asarray(time_coord)
        expected = n_time + horizon
        if time_values.shape[0] != expected:
            msg = (
                f"time_coord has length {time_values.shape[0]} but must cover "
                f"{expected} steps ({n_time} in-sample plus {horizon} forecast)"
            )
            raise ValueError(msg)
        time = time_values[:n_time]
        future_time = time_values[n_time:]
    else:
        time = np.arange(n_time)
        future_time = np.arange(n_time, n_time + horizon)
    merged_coords = _merge_coords(time, coords)

    merged_arg = cast("dict[Any, Any]", merged_coords)

    posterior = _posterior_reshape(fit, samples)
    posterior_attrs = None if is_mcmc else {"variational": True}
    posterior_ds = arviz_base.dict_to_dataset(
        posterior,
        sample_dims=_SAMPLE_DIMS,
        dims=cast("dict[Any, Any] | None", posterior_dims),
        coords=merged_arg,
        attrs=posterior_attrs,
    )

    covariates_insample = covariates[..., :n_time, :]
    predictive = predict_in_sample(key_pred, model, samples, covariates_insample)
    pp_ds = arviz_base.dict_to_dataset(
        {"obs": _reshape_predictive(fit, predictive)},
        sample_dims=_SAMPLE_DIMS,
        dims={"obs": ["time", "obs_dim"]},
        coords=merged_arg,
    )

    observed_ds = arviz_base.dict_to_dataset(
        {"obs": np.asarray(data)},
        sample_dims=[],
        dims={"obs": ["time", "obs_dim"]},
        coords=merged_arg,
    )
    constant_ds = arviz_base.dict_to_dataset(
        {"covariates": np.asarray(covariates_insample)},
        sample_dims=[],
        dims={"covariates": cov_dims},
        coords=merged_arg,
    )

    groups: dict[str, Any] = {
        "posterior": posterior_ds,
        "posterior_predictive": pp_ds,
        "observed_data": observed_ds,
        "constant_data": constant_ds,
    }
    if key_forecast is not None:
        forecast_samples = forecast(key_forecast, model, samples, data, covariates)
        predictions_ds, predictions_constant_ds = _forecast_group_datasets(
            _reshape_predictive(fit, forecast_samples),
            covariates[..., n_time:, :],
            future_time,
            coords=coords,
            covariate_dims=cov_dims,
        )
        groups["predictions"] = predictions_ds
        groups["predictions_constant_data"] = predictions_constant_ds

    tree = xarray.DataTree.from_dict(groups)
    tree.attrs.update(
        {
            "inference_library": "numpyro",
            "creation_library": "numpyro_forecast",
            "sample_dims": _SAMPLE_DIMS,
        }
    )
    return tree


def add_forecast_groups(
    tree: "xarray.DataTree",
    forecast_samples: Array,
    covariates_future: Array,
    *,
    time_coord: Sequence[Any] | None = None,
    covariate_dims: Sequence[str] | None = None,
) -> "xarray.DataTree":
    """Attach out-of-sample forecast groups to a copy of ``tree``.

    Adds a ``predictions`` group (the forecast ``obs`` draws) and a
    ``predictions_constant_data`` group (the future covariates). The forecast
    ``time`` coordinate continues the in-sample one: integer continuation by
    default, or explicit values via ``time_coord``. This is the step-by-step
    route for draws you produced yourself; :func:`to_datatree` attaches the same
    groups automatically when its ``covariates`` extend beyond ``data``.

    Parameters
    ----------
    tree
        A tree from :func:`to_datatree` (its ``observed_data`` time coordinate is
        continued).
    forecast_samples
        Forecast draws shaped ``(num_samples, future, obs)`` from
        :func:`~numpyro_forecast.functional.prediction.forecast`.
    covariates_future
        Future covariates shaped ``(future, covariate_dim)``, or any layout
        with time at axis ``-2`` when ``covariate_dims`` names the axes.
    time_coord
        Optional explicit forecast time coordinate; defaults to integer
        continuation of the in-sample time. Required when the in-sample time
        coordinate is non-integer (e.g. datetime64): auto-continuing would have
        to guess the frequency, so explicit values are demanded instead.
    covariate_dims
        Optional dimension names for ``covariates_future``, one per axis;
        defaults to ``("time", "covariate_dim")``. See :func:`to_datatree`.

    Returns
    -------
    xarray.DataTree
        A new tree with the ``predictions`` and ``predictions_constant_data``
        groups added.

    Raises
    ------
    ValueError
        If ``time_coord`` is given but its length differs from the forecast
        horizon, if it is omitted while the in-sample time coordinate is
        non-integer, or if ``covariate_dims`` does not name every
        ``covariates_future`` axis.
    """
    in_time = np.asarray(tree["observed_data"].coords["time"].values)
    future_len = forecast_samples.shape[-2]
    if time_coord is not None:
        future_time = np.asarray(time_coord)
        if future_time.shape[0] != future_len:
            msg = (
                f"time_coord has length {future_time.shape[0]} but "
                f"forecast_samples spans {future_len} future steps"
            )
            raise ValueError(msg)
    elif np.issubdtype(in_time.dtype, np.integer):
        start = int(in_time[-1]) + 1
        future_time = np.arange(start, start + future_len)
    else:
        msg = (
            f"the in-sample time coordinate is non-integer (dtype {in_time.dtype}); "
            "pass explicit time_coord for the forecast horizon"
        )
        raise ValueError(msg)
    predictions_ds, predictions_constant_ds = _forecast_group_datasets(
        np.asarray(forecast_samples)[None],
        covariates_future,
        future_time,
        covariate_dims=covariate_dims,
    )

    groups: dict[str, Any] = {name: node.dataset for name, node in tree.children.items()}
    groups["predictions"] = predictions_ds
    groups["predictions_constant_data"] = predictions_constant_ds
    new_tree = xarray.DataTree.from_dict(groups)
    new_tree.attrs.update(dict(tree.attrs))
    return new_tree


def predictions_to_datatree(
    predictions: Float[np.ndarray | Array, " sample time series"],
    x: Num[np.ndarray | Array, " time"],
    series: Sequence[Any],
    *,
    group: str = "posterior_predictive",
    observed: Float[np.ndarray | Array, " time series"] | None = None,
) -> "xarray.DataTree":
    """Pack prediction draws into a DataTree laid out for per-series ``plot_lm`` faceting.

    The array-level counterpart of :func:`to_datatree`: instead of a fit, it takes
    prediction draws from **any** predictive group (prior predictive, posterior
    predictive, or forecasts), possibly already transformed (rescaled to original
    units, clipped at zero, subset to a few series). The draws get a single
    pseudo-chain, and ``constant_data`` carries the independent variable ``"t"``
    broadcast to ``(time, series)`` so that
    ``arviz.plot_lm(tree, y="obs", x="t", plot_dim="time", ...)`` facets one panel
    per series; band artists are then reachable via ``pc.viz["ci_band"]["t"]`` and
    axes via ``pc.get_target("t", {"series": label})``.

    ``plot_lm`` requires an ``observed_data`` group even when the observation
    scatter is disabled, so when ``observed`` is ``None`` a zeros placeholder is
    stored; it is never drawn under ``visuals={"observed_scatter": False}``.

    Parameters
    ----------
    predictions
        Prediction draws with the sample axis first, shape ``(sample, time, series)``.
    x
        Independent-variable values, shape ``(time,)``. Must be numeric:
        ``plot_lm`` cannot draw ``datetime64`` values (it concatenates ``x`` with
        the float predictions internally), so pass
        :func:`matplotlib.dates.date2num` floats and re-format the tick labels
        with :class:`matplotlib.dates.ConciseDateFormatter`.
    series
        One label per series, defining the ``series`` coordinate.
    group
        Predictive group to store the draws under (e.g. ``"prior_predictive"``,
        ``"posterior_predictive"``, ``"predictions"``).
    observed
        Optional observations, shape ``(time, series)``, stored in
        ``observed_data``; when ``None`` a zeros placeholder is stored instead.

    Returns
    -------
    xarray.DataTree
        A tree with the ``group``, ``observed_data``, and ``constant_data``
        groups; ``obs`` has dims ``(chain, draw, time, series)`` and ``t`` has
        dims ``(time, series)``.

    Raises
    ------
    ValueError
        If ``series`` does not have one label per series in ``predictions``.
    """
    preds = np.asarray(predictions)[None]
    if len(series) != preds.shape[-1]:
        msg = f"series has length {len(series)} but predictions carry {preds.shape[-1]} series"
        raise ValueError(msg)
    x_values = np.asarray(x)
    x_grid = np.broadcast_to(x_values[:, None], preds.shape[2:])
    coords = cast("dict[Any, Any]", {"time": x_values, "series": list(series)})
    predictive_ds = arviz_base.dict_to_dataset(
        {"obs": preds},
        sample_dims=_SAMPLE_DIMS,
        dims={"obs": ["time", "series"]},
        coords=coords,
    )
    observed_values = np.zeros(preds.shape[2:]) if observed is None else np.asarray(observed)
    observed_ds = arviz_base.dict_to_dataset(
        {"obs": observed_values},
        sample_dims=[],
        dims={"obs": ["time", "series"]},
        coords=coords,
    )
    constant_ds = arviz_base.dict_to_dataset(
        {"t": x_grid},
        sample_dims=[],
        dims={"t": ["time", "series"]},
        coords=coords,
    )
    tree = xarray.DataTree.from_dict(
        {group: predictive_ds, "observed_data": observed_ds, "constant_data": constant_ds}
    )
    tree.attrs.update({"creation_library": "numpyro_forecast", "sample_dims": _SAMPLE_DIMS})
    return tree
