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

from numpyro_forecast.functional import MCMCFit, draw_posterior, predict_in_sample
from numpyro_forecast.typing import Array, ForecastModel

_DEFAULT_NUM_PREDICTIVE_SAMPLES = 1_000
"""Default posterior draw count for variational fits when unspecified."""

_SAMPLE_DIMS = ["chain", "draw"]
"""The ArviZ sample dimensions shared by the posterior groups."""


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
) -> "xarray.DataTree":
    r"""Convert a fit into an ArviZ-schema :class:`xarray.DataTree`.

    PRNG: ``rng_key`` is consumed by the in-sample posterior-predictive draws (and,
    for a variational fit, the posterior draws).

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
        In-sample covariates with time at axis ``-2``.
    num_predictive_samples
        Number of posterior draws for a variational fit (ignored for
        :class:`~numpyro_forecast.functional.mcmc.MCMCFit`, which uses its own draws).
        Defaults to ``1_000``.
    coords
        Optional extra coordinates; these take precedence over the generated
        ``time`` coordinate.
    time_coord
        Optional explicit in-sample time coordinate values; defaults to
        ``range(n_time)``.
    posterior_dims
        Optional mapping from a posterior site name to its non-sample dimension
        names, e.g. ``{"drift": ["time"]}``. Sites listed here share the tree-wide
        ``time`` coordinate; unlisted sites keep ArviZ's auto-named dims. This is
        an explicit opt-in on purpose: inferring time-indexed sites from trace
        shapes is fragile (a coincidental ``n_params == n_time`` would misattribute
        the axis).

    Returns
    -------
    xarray.DataTree
        A tree with ``posterior`` (``(chain, draw, ...)``; a single pseudo-chain
        plus ``variational: True`` attrs for SVI/Pathfinder), ``posterior_predictive``
        (in-sample ``obs``), ``observed_data``, and ``constant_data`` groups.

    Notes
    -----
    ``rng_key`` is split internally: one subkey drives the posterior draws (for
    variational fits) and the other the in-sample predictive. The split is a
    deterministic derivation applied for every fit type, so passing the same key
    twice never correlates the two sample sets.
    """
    key_post, key_pred = random.split(rng_key)

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

    n_time = data.shape[-2]
    time = np.asarray(time_coord) if time_coord is not None else np.arange(n_time)
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

    predictive = predict_in_sample(key_pred, model, samples, covariates)
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
        {"covariates": np.asarray(covariates)},
        sample_dims=[],
        dims={"covariates": ["time", "covariate_dim"]},
        coords=merged_arg,
    )

    tree = xarray.DataTree.from_dict(
        {
            "posterior": posterior_ds,
            "posterior_predictive": pp_ds,
            "observed_data": observed_ds,
            "constant_data": constant_ds,
        }
    )
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
) -> "xarray.DataTree":
    """Attach out-of-sample forecast groups to a copy of ``tree``.

    Adds a ``predictions`` group (the forecast ``obs`` draws) and a
    ``predictions_constant_data`` group (the future covariates). The forecast
    ``time`` coordinate continues the in-sample one: integer continuation by
    default, or explicit values via ``time_coord``.

    Parameters
    ----------
    tree
        A tree from :func:`to_datatree` (its ``observed_data`` time coordinate is
        continued).
    forecast_samples
        Forecast draws shaped ``(num_samples, future, obs)`` from
        :func:`~numpyro_forecast.functional.prediction.forecast`.
    covariates_future
        Future covariates shaped ``(future, covariate_dim)``.
    time_coord
        Optional explicit forecast time coordinate; defaults to integer
        continuation of the in-sample time. Required when the in-sample time
        coordinate is non-integer (e.g. datetime64): auto-continuing would have
        to guess the frequency, so explicit values are demanded instead.

    Returns
    -------
    xarray.DataTree
        A new tree with the ``predictions`` and ``predictions_constant_data``
        groups added.

    Raises
    ------
    ValueError
        If ``time_coord`` is given but its length differs from the forecast
        horizon, or if it is omitted while the in-sample time coordinate is
        non-integer.
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
    future_coords = cast("dict[Any, Any]", {"time": future_time})

    predictions_ds = arviz_base.dict_to_dataset(
        {"obs": np.asarray(forecast_samples)[None]},
        sample_dims=_SAMPLE_DIMS,
        dims={"obs": ["time", "obs_dim"]},
        coords=future_coords,
    )
    predictions_constant_ds = arviz_base.dict_to_dataset(
        {"covariates": np.asarray(covariates_future)},
        sample_dims=[],
        dims={"covariates": ["time", "covariate_dim"]},
        coords=future_coords,
    )

    groups: dict[str, Any] = {name: node.dataset for name, node in tree.children.items()}
    groups["predictions"] = predictions_ds
    groups["predictions_constant_data"] = predictions_constant_ds
    new_tree = xarray.DataTree.from_dict(groups)
    new_tree.attrs.update(dict(tree.attrs))
    return new_tree
