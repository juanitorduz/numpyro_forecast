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
from typing import Any, cast

import arviz_base
import jax
import numpy as np
import xarray
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Float, Num

from numpyro_forecast.exceptions import CovariateDimsError
from numpyro_forecast.functional import forecast, predict_in_sample
from numpyro_forecast.functional._offload import _resolve_device
from numpyro_forecast.functional._validation import _require_covariates_cover_data
from numpyro_forecast.typing import Array, ForecastModel

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
    CovariateDimsError
        If the number of names does not match ``covariates.ndim``.
    """
    dims = _DEFAULT_COVARIATE_DIMS if covariate_dims is None else covariate_dims
    ndim = np.asarray(covariates).ndim
    if len(dims) != ndim:
        msg = (
            f"covariate_dims has {len(dims)} names {list(dims)} but covariates "
            f"has {ndim} dimensions; pass one name per axis"
        )
        raise CovariateDimsError(msg)
    return list(dims)


def _reconcile_tree_covariate_dims(
    tree: "xarray.DataTree",
    covariates_future: Array,
    covariate_dims: Sequence[str] | None,
) -> Sequence[str] | None:
    """Inherit or validate ``covariate_dims`` against the tree's stored covariates.

    :func:`add_forecast_groups` must name the future covariates' axes exactly
    like the in-sample ones already stored on ``constant_data["covariates"]``,
    or the two groups silently disagree on axis names. When the tree carries
    stored covariates, an omitted ``covariate_dims`` inherits their names and an
    explicit one is cross-checked; trees without stored covariates pass through
    unchanged (the downstream default applies).

    Parameters
    ----------
    tree
        The tree being extended.
    covariates_future
        The future covariates whose axes the names must cover.
    covariate_dims
        The user-supplied dimension names, or ``None`` to inherit.

    Returns
    -------
    Sequence[str] | None
        The reconciled dimension names (``None`` only when the tree has no
        stored covariates and none were supplied).

    Raises
    ------
    CovariateDimsError
        If inherited names do not cover every ``covariates_future`` axis, or if
        explicit names disagree with the stored ones.
    """
    constant_node = tree.children.get("constant_data")
    if constant_node is None or "covariates" not in constant_node.dataset:
        return covariate_dims
    stored_dims = [str(dim) for dim in constant_node.dataset["covariates"].dims]
    if covariate_dims is None:
        ndim = np.asarray(covariates_future).ndim
        if len(stored_dims) != ndim:
            msg = (
                f"covariates_future has {ndim} dimensions but the tree's "
                f"constant_data covariates carry {len(stored_dims)} axis names "
                f"{stored_dims}; pass covariates_future with the same layout as "
                "the in-sample covariates (time at axis -2) or explicit "
                "covariate_dims"
            )
            raise CovariateDimsError(msg)
        return stored_dims
    if list(covariate_dims) != stored_dims:
        msg = (
            f"covariate_dims {list(covariate_dims)} disagree with the names "
            f"{stored_dims} already stored on the tree's constant_data covariates "
            "(set by the earlier to_datatree(covariate_dims=...) call); omit "
            "covariate_dims to inherit them, or pass matching names"
        )
        raise CovariateDimsError(msg)
    return covariate_dims


def _reshape_chains(array: "Array | np.ndarray", num_chains: int) -> "np.ndarray":
    """Split a flattened leading sample axis into ``(num_chains, draw, ...)``.

    Applied to every posterior leaf and to the in-sample/forecast predictive
    draws alike: both are drawn with a sample count equal to the posterior's,
    so the same ``num_chains`` reshape recovers a consistent ``(chain, draw,
    ...)`` layout across all groups.

    Parameters
    ----------
    array
        Draws with a single flattened sample axis leading, of length
        ``num_chains * draw``.
    num_chains
        Number of chains to split the leading axis into (``1`` for a fit with
        no chain structure, e.g. SVI or Pathfinder).

    Returns
    -------
    numpy.ndarray
        ``array`` reshaped to ``(num_chains, draw, *array.shape[1:])``.

    Raises
    ------
    ValueError
        If the leading axis length is not evenly divisible by ``num_chains``.

    Notes
    -----
    NumPyro's flattened ``mcmc.get_samples()`` (the default,
    ``group_by_chain=False``) concatenates chains in a fixed order, so this
    plain reshape recovers the exact per-chain layout. ``to_datatree`` (and
    hence this helper) only accepts that flattened, chain-major layout;
    ``mcmc.get_samples(group_by_chain=True)`` output, whose leaves already
    carry explicit ``(chain, draw, ...)`` axes, is not a valid input here. If
    the flattened ordering guarantee is a concern, draw with
    ``group_by_chain=True`` upstream only to disambiguate chain order for
    yourself, then flatten back before passing samples to ``to_datatree``.
    """
    reshaped = np.asarray(array)
    n = reshaped.shape[0]
    if n % num_chains != 0:
        msg = f"posterior sample count {n} is not evenly divisible by num_chains {num_chains}"
        raise ValueError(msg)
    return reshaped.reshape(num_chains, n // num_chains, *reshaped.shape[1:])


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
    model: ForecastModel,
    posterior: Mapping[str, ArrayLike],
    data: Array,
    covariates: Array,
    *,
    num_chains: int = 1,
    predictive_batch_size: int | None = None,
    predictive_device: jax.Device | str | None = "host",
    coords: Mapping[str, Sequence[Any]] | None = None,
    time_coord: Sequence[Any] | None = None,
    posterior_dims: Mapping[str, Sequence[str]] | None = None,
    covariate_dims: Sequence[str] | None = None,
) -> "xarray.DataTree":
    r"""Convert an already-drawn posterior into an ArviZ-schema :class:`xarray.DataTree`.

    Posterior-first: callers draw their own posterior (``mcmc.get_samples()``
    for MCMC, :func:`~numpyro_forecast.functional.posterior.draw_posterior` for a
    variational fit) and pass it in; ``to_datatree`` never draws a posterior of
    its own. ``rng_key`` is consumed only by the in-sample posterior-predictive
    draws and, when a forecast horizon is present, the forecast draws.

    Parameters
    ----------
    rng_key
        PRNG key for the in-sample predictive draws and, when a horizon is
        present, the forecast draws.
    model
        The forecasting model that produced ``posterior``.
    posterior
        Posterior samples of the latent sites, with a single flattened sample
        axis leading (NumPyro's ``mcmc.get_samples()`` order, or the output of
        :func:`~numpyro_forecast.functional.posterior.draw_posterior`).
        Host-committed leaves (the output of ``draw_posterior(...,
        device="host")``) and NumPy leaves are accepted directly.
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2``. When ``covariates`` extends beyond
        ``data`` along the time axis (the package-wide shape convention for a
        forecast horizon), the trailing rows are treated as future covariates:
        the returned tree additionally carries ``predictions`` (forecast ``obs``
        draws from :func:`~numpyro_forecast.functional.prediction.forecast`) and
        ``predictions_constant_data`` groups.
    num_chains
        Number of chains to split ``posterior``'s flattened sample axis into
        (and, identically, the in-sample/forecast predictive draws, which are
        drawn with the same sample count). Defaults to ``1`` (a single
        pseudo-chain, correct for a posterior with no chain structure, e.g.
        SVI or Pathfinder draws). For an MCMC posterior, pass the
        ``num_chains`` the sampler was run with; see :func:`_reshape_chains`
        for the reshape contract and its divisibility requirement.
    predictive_batch_size
        Optional chunk size that bounds how many draws touch the accelerator
        at once, across both the in-sample and forecast predictive sampling.
        When set, sampling runs in chunks of this many draws, each chunk moved
        to ``predictive_device`` before the next is drawn. The per-chunk
        accelerator footprint is a handful of ``(batch_size, time, series)``
        buffers, so it scales linearly with this value times the panel width:
        on wide panels lower it until a chunk fits. The batch size must be
        strictly below the draw count for that bound to hold: at or above it,
        sampling falls back to the single-shot path and the full array is
        materialized on the default device before the single transfer.
        Chunking changes the PRNG stream layout of the predictive draws, so
        results are reproducible per ``(rng_key, predictive_batch_size)``.
        ``None`` (default) samples everything in one shot (the results are
        still moved to ``predictive_device``).
    predictive_device
        Where the predictive draws are moved as they are sampled, forwarded to
        the ``device`` argument of
        :func:`~numpyro_forecast.functional.prediction.predict_in_sample` and
        :func:`~numpyro_forecast.functional.prediction.forecast`. It is
        resolved once and the same placement is handed to both. The default
        ``"host"`` keeps the predictive draws in pageable host memory, which
        is what bounds accelerator memory when ``predictive_batch_size`` is
        set: with the JAX CPU backend initialized every chunk is committed to
        ``jax.devices("cpu")[0]`` (jax Arrays the tree views as NumPy without a
        copy); without it (for example after ``numpyro.set_platform("cuda")``,
        or a ``JAX_PLATFORMS`` preset) every chunk is copied with
        :func:`jax.device_get` (NumPy arrays), so no CPU backend is needed and
        nothing is pinned. ``"pinned_host"`` uses the accelerator's pinned
        host memory kind instead (capped at 64 GB by default on CUDA,
        ``XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB``). A :class:`jax.Device` or
        platform name like ``"cpu"`` commits the draws to that device
        (``"cpu"`` warns and takes the NumPy path when the CPU backend is
        missing); pass ``None`` to keep the draws on the default device
        (chunked compute without per-chunk host transfers, for when the draws
        fit on the accelerator and transfers would dominate runtime).
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
        A tree with ``posterior`` (``(chain, draw, ...)``, split per
        ``num_chains``), ``posterior_predictive`` (in-sample ``obs``),
        ``observed_data``, and ``constant_data`` groups. When ``covariates``
        extends beyond ``data``, also ``predictions`` and
        ``predictions_constant_data`` groups (sharing the same ``num_chains``
        split).

    Raises
    ------
    ValueError
        If ``covariates`` is shorter than ``data`` along the time axis, if
        ``time_coord`` is given but its length does not match the in-sample
        window plus the forecast horizon, or if ``posterior``'s sample count is
        not evenly divisible by ``num_chains``.
    CovariateDimsError
        If ``covariate_dims`` does not name every ``covariates`` axis.
    RuntimeError
        If ``predictive_device="pinned_host"`` is requested on a device that
        exposes no host memory kind (see
        :func:`~numpyro_forecast.functional._offload._host_memory_kind`).

    Warns
    -----
    UserWarning
        If ``predictive_device="cpu"`` is requested and the JAX CPU backend is
        not initialized, so the predictive draws take the NumPy path of
        ``"host"`` instead (once per call).

    Notes
    -----
    ``to_datatree`` no longer accepts a fit object or draws a posterior itself
    (no ``num_predictive_samples``, no internal
    :func:`~numpyro_forecast.functional.posterior.draw_posterior` call): callers
    draw the posterior first and pass it in. The ``variational``/``is_mcmc``
    attrs previously stamped on the ``posterior`` group are gone too, since a
    fit type is no longer knowable from a plain posterior dict; use
    ``num_chains`` (``1`` vs. ``> 1``) to tell the two apart if needed.
    When a forecast horizon is present, ``rng_key`` is split internally into a
    predictive subkey and a forecast subkey, so passing the same key twice
    never correlates the two sample sets. When there is no horizon, ``rng_key``
    is used unsplit for the in-sample predictive draw. ``predictive_batch_size``
    is the built-in route to memory-bounded predictive sampling; for fully
    manual control over the forecast draws, build the in-sample tree with
    matching-length covariates and attach the horizon with
    :func:`add_forecast_groups`.
    """
    _require_covariates_cover_data(data, covariates)
    cov_dims = _resolve_covariate_dims(covariates, covariate_dims)
    n_time = data.shape[-2]
    horizon = covariates.shape[-2] - n_time

    if horizon > 0:
        key_pred, key_forecast = random.split(rng_key, 2)
    else:
        key_pred, key_forecast = rng_key, None

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

    posterior_reshaped = {
        name: _reshape_chains(cast("Array | np.ndarray", value), num_chains)
        for name, value in posterior.items()
    }
    posterior_ds = arviz_base.dict_to_dataset(
        posterior_reshaped,
        sample_dims=_SAMPLE_DIMS,
        dims=cast("dict[Any, Any] | None", posterior_dims),
        coords=merged_arg,
    )

    covariates_insample = covariates[..., :n_time, :]
    # Resolve once so the two predictive drivers share one placement (and an
    # unmet explicit "cpu" request warns once per export).
    resolved_device = _resolve_device(predictive_device)
    predictive = predict_in_sample(
        key_pred,
        model,
        posterior,
        covariates_insample,
        batch_size=predictive_batch_size,
        device=resolved_device,
    )
    pp_ds = arviz_base.dict_to_dataset(
        {"obs": _reshape_chains(predictive, num_chains)},
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
        forecast_samples = forecast(
            key_forecast,
            model,
            posterior,
            data,
            covariates,
            batch_size=predictive_batch_size,
            device=resolved_device,
        )
        predictions_ds, predictions_constant_ds = _forecast_group_datasets(
            _reshape_chains(forecast_samples, num_chains),
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
    forecast_samples: Num[ArrayLike, " sample future obs"],
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
        Optional dimension names for ``covariates_future``, one per axis. When
        omitted, the names are inherited from the tree's
        ``constant_data["covariates"]`` variable (falling back to
        ``("time", "covariate_dim")`` if the tree carries no stored
        covariates), so the forecast covariates always share the in-sample
        axis names. When given explicitly, the names must match the stored
        ones. See :func:`to_datatree`.

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
    CovariateDimsError
        If the resolved ``covariate_dims`` (explicit or inherited) do not name
        every ``covariates_future`` axis, or if explicit names disagree with
        the dimension names already stored on the tree's
        ``constant_data["covariates"]``.
    """
    covariate_dims = _reconcile_tree_covariate_dims(tree, covariates_future, covariate_dims)
    in_time = np.asarray(tree["observed_data"].coords["time"].values)
    forecast_samples_arr = np.asarray(forecast_samples)
    future_len = forecast_samples_arr.shape[-2]
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
        forecast_samples_arr[None],
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
    predictions: Float[ArrayLike, " sample time series"],
    x: Num[ArrayLike, " time"],
    series: Sequence[Any],
    *,
    group: str = "posterior_predictive",
    observed: Float[ArrayLike, " time series"] | None = None,
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
