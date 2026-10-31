"""Convert fits into ArviZ-schema `xarray.DataTree` objects.

ArviZ (>= 1.0, the DataTree line) is a hard dependency of this package, so
``arviz_base`` and ``xarray`` are imported at module top like any other
dependency.

The single normative construction rule (enforced by
``tests/test_convert.py::test_all_groups_via_dict_to_dataset``) is that **every**
group is built with `arviz_base.dict_to_dataset()`: it owns dim naming,
``sample_dims`` handling, and schema evolution, so hand-rolling
`xarray.Dataset` objects would silently drift from the ArviZ schema. The
groups are then assembled with `xarray.DataTree.from_dict()`.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

import arviz_base
import jax
import numpy as np
import xarray
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Float, Num

from numpyro_forecast.exceptions import CovariateDimsError
from numpyro_forecast.offload import resolve_device
from numpyro_forecast.predictive import forecast, predict_in_sample
from numpyro_forecast.typing import Array, ForecastModel
from numpyro_forecast.validation import (
    drop_obs_site,
    require_covariates_cover_data,
    require_no_horizon_sites,
    require_sample_axis,
)

_SAMPLE_DIMS = ["chain", "draw"]
"""The ArviZ sample dimensions shared by the posterior groups."""

_DEFAULT_COVARIATE_DIMS = ("time", "covariate_dim")
"""Default dimension names for the stored covariates (the 2-D layout)."""

_DEFAULT_OBS_DIMS = ("time", "obs_dim")
"""Default names of the trailing ``(time, obs)`` axes of every stored ``obs`` variable."""

_RESERVED_PREDICTION_GROUPS = frozenset({"observed_data", "constant_data"})
"""The groups `predictions_to_datatree()` always writes, so ``group`` cannot name them."""


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
        The dimension names to hand to `arviz_base.dict_to_dataset()`.

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


def _resolve_obs_dims(ndim: int, obs_dims: Sequence[str] | None) -> list[str]:
    """Validate and normalize the dimension names of the ``obs`` variables.

    Parameters
    ----------
    ndim
        Number of axes of one ``obs`` draw (``data.ndim``).
    obs_dims
        One dimension name per axis, or ``None`` for the default:
        ``obs_batch_0``, ``obs_batch_1``, ... for the leading batch axes, then
        ``("time", "obs_dim")``.

    Returns
    -------
    list[str]
        The dimension names to hand to `arviz_base.dict_to_dataset()`.

    Raises
    ------
    ValueError
        If the number of names does not match ``ndim``, if axis ``-2`` is not
        named ``"time"``, if a name repeats, or if a name is a sample
        dimension (``"chain"`` or ``"draw"``).
    """
    if obs_dims is None:
        return [*(f"obs_batch_{i}" for i in range(ndim - 2)), *_DEFAULT_OBS_DIMS]
    dims = list(obs_dims)
    if len(dims) != ndim:
        msg = (
            f"obs_dims has {len(dims)} names {dims} but data has {ndim} dimensions; "
            "pass one name per axis, with 'time' at axis -2"
        )
        raise ValueError(msg)
    if dims[-2] != "time":
        msg = (
            "obs_dims must name the time axis (axis -2) 'time', which carries the "
            f"tree-wide time coordinate, got {dims[-2]!r}"
        )
        raise ValueError(msg)
    if len(set(dims)) != len(dims):
        msg = f"obs_dims repeats a name: {dims}; pass one distinct name per axis"
        raise ValueError(msg)
    reserved = [dim for dim in dims if dim in _SAMPLE_DIMS]
    if reserved:
        msg = (
            f"obs_dims uses the sample dimension names {reserved}, which ArviZ reserves "
            f"for {_SAMPLE_DIMS}; rename those axes"
        )
        raise ValueError(msg)
    return dims


def _require_draws_match_obs_dims(
    label: str, draws: "Array | np.ndarray", obs_dims: Sequence[str]
) -> None:
    """Raise ``ValueError`` unless every draw of ``draws`` has one axis per ``obs_dims`` name.

    ArviZ assigns the listed names from the left and auto-names any extra axis,
    so a draw with more axes than names would put ``time`` on the wrong axis
    without an error.
    """
    shape = tuple(draws.shape[1:])
    if len(shape) != len(obs_dims):
        msg = (
            f"{label} have shape {shape} per draw ({len(shape)} axes), but the obs "
            f"dimension names {list(obs_dims)} name {len(obs_dims)} axes; every obs "
            "draw must have the (*batch, time, obs) layout of data"
        )
        raise ValueError(msg)


def _require_in_sample_draws_like_data(draws: "Array | np.ndarray", data: Array) -> None:
    """Raise ``ValueError`` unless every in-sample draw has the shape of ``data``.

    `~~numpyro_forecast.predictive.predict_in_sample()` calls the model with
    ``data=None``, so a model whose predictor lacks the series axis of the data
    (a ``(duration, 1)`` predictor over ``(time, n)`` data, without a series
    plate) draws one column, and the export would store it next to the ``n``
    observed columns (issue #259).
    """
    shape = tuple(draws.shape[1:])
    if shape != tuple(data.shape):
        msg = (
            f"the in-sample predictive draws have shape {shape} per draw, but data has shape "
            f"{tuple(data.shape)}. The model draws obs without data, so the predictor or the "
            "plates around predict must carry every axis of data: for a predictor shared by n "
            "series, call predict inside numpyro.plate('series', n, dim=-1) or broadcast the "
            "predictor to (duration, n)."
        )
        raise ValueError(msg)


def _reconcile_tree_covariate_layout(
    tree: "xarray.DataTree",
    covariates_future: Array,
    covariate_dims: Sequence[str] | None,
) -> tuple[Sequence[str] | None, dict[str, Any]]:
    """Inherit or validate ``covariate_dims`` against the tree's stored covariates.

    `add_forecast_groups()` must name and label the future covariates' axes
    exactly like the in-sample ones already stored on
    ``constant_data["covariates"]``, or the two groups silently disagree. When
    the tree carries stored covariates, an omitted ``covariate_dims`` inherits
    their names and an explicit one is cross-checked; trees without stored
    covariates pass through unchanged (the downstream default applies).

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
    tuple[Sequence[str] | None, dict[str, Any]]
        The reconciled dimension names (``None`` only when the tree has no
        stored covariates and none were supplied), and the coordinate labels of
        every non-time axis of the stored covariates (empty without them).

    Raises
    ------
    CovariateDimsError
        If inherited names do not cover every ``covariates_future`` axis, or if
        explicit names disagree with the stored ones.
    """
    constant_node = tree.children.get("constant_data")
    if constant_node is None or "covariates" not in constant_node.dataset:
        return covariate_dims, {}
    stored = constant_node.dataset["covariates"]
    stored_dims = [str(dim) for dim in stored.dims]
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
        return stored_dims, _non_time_labels(stored)
    if list(covariate_dims) != stored_dims:
        msg = (
            f"covariate_dims {list(covariate_dims)} disagree with the names "
            f"{stored_dims} already stored on the tree's constant_data covariates "
            "(set by the earlier to_datatree(covariate_dims=...) call); omit "
            "covariate_dims to inherit them, or pass matching names"
        )
        raise CovariateDimsError(msg)
    return covariate_dims, _non_time_labels(stored)


def _non_time_labels(variable: "xarray.DataArray") -> dict[str, Any]:
    """Return the coordinate labels of every non-time axis of ``variable``.

    The time axis is skipped by position (axis ``-2``, the package-wide
    convention) and by name, so a time axis named otherwise never hands its
    in-sample labels to a forecast variable.
    """
    time_axis = variable.ndim - 2
    return {
        str(dim): variable.coords[dim].values
        for axis, dim in enumerate(variable.dims)
        if axis != time_axis and dim != "time" and dim in variable.coords
    }


def _require_in_sample_sizes(
    argument: str,
    source: str,
    dims: Sequence[str],
    shape: Sequence[int],
    labels: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` unless every labeled axis of a forecast variable keeps its size.

    `add_forecast_groups()` attaches the in-sample ``labels`` (from
    `_non_time_labels()`) to the forecast variable, so an axis of another size
    than its labels is reported here, by argument and axis name, instead of
    by xarray when the dataset is built.

    Parameters
    ----------
    argument
        The argument name to report on a mismatch.
    source
        The stored in-sample variable to report on a mismatch.
    dims
        The dimension names of the non-sample axes of the forecast variable.
    shape
        The sizes of those axes.
    labels
        The in-sample labels per dimension name.
    """
    for dim, size in zip(dims, shape, strict=True):
        values = labels.get(dim)
        if values is not None and len(values) != size:
            msg = (
                f"{argument} has size {size} along {dim!r} but the tree's {source} has size "
                f"{len(values)} along it; every non-time axis of {argument} must keep its "
                "in-sample size"
            )
            raise ValueError(msg)


def _tree_obs_layout(tree: "xarray.DataTree", ndim: int) -> tuple[list[str], dict[str, Any]]:
    """Inherit the ``obs`` dimension names and labels stored on the tree's ``observed_data``.

    `add_forecast_groups()` must lay out the forecast ``obs`` like the in-sample
    one, or the groups disagree on axis names and labels. The stored
    ``observed_data["obs"]`` is inherited when it follows the package layout
    (``time`` at axis ``-2``). A tree without a stored ``obs``, or with one
    that does not follow the layout (an ``observed_data`` squeezed to one
    series before plotting), falls back to the default names of
    `_resolve_obs_dims()` and no labels.

    Parameters
    ----------
    tree
        The tree being extended.
    ndim
        Number of axes of one forecast draw, for the fallback names.

    Returns
    -------
    tuple[list[str], dict[str, Any]]
        The dimension names, and the coordinate labels of every non-time axis
        to share with the forecast ``obs``.
    """
    observed = tree["observed_data"].dataset
    if "obs" in observed:
        stored = observed["obs"]
        dims = [str(dim) for dim in stored.dims]
        if len(dims) >= 2 and dims[-2] == "time":
            return dims, _non_time_labels(stored)
    return _resolve_obs_dims(ndim, None), {}


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
        If ``num_chains`` is not a positive integer (``bool`` included), or if
        the leading axis length is not evenly divisible by ``num_chains``.

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
    if isinstance(num_chains, bool) or num_chains < 1:
        msg = (
            f"num_chains must be a positive integer, got {num_chains}; pass the num_chains "
            "the sampler was run with (1 for SVI or Pathfinder draws)"
        )
        raise ValueError(msg)
    reshaped = np.asarray(array)
    n = reshaped.shape[0]
    if n % num_chains != 0:
        msg = f"posterior sample count {n} is not evenly divisible by num_chains {num_chains}"
        raise ValueError(msg)
    return reshaped.reshape(num_chains, n // num_chains, *reshaped.shape[1:])


def _coordinate_values(name: str, values: Collection[Any]) -> "np.ndarray":
    """Convert one user coordinate to a one-dimensional NumPy array.

    A named ``pd.Index`` handed to xarray as is becomes a coordinate on a
    dimension named after ``Index.name``, which ``arviz_base.dict_to_dataset()``
    rejects. ``np.asarray`` drops the name and keeps the zone of a
    timezone-aware index (an object array of aware timestamps), where
    ``index.values`` would shift the clock to naive UTC.

    A set or a dict is a ``Collection`` too, but it has no order and
    ``np.asarray`` makes it a 0-d object array; a ``pd.MultiIndex`` would
    become one tuple per position and lose its levels. Both raise here, by
    name, instead of failing later inside xarray.
    """
    levels = getattr(values, "nlevels", 1)
    if levels != 1:
        msg = (
            f"{name} is a pandas MultiIndex with {levels} levels, but a coordinate holds one "
            "label per position: pass one level (index.get_level_values(level)) or join the "
            "levels into one label per position"
        )
        raise ValueError(msg)
    array = np.asarray(values)
    if array.ndim != 1:
        msg = (
            f"{name} must be an ordered one-dimensional collection of labels (a list, a "
            f"tuple, a NumPy array or a pandas index), got {type(values).__name__} that "
            f"converts to shape {array.shape} (a set or a dict has no order)"
        )
        raise ValueError(msg)
    return array


def _normalize_coords(coords: Mapping[str, Collection[Any]] | None) -> dict[str, "np.ndarray"]:
    """Convert every user coordinate with `_coordinate_values()`."""
    if coords is None:
        return {}
    return {
        name: _coordinate_values(f"coords[{name!r}]", values) for name, values in coords.items()
    }


def _merge_coords(time: "np.ndarray", coords: Mapping[str, "np.ndarray"]) -> dict[str, Any]:
    """Merge coordinates with precedence user ``coords`` > generated ``time``."""
    return {"time": time, **coords}


def _forecast_group_datasets(
    forecast_draws: "np.ndarray",
    covariates_future: Array,
    future_time: "np.ndarray",
    *,
    obs_dims: Sequence[str],
    obs_coords: Mapping[str, Any] | None = None,
    covariate_coords: Mapping[str, Any] | None = None,
    covariate_dims: Sequence[str] | None = None,
) -> tuple["xarray.Dataset", "xarray.Dataset"]:
    """Build the ``predictions`` and ``predictions_constant_data`` datasets.

    ``forecast_draws`` must already carry the ``(chain, draw, *batch, future,
    obs)`` layout; callers own the chain reshape (`to_datatree()` applies the
    fit's ``num_chains``, `add_forecast_groups()` the layout of the tree's
    posterior). ``obs_dims`` names the non-sample axes of ``forecast_draws``.

    ``obs_coords`` and ``covariate_coords`` are the coordinates of the two
    datasets. They are separate because a dimension name that both variables
    use can carry different labels in each (for example default integer labels
    of different lengths, or the labels `add_forecast_groups()` copies from
    the in-sample counterparts). In both, ``future_time`` always wins over a
    ``time`` entry: `to_datatree()` passes its user ``coords``, whose ``time``
    entry covers the in-sample window (the precedence of `_merge_coords()` is
    inverted here; ``time_coord`` is the sanctioned route for explicit
    forecast time values). ``covariate_dims`` names the axes of
    ``covariates_future`` (default ``("time", "covariate_dim")``).
    """
    dims = _resolve_covariate_dims(covariates_future, covariate_dims)
    obs_future: dict[Any, Any] = {**(obs_coords or {}), "time": future_time}
    covariates_future_coords: dict[Any, Any] = {**(covariate_coords or {}), "time": future_time}
    predictions_ds = arviz_base.dict_to_dataset(
        {"obs": forecast_draws},
        sample_dims=_SAMPLE_DIMS,
        dims={"obs": list(obs_dims)},
        coords=obs_future,
    )
    predictions_constant_ds = arviz_base.dict_to_dataset(
        {"covariates": np.asarray(covariates_future)},
        sample_dims=[],
        dims={"covariates": dims},
        coords=covariates_future_coords,
    )
    return predictions_ds, predictions_constant_ds


def _tree_sample_layout(
    tree: "xarray.DataTree", num_samples: int
) -> tuple[int, dict[str, "np.ndarray"]]:
    """Read the ``(chain, draw)`` layout and labels that forecast draws must take.

    The reference is the tree's ``posterior`` group or, without one, its
    ``posterior_predictive`` group: the first of the two that has a ``draw``
    dimension. `to_datatree()` gives both groups the same layout. A group
    without a ``chain`` dimension (for example after ``tree.isel(chain=0)``)
    counts as one chain, labeled by its scalar ``chain`` coordinate if it has
    one.

    Parameters
    ----------
    tree
        The tree that `add_forecast_groups()` extends.
    num_samples
        The length of the flattened sample axis of the forecast draws.

    Returns
    -------
    tuple[int, dict[str, numpy.ndarray]]
        The number of chains and the ``chain`` and ``draw`` labels of the
        reference group, or ``(1, {})`` (one pseudo-chain with default labels)
        when the tree has no reference group.

    Raises
    ------
    ValueError
        If ``num_samples`` is not the number of chains times the number of
        draws of the reference group.
    """
    for group in ("posterior", "posterior_predictive"):
        node = tree.children.get(group)
        if node is None or "draw" not in node.dataset.sizes:
            continue
        num_chains = node.dataset.sizes.get("chain", 1)
        num_draws = node.dataset.sizes["draw"]
        if num_samples != num_chains * num_draws:
            msg = (
                f"forecast_samples has {num_samples} draws but the tree's {group} group has "
                f"{num_chains * num_draws} (chain x draw = {num_chains} x {num_draws}); draw "
                "one forecast per draw of that group, in the order of "
                f'tree["{group}"].stack(sample=("chain", "draw")), and to attach fewer '
                "draws, subset the tree first (for example with tree.isel(draw=...))"
            )
            raise ValueError(msg)
        coords = node.dataset.coords
        labels = {dim: np.atleast_1d(coords[dim].values) for dim in _SAMPLE_DIMS if dim in coords}
        return num_chains, labels
    return 1, {}


def to_datatree(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, ArrayLike],
    data: Array,
    covariates: Array,
    *,
    num_chains: int = 1,
    batch_size: int | None = None,
    device: jax.Device | str | None = "host",
    coords: Mapping[str, Collection[Any]] | None = None,
    time_coord: Collection[Any] | None = None,
    posterior_dims: Mapping[str, Sequence[str]] | None = None,
    covariate_dims: Sequence[str] | None = None,
    obs_dims: Sequence[str] | None = None,
) -> "xarray.DataTree":
    r"""Convert an already-drawn posterior into an ArviZ-schema `xarray.DataTree`.

    Posterior-first: callers draw their own posterior (``mcmc.get_samples()``
    for MCMC, `~~numpyro_forecast.predictive.draw_posterior()` for a
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
        Posterior samples of the latent sites of the training run, with a
        single flattened sample axis leading (NumPyro's ``mcmc.get_samples()``
        order, or the output of
        `~~numpyro_forecast.predictive.draw_posterior()`).
        Host-committed leaves (the output of ``draw_posterior(...,
        device="host")``) and NumPy leaves are accepted directly.
        Forecast-horizon keys follow the rule of
        `~~numpyro_forecast.predictive.forecast()`: one with values (a key
        that ends in ``"_future"`` or contains ``"_future_"``, or
        ``"forecast"``) raises ``ValueError``. One with an empty leaf (a site
        that the model registers on every call, recorded by the training run
        with a size-0 time axis) is kept in the ``posterior`` group as a
        size-0 variable, and the predictive drivers drop it. An ``"obs"``
        key is dropped, as in
        `~~numpyro_forecast.predictive.predict_in_sample()`: it is not in the
        ``posterior`` group, and the in-sample draws come from the latent
        sites.
    data
        In-sample data shaped ``(*batch, time, obs)``: time at axis ``-2``,
        the observation dim at axis ``-1`` and any batch axes to the left
        (``obs_dims`` names them).
    covariates
        Covariates with time at axis ``-2``. When ``covariates`` extends beyond
        ``data`` along the time axis (the package-wide shape convention for a
        forecast horizon), the trailing rows are treated as future covariates:
        the returned tree additionally carries ``predictions`` (forecast ``obs``
        draws from `~~numpyro_forecast.predictive.forecast()`) and
        ``predictions_constant_data`` groups.
    num_chains
        Number of chains to split ``posterior``'s flattened sample axis into
        (and, identically, the in-sample/forecast predictive draws, which are
        drawn with the same sample count). Defaults to ``1`` (a single
        pseudo-chain, correct for a posterior with no chain structure, e.g.
        SVI or Pathfinder draws). For an MCMC posterior, pass the
        ``num_chains`` the sampler was run with; see `_reshape_chains()`
        for the reshape contract and its divisibility requirement. Must be a
        positive integer.
    batch_size
        Optional chunk size that bounds how many draws touch the accelerator
        at once, across both the in-sample and forecast predictive sampling.
        When set, sampling runs in chunks of this many draws, each chunk moved
        to ``device`` before the next is drawn. The per-chunk
        accelerator footprint is a handful of ``(batch_size, time, series)``
        buffers, so it scales linearly with this value times the panel width:
        on wide panels lower it until a chunk fits. The batch size must be
        strictly below the draw count for that bound to hold: at or above it,
        sampling falls back to the single-shot path and the full array is
        materialized on the default device before the single transfer.
        Chunking changes the PRNG stream layout of the predictive draws, so
        results are reproducible per ``(rng_key, batch_size)``.
        ``None`` (default) samples everything in one shot (the results are
        still moved to ``device``).
    device
        Where the predictive draws are moved as they are sampled, forwarded to
        the ``device`` argument of
        `~~numpyro_forecast.predictive.predict_in_sample()` and
        `~~numpyro_forecast.predictive.forecast()` (the placement contract
        of `~~numpyro_forecast.predictive.draw_posterior()`). It is resolved
        once and the same placement is handed to both, so an unmet ``"cpu"``
        warns once per export. The default ``"host"`` keeps the predictive
        draws in pageable host memory (jax Arrays the tree views as NumPy
        without a copy, or NumPy arrays when no CPU backend is initialized),
        which is what bounds accelerator memory when ``batch_size``
        is set; pass ``None`` to keep the draws on the default device (chunked
        compute without per-chunk host transfers, for when the draws fit on
        the accelerator and transfers would dominate runtime).
    coords
        Optional extra coordinates; these take precedence over the generated
        ``time`` coordinate. They also propagate to the forecast groups, where
        the generated forecast ``time`` takes precedence instead (a user
        ``time`` entry covers the in-sample window; use ``time_coord`` for
        explicit forecast time values). Each value is an ordered
        one-dimensional collection of labels (a list, a NumPy array, a pandas
        index) and is stored as ``np.asarray(value)``, so the name of a pandas
        index is dropped.
    time_coord
        Optional explicit time coordinate values. Without a forecast horizon it
        covers the in-sample window (defaults to ``range(n_time)``); with a
        horizon it must cover the full ``covariates`` length and is split into
        the in-sample and forecast time coordinates (the default is the integer
        continuation). Any ordered one-dimensional collection is accepted (a
        list, a NumPy array, a pandas index) and stored as
        ``np.asarray(time_coord)``. A timezone-aware ``pd.DatetimeIndex`` keeps
        its zone and clock time, as an object array of aware ``pd.Timestamp``
        values (``arviz_base.dict_to_dataset()`` cannot store a zoned datetime
        dtype). xarray does not treat that coordinate as a datetime: ``.dt``,
        ``resample()`` and selection by a date string fail (select with a
        ``pd.Timestamp``), and ``to_netcdf()`` cannot write it. For a datetime
        coordinate pass ``index.tz_convert("UTC").tz_localize(None)`` (UTC
        times), or ``index.tz_localize(None)`` (local clock times, which
        repeat an hour when daylight saving time ends).
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
    obs_dims
        Optional dimension names for the ``obs`` variables of
        ``observed_data``, ``posterior_predictive`` and ``predictions``, one per
        ``data`` axis. Defaults to ``("time", "obs_dim")`` for 2-D ``data`` and
        names the leading batch axes of a panel ``obs_batch_0``,
        ``obs_batch_1``, ..., e.g. ``("obs_batch_0", "time", "obs_dim")`` for
        ``(batch, time, obs)`` data. Give a batch axis the name of the
        matching ``covariate_dims`` entry so the groups share it, e.g.
        ``obs_dims=["origin", "time", "destin"]`` with
        ``covariate_dims=["origin", "time", "covariate_dim"]``; ``coords``
        labels and ``posterior_dims`` entries refer to the same names. Axis
        ``-2`` must be named ``"time"`` (it carries the tree-wide time
        coordinate), the names must be distinct, and ``"chain"`` and
        ``"draw"`` are reserved for the sample dims.

    Returns
    -------
    xarray.DataTree
        A tree with ``posterior`` (``(chain, draw, ...)``, split per
        ``num_chains``), ``posterior_predictive`` (in-sample ``obs``),
        ``observed_data``, and ``constant_data`` groups. When ``covariates``
        extends beyond ``data``, also ``predictions`` and
        ``predictions_constant_data`` groups (sharing the same ``num_chains``
        split). The ``obs`` variable has dims ``obs_dims`` in
        ``observed_data`` and ``(chain, draw, *obs_dims)`` in
        ``posterior_predictive`` and ``predictions``.

    Raises
    ------
    ValueError
        If ``data`` or ``covariates`` has fewer than two axes (time must be at
        axis ``-2``), if ``data`` has no time steps, if ``covariates`` is
        shorter than ``data`` along the time axis, if a ``posterior`` leaf has
        no sample axis, if a forecast-horizon key of ``posterior`` (one that
        ends in ``"_future"`` or contains ``"_future_"``, or ``"forecast"``)
        has a non-empty leaf, if ``"obs"`` is the only key of ``posterior``, if
        ``time_coord`` or a ``coords`` value is not an ordered one-dimensional
        collection (a set, a dict, a scalar, a 2-D array) or is a
        ``pd.MultiIndex``, if ``time_coord`` is given but its length does not
        match the in-sample window plus the forecast horizon, if ``num_chains``
        is not a positive integer (``bool`` included), if ``posterior``'s
        sample count is not evenly divisible by ``num_chains``, if a
        ``posterior`` value of an
        `~~numpyro_forecast.models.innovations()` site does not have one step
        per ``data`` row (a posterior drawn for a different training length),
        or if the in-sample draws do not have one time step per ``data`` row or
        the forecast draws one per ``covariates`` row past ``data`` (see
        `~~numpyro_forecast.predictive.predict_in_sample()` and
        `~~numpyro_forecast.predictive.forecast()`). Also if ``obs_dims`` does
        not name every ``data`` axis, does not name axis ``-2`` ``"time"``,
        repeats a name or uses ``"chain"`` or ``"draw"``, if the in-sample
        or forecast draws do not have one axis per ``obs_dims`` name (the
        model's ``obs`` site must have the layout of ``data``), or if an
        in-sample draw does not have the shape of ``data`` (a predictor shared
        by the series of ``data`` without a series plate draws one column).
    CovariateDimsError
        If ``covariate_dims`` does not name every ``covariates`` axis.
    HostMemoryKindError
        If ``device="pinned_host"`` is requested on a device that
        exposes no host memory kind (see
        `~~numpyro_forecast.offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not
        initialized (see `~~numpyro_forecast.offload.resolve_device()`).

    Warns
    -----
    UserWarning
        If ``device="cpu"`` is requested and the JAX CPU backend is
        not initialized, so the predictive draws take the NumPy path of
        ``"host"`` instead (once per call).

    Notes
    -----
    When a forecast horizon is present, ``rng_key`` is split internally into a
    predictive subkey and a forecast subkey, so passing the same key twice
    never correlates the two sample sets. When there is no horizon, ``rng_key``
    is used unsplit for the in-sample predictive draw. ``batch_size``
    is the built-in route to memory-bounded predictive sampling; for fully
    manual control over the forecast draws, build the in-sample tree with
    matching-length covariates and attach the horizon with
    `add_forecast_groups()`.
    """
    require_covariates_cover_data(data, covariates)
    cov_dims = _resolve_covariate_dims(covariates, covariate_dims)
    resolved_obs_dims = _resolve_obs_dims(data.ndim, obs_dims)
    n_time = data.shape[-2]
    horizon = covariates.shape[-2] - n_time

    if horizon > 0:
        key_pred, key_forecast = random.split(rng_key, 2)
    else:
        key_pred, key_forecast = rng_key, None

    if time_coord is not None:
        time_values = _coordinate_values("time_coord", time_coord)
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
    user_coords = _normalize_coords(coords)
    merged_arg = cast("dict[Any, Any]", _merge_coords(time, user_coords))

    require_sample_axis(posterior)
    require_no_horizon_sites(posterior)
    posterior = drop_obs_site(posterior)
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
    resolved_device = resolve_device(device)
    predictive = predict_in_sample(
        key_pred,
        model,
        posterior,
        covariates_insample,
        batch_size=batch_size,
        device=resolved_device,
    )
    _require_in_sample_draws_like_data(predictive, data)
    pp_ds = arviz_base.dict_to_dataset(
        {"obs": _reshape_chains(predictive, num_chains)},
        sample_dims=_SAMPLE_DIMS,
        dims={"obs": resolved_obs_dims},
        coords=merged_arg,
    )

    observed_ds = arviz_base.dict_to_dataset(
        {"obs": np.asarray(data)},
        sample_dims=[],
        dims={"obs": resolved_obs_dims},
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
            batch_size=batch_size,
            device=resolved_device,
        )
        _require_draws_match_obs_dims("the forecast draws", forecast_samples, resolved_obs_dims)
        predictions_ds, predictions_constant_ds = _forecast_group_datasets(
            _reshape_chains(forecast_samples, num_chains),
            covariates[..., n_time:, :],
            future_time,
            obs_dims=resolved_obs_dims,
            obs_coords=user_coords,
            covariate_coords=user_coords,
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
    forecast_samples: Num[ArrayLike, " sample *batch future obs"],
    covariates_future: Array,
    *,
    time_coord: Collection[Any] | None = None,
    covariate_dims: Sequence[str] | None = None,
) -> "xarray.DataTree":
    """Attach out-of-sample forecast groups to a copy of ``tree``.

    Adds a ``predictions`` group (the forecast ``obs`` draws) and a
    ``predictions_constant_data`` group (the future covariates). The forecast
    ``time`` coordinate continues the in-sample one: integer continuation by
    default, or explicit values via ``time_coord``. This is the step-by-step
    route for draws you produced yourself; `to_datatree()` attaches the same
    groups automatically when its ``covariates`` extend beyond ``data``. Both
    routes give the forecast groups the ``(chain, draw)`` layout of the
    posterior and the dimension names and labels of the in-sample ``obs`` and
    covariates on every non-time axis, so xarray aligns the forecast with the
    rest of the tree draw by draw and label by label.

    Parameters
    ----------
    tree
        A tree from `to_datatree()`. The forecast continues its
        ``observed_data`` time coordinate, takes the ``(chain, draw)`` layout
        and labels of its ``posterior`` group, and copies the labels of the
        non-time axes of ``observed_data["obs"]`` and
        ``constant_data["covariates"]`` to the same axes of ``predictions`` and
        ``predictions_constant_data``.
    forecast_samples
        Forecast draws shaped ``(num_samples, *batch, future, obs)`` from
        `~~numpyro_forecast.predictive.forecast()`, with the layout of the
        tree's in-sample ``obs``, one per draw of the tree's ``posterior``
        group as it is now, in the order of
        ``tree["posterior"].stack(sample=("chain", "draw"))`` (for a tree
        straight from `to_datatree()`, the order of the flattened posterior it
        was built from). They are split into the ``(chain, draw)`` layout of
        that group; a group without a ``chain`` dimension (for example after
        ``tree.isel(chain=0)``) counts as one chain. A tree with no
        ``posterior`` group uses its ``posterior_predictive`` group instead,
        and a tree with neither gets one pseudo-chain. The ``predictions``
        ``obs`` takes the dimension names and the non-time coordinate labels
        of the tree's ``observed_data["obs"]`` (set by
        ``to_datatree(obs_dims=..., coords=...)``), so each non-time axis must
        have the size of the stored one. If the tree's stored ``obs`` is
        missing or has no ``time`` axis at ``-2`` (an ``observed_data``
        squeezed to one series), it takes the default names of
        `to_datatree()` and no labels instead.
    covariates_future
        Future covariates shaped ``(future, covariate_dim)``, or any layout
        with time at axis ``-2`` when ``covariate_dims`` names the axes, with
        one time step per forecast step.
    time_coord
        Optional explicit forecast time coordinate; defaults to integer
        continuation of the in-sample time. Required when the in-sample time
        coordinate is non-integer (e.g. datetime64): auto-continuing would have
        to guess the frequency, so explicit values are demanded instead. Any
        ordered one-dimensional collection is accepted (a list, a NumPy array,
        a pandas index) and stored as ``np.asarray(time_coord)``. A
        timezone-aware ``pd.DatetimeIndex`` keeps its zone, as an object array
        of aware timestamps that xarray does not treat as a datetime; see
        ``time_coord`` of `to_datatree()` for the costs and the alternatives.
    covariate_dims
        Optional dimension names for ``covariates_future``, one per axis. When
        omitted, the names are inherited from the tree's
        ``constant_data["covariates"]`` variable (falling back to
        ``("time", "covariate_dim")`` if the tree carries no stored
        covariates), so the forecast covariates always share the in-sample
        axis names. When given explicitly, the names must match the stored
        ones. The future covariates also take the non-time coordinate labels
        of the stored covariates. See `to_datatree()`.

    Returns
    -------
    xarray.DataTree
        A new tree with the ``predictions`` and ``predictions_constant_data``
        groups added.

    Raises
    ------
    ValueError
        If ``forecast_samples`` does not have one draw per draw of the tree's
        posterior (to attach fewer draws, subset the tree first, for example
        with ``tree.isel(draw=...)``), if it does not have one axis per
        inherited ``obs`` dimension name after the sample axis, if a non-time
        axis of ``forecast_samples`` or ``covariates_future`` differs in size
        from the same axis of the tree's in-sample ``obs`` or covariates, if
        ``covariates_future`` does not have one time step per forecast step,
        if ``time_coord`` is not an ordered one-dimensional collection (a set,
        a dict, a scalar, a 2-D array) or is a ``pd.MultiIndex``, if
        ``time_coord`` is given but its length differs from the forecast
        horizon, or if it is omitted while the in-sample time coordinate is
        non-integer.
    CovariateDimsError
        If the resolved ``covariate_dims`` (explicit or inherited) do not name
        every ``covariates_future`` axis, or if explicit names disagree with
        the dimension names already stored on the tree's
        ``constant_data["covariates"]``.
    """
    covariate_dims, covariate_coords = _reconcile_tree_covariate_layout(
        tree, covariates_future, covariate_dims
    )
    covariate_dims = _resolve_covariate_dims(covariates_future, covariate_dims)
    in_time = np.asarray(tree["observed_data"].coords["time"].values)
    forecast_samples_arr = np.asarray(forecast_samples)
    covariates_future_arr = np.asarray(covariates_future)
    obs_dims, obs_coords = _tree_obs_layout(tree, forecast_samples_arr.ndim - 1)
    _require_draws_match_obs_dims("forecast_samples", forecast_samples_arr, obs_dims)
    _require_in_sample_sizes(
        "forecast_samples",
        "observed_data obs",
        obs_dims,
        forecast_samples_arr.shape[1:],
        obs_coords,
    )
    _require_in_sample_sizes(
        "covariates_future",
        "constant_data covariates",
        covariate_dims,
        covariates_future_arr.shape,
        covariate_coords,
    )
    future_len = forecast_samples_arr.shape[-2]
    if covariates_future_arr.shape[-2] != future_len:
        msg = (
            f"covariates_future has length {covariates_future_arr.shape[-2]} along the time "
            f"axis (-2) but forecast_samples has length {future_len} along it; pass the "
            "covariate rows of the forecast horizon only, e.g. covariates[..., n_time:, :]"
        )
        raise ValueError(msg)
    if time_coord is not None:
        future_time = _coordinate_values("time_coord", time_coord)
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
    num_chains, sample_labels = _tree_sample_layout(tree, forecast_samples_arr.shape[0])
    predictions_ds, predictions_constant_ds = _forecast_group_datasets(
        _reshape_chains(forecast_samples_arr, num_chains),
        covariates_future,
        future_time,
        obs_dims=obs_dims,
        obs_coords={**sample_labels, **obs_coords},
        covariate_coords=covariate_coords,
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
    series: Collection[Any],
    *,
    group: str = "posterior_predictive",
    observed: Float[ArrayLike, " time series"] | None = None,
) -> "xarray.DataTree":
    """Pack prediction draws into a DataTree laid out for per-series ``plot_lm`` faceting.

    The array-level counterpart of `to_datatree()`: instead of a fit, it takes
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
        `matplotlib.dates.date2num()` floats and re-format the tick labels
        with `matplotlib.dates.ConciseDateFormatter`.
    series
        One label per series, defining the ``series`` coordinate. Any
        ordered one-dimensional collection is accepted (a list, a NumPy
        array, a pandas index) and stored as ``np.asarray(series)``.
    group
        Predictive group to store the draws under (e.g. ``"prior_predictive"``,
        ``"posterior_predictive"``, ``"predictions"``). ``"observed_data"`` and
        ``"constant_data"`` are reserved: the function always writes those two
        groups.
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
        If ``group`` is ``"observed_data"`` or ``"constant_data"`` (reserved
        names), if ``series`` is not an ordered one-dimensional collection (a
        set, a dict, a scalar, a 2-D array) or is a ``pd.MultiIndex``, or if
        ``series`` does not have one label per series in ``predictions``.
    """
    if group in _RESERVED_PREDICTION_GROUPS:
        msg = (
            f"group={group!r} is reserved: predictions_to_datatree() always writes the "
            "'observed_data' and 'constant_data' groups. Store the draws under another "
            "name, for example 'posterior_predictive', 'prior_predictive' or 'predictions'"
        )
        raise ValueError(msg)
    preds = np.asarray(predictions)[None]
    series_values = _coordinate_values("series", series)
    if series_values.shape[0] != preds.shape[-1]:
        msg = (
            f"series has length {series_values.shape[0]} but predictions carry "
            f"{preds.shape[-1]} series"
        )
        raise ValueError(msg)
    x_values = np.asarray(x)
    x_grid = np.broadcast_to(x_values[:, None], preds.shape[2:])
    coords = cast("dict[Any, Any]", {"time": x_values, "series": series_values})
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
