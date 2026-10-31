"""Model building blocks: plain functions that register the train/forecast sites.

The `Horizon` value carries the train/forecast split, and the building
blocks (`innovations()`, `markov_series()`, `ssoe()`, `predict()`)
sample latents and observation sites against it. A model is a plain NumPyro function
``(covariates, data=None) -> None`` whose first line derives its
`Horizon` from the shapes with `Horizon.from_data()` and which then
calls the blocks directly. They are ordinary Python functions that call
``numpyro.sample`` and ``numpyro.deterministic`` on your behalf: not NumPyro
primitives, and not effect handlers.

The recursive blocks take plain per-step functions: `markov_series()` a
`Transition` (and an optional `Advance`), `ssoe()` an `SSOEMean` and an
`SSOEUpdate`; none of them returns a closure, and the wrapper owns every
sample site.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from enum import StrEnum
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Float, PyTree
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import Reparam
from numpyro.primitives import _PYRO_STACK

from numpyro_forecast.arrays import concat_future
from numpyro_forecast.surgery import prefix_condition, shift_loc, slice_time
from numpyro_forecast.typing import Array
from numpyro_forecast.validation import require_time_axis


class PlateName(StrEnum):
    """Names of the plates the building blocks open, as ``str`` members.

    ``TIME`` is the in-sample plate of `innovations()`; ``TIME_FUTURE`` is the
    forecast-horizon plate of `innovations()` and `ssoe()`.
    `~~numpyro_forecast.reparam.time_reparam()` targets the sites under ``TIME`` by
    name, so both modules read this enum rather than spelling the literal.
    """

    TIME = "time"
    TIME_FUTURE = "time_future"


class Horizon(eqx.Module):
    """The train/forecast split for a single model call.

    An immutable value derived once per model call from the covariate and data
    shapes by `from_data()`; every building block (`innovations()`,
    `markov_series()`, `ssoe()`, `predict()`) takes it as its first
    argument.

    A JAX pytree (an `equinox.Module`): ``data`` is the only leaf, while
    ``t_obs``, ``future`` and ``duration`` are static metadata. A jitted function
    that takes a `Horizon` can therefore use the three integers as shapes and
    recompiles once per horizon length.

    Attributes
    ----------
    data
        Observed in-sample data with time at axis ``-2`` (``None`` during pure
        prior sampling).
    t_obs
        Number of observed (in-sample) time steps ``t``.
    future
        Number of forecast time steps ``f`` (``0`` while training).
    duration
        Total horizon length ``t + future`` (in time steps).
    """

    data: Array | None
    t_obs: int = eqx.field(static=True)
    future: int = eqx.field(static=True)
    duration: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate that the horizon fields are internally consistent."""
        if self.t_obs < 0 or self.future < 0:
            msg = "t_obs and future must be non-negative"
            raise ValueError(msg)
        if self.duration != self.t_obs + self.future:
            msg = "duration must equal t_obs + future"
            raise ValueError(msg)

    @property
    def zero_data(self) -> Array | None:
        """Zeros shaped like ``data`` extended to the full horizon.

        Mirrors Pyro's ``zero_data`` (and `numpyro_forecast.arrays.zero_data_like()`):
        it exposes the shape/dtype of the data over the forecast horizon without
        leaking observed values. ``None`` when there is no data.

        Returns
        -------
        Array | None
            Zeros of shape ``(*batch, duration, obs)``, or ``None``.
        """
        if self.data is None:
            return None
        shape = (*self.data.shape[:-2], self.duration, self.data.shape[-1])
        return jnp.zeros(shape, dtype=self.data.dtype)

    @classmethod
    def from_data(cls, covariates: Array, data: Array | None) -> "Horizon":
        """Derive the horizon from the covariate and data shapes.

        The first line of every model: ``h = Horizon.from_data(covariates, data)``.

        Parameters
        ----------
        covariates
            Covariates with time at axis ``-2`` spanning the full horizon.
        data
            Observed data with time at axis ``-2`` (``None`` for prior sampling).

        Returns
        -------
        Horizon
            The horizon with ``duration = covariates.shape[-2]``,
            ``t_obs = data.shape[-2]`` (or ``duration`` when ``data`` is ``None``),
            and ``future = duration - t_obs``.

        Raises
        ------
        ValueError
            If ``covariates`` or ``data`` has fewer than two axes (time must be
            at axis ``-2``), if ``data`` has no time steps (for a series with
            no history, call the model with ``data=None``), if ``covariates``
            has no time steps, or if ``data`` is longer than ``covariates``
            along the time axis.
        """
        require_time_axis("covariates", covariates)
        duration = covariates.shape[-2]
        t_obs = duration
        if data is not None:
            require_time_axis("data", data)
            t_obs = data.shape[-2]
            if t_obs == 0:
                msg = (
                    f"data has no observed time steps (shape {tuple(data.shape)}); a model "
                    "needs at least one in-sample step. For a series with no history, call the "
                    "model with data=None: predict_in_sample(rng_key, model, posterior, "
                    "covariates) with only the global sites in posterior, or "
                    "numpyro.infer.Predictive(model, num_samples=...)(rng_key, covariates) "
                    "for the prior"
                )
                raise ValueError(msg)
        if duration == 0:
            msg = (
                f"covariates has no time steps (shape {tuple(covariates.shape)}); the horizon "
                "needs at least one time step at axis -2"
            )
            raise ValueError(msg)
        if t_obs > duration:
            msg = "data must not be longer than covariates along the time axis"
            raise ValueError(msg)
        return cls(data=data, t_obs=t_obs, future=duration - t_obs, duration=duration)


def _sample_time_block(
    site: str,
    size: int,
    plate_name: str,
    prior: dist.Distribution,
    reparam: Reparam | None,
) -> Array:
    """Sample a single time block of ``size`` steps under a time plate at axis ``-2``."""
    with ExitStack() as stack:
        if reparam is not None:
            stack.enter_context(numpyro.handlers.reparam(config={site: reparam}))
        stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
        value = jnp.asarray(numpyro.sample(site, prior))
    _require_block_steps(site, size, plate_name, len(prior.event_shape), value)
    return value


def _require_block_steps(
    site: str, size: int, plate_name: str, event_rank: int, value: Array
) -> None:
    """Require the value of a time block to have ``size`` steps on its plate axis.

    The plate draws ``size`` steps, but NumPyro substitutes a given value (a
    ``Predictive`` posterior, ``substitute``, ``condition``, ``init_to_value``)
    without checking it against the plate. A posterior drawn for another
    training length would otherwise change the length of the series silently,
    which a model that slices its own sites can hide from the drivers (issue
    #168). The plate axis is ``-2`` counted left of the ``event_rank`` event axes.
    """
    # An enumerated discrete site (``infer={"enumerate": "parallel"}``) holds its
    # support as ``(K, 1, ..., 1)``, not the plate length, so it fails this check;
    # supporting enumeration would mean skipping the check for such sites.
    axis = -2 - event_rank
    shape = tuple(value.shape)
    if value.ndim < -axis:
        msg = (
            f"innovations expects the value of site {site!r} to have at least {-axis} axes, "
            f"with time at axis {axis} (the {size} steps of the '{plate_name}' plate), but it "
            f"has shape {shape}. A univariate series keeps a size-1 observation axis: "
            "(time, 1), not (time,); do not squeeze the posterior leaf."
        )
        raise ValueError(msg)
    if shape[axis] == size:
        return
    if plate_name == PlateName.TIME:
        cause = (
            f"a posterior drawn for a different training length: fit it on data with {size} "
            "rows, or pass the data and covariates of the window it was fit on"
        )
    else:
        cause = (
            "a posterior that holds the forecast-horizon site: pass the posterior of the "
            "training run, which holds no values for the _future sites"
        )
    msg = (
        f"innovations expects site {site!r} to have {size} time steps at axis {axis} (the "
        f"size of the '{plate_name}' plate), but its value has shape {shape}. "
        f"The usual cause is {cause}."
    )
    raise ValueError(msg)


def innovations(
    h: Horizon,
    name: str,
    prior: dist.Distribution,
    *,
    reparam: Reparam | None = None,
) -> Array:
    """Sample conditionally iid per-step innovations over the full horizon.

    The in-sample portion is sampled under ``plate("time", t)`` with the fixed
    site ``name``; when forecasting, the horizon portion is sampled under a
    separate site ``f"{name}_future"`` and concatenated. The separate site keeps
    the guide shape fixed and lets ``Predictive`` draw the forecast suffix from
    the prior. Build the series arithmetically from the result (a random walk
    is ``jnp.cumsum(drift, axis=-2)``); a latent whose per-step distribution
    depends on the previous state is `markov_series()`, and a deterministic
    error-feedback recursion driven by the observed series is `ssoe()`.

    Parameters
    ----------
    h
        The horizon for the current model call (see `Horizon`).
    name
        Base sample-site name for the in-sample latent.
    prior
        The per-step prior distribution, shared by the in-sample and forecast
        sites (each time plate expands a copy; the instance is never mutated).
        Its batch shape is the per-step shape, for example ``()`` for a scalar
        latent or ``(n_series,)`` under an enclosing series plate; the time axis
        comes from the plate.
    reparam
        Optional reparameterization (e.g. ``LocScaleReparam``) applied to both
        the in-sample and forecast sites.

    Returns
    -------
    Array
        The latent over the full horizon with time at axis ``-2``.

    Raises
    ------
    ValueError
        If a given value of the ``name`` or ``f"{name}_future"`` site (for
        example a posterior that ``Predictive`` substitutes) does not have
        ``h.t_obs`` or ``h.future`` steps on its time plate: the usual cause is
        a posterior drawn for a different training length.
    """
    prefix = _sample_time_block(name, h.t_obs, PlateName.TIME, prior, reparam)
    if h.future <= 0:
        return prefix
    suffix = _sample_time_block(f"{name}_future", h.future, PlateName.TIME_FUTURE, prior, reparam)
    return concat_future(prefix, suffix, axis=-2)


type Transition[Carry] = Callable[[Carry, PyTree[Array] | None], dist.Distribution]
"""``(carry, x_t) -> dist_t``: the distribution of the next latent given the carry.
The wrapper owns the sample statement; see `Advance` for the carry update.

``Carry`` is the user's carry type (any PyTree), bound per `markov_series()`
call; ``x_t`` is one row of the ``xs`` PyTree (``None`` for autonomous dynamics)."""

type Advance[Carry] = Callable[[Carry, Array, PyTree[Array] | None], Carry]
"""``(carry, z_t, x_t) -> carry``: the next carry from the current carry, the
*sampled* latent ``z_t`` and the exogenous row. Omit it when the carry is the
latent itself (the AR(1) case); a lag window keeps the last ``p`` samples."""


@contextmanager
def _plate_stack(plates: Sequence[tuple[str, int]]) -> Iterator[None]:
    """Open nested ``numpyro.plate`` contexts (innermost last)."""
    with ExitStack() as stack:
        for plate_name, size in plates:
            stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
        yield


def _reject_enclosing_plates() -> None:
    """Raise if ``markov_series`` is called inside an enclosing plate.

    NumPyro exposes no public accessor for the active effect handlers, so this
    deliberately walks its private message stack (``numpyro.primitives._PYRO_STACK``,
    imported at module level so a rename fails at import time rather than
    silently disabling the check). ``numpyro.plate`` is the public handler class,
    so the match is an ``isinstance`` check, not a name comparison.
    ``tests/test_markov.py::test_enclosing_plate_rejected_with_guidance`` is the
    canary that fails loudly if numpyro's internals shift.
    """
    for msg in _PYRO_STACK:
        if isinstance(msg, numpyro.plate):
            msg_text = (
                "markov_series opens plates internally via the plates= "
                "argument; do not wrap the call in an enclosing numpyro.plate."
            )
            raise ValueError(msg_text)


def _validate_markov_step_dist(dist_t: dist.Distribution) -> None:
    """Require a non-degenerate per-step shape with an observation axis."""
    if len(dist_t.event_shape) == 0 and len(dist_t.batch_shape) == 0:
        msg = (
            "markov_series requires the transition distribution to carry "
            "the trailing observation dimension; add it, e.g. loc=...[..., None]."
        )
        raise ValueError(msg)


def _validate_markov_default_carry[Carry](carry: Carry, z: Array) -> Carry:
    """Require the sampled latent to match the carry when ``advance`` is omitted."""
    leaves = jax.tree.leaves(carry)
    structure_differs = jax.tree.structure(carry) != jax.tree.structure(z)
    if structure_differs or leaves[0].shape != z.shape or leaves[0].dtype != z.dtype:
        msg = (
            "markov_series without advance= uses the sampled latent as the next carry, so "
            f"init_carry must be a single array shaped like one draw ({z.dtype}{z.shape}); got "
            f"{jax.tree.structure(carry)} with leaves "
            f"{[f'{leaf.dtype}{leaf.shape}' for leaf in leaves]}; pass advance=(carry, z_t, x_t) "
            "-> carry to build a structured carry (for example a lag window) from the draw."
        )
        raise ValueError(msg)
    return cast(Carry, z)


def markov_series[Carry](
    h: Horizon,
    name: str,
    init_carry: Carry,
    transition: Transition[Carry],
    xs: PyTree[Array] | None = None,
    *,
    advance: Advance[Carry] | None = None,
    plates: Sequence[tuple[str, int]] = (),
    reparam_config: Mapping[str, Reparam] | None = None,
) -> Array:
    """Sample a Markov (state-space) latent over the full horizon.

    In-sample steps run in a ``numpyro.contrib.control_flow.scan`` with site
    ``name``; when forecasting, horizon steps run in a second scan with site
    ``f"{name}_future"`` **seeded by the final in-sample carry**. The guide
    never sees the future site (same invariant as `innovations()`), and
    under posterior replay the carry is a deterministic function of the replayed
    draws, so the forecast is conditioned through the state.

    Parameters
    ----------
    h
        The horizon for the current model call (see `Horizon`).
    name
        Base sample-site name for the in-sample latent scan.
    init_carry
        Initial carry passed to the first transition.
    transition
        Per-step ``(carry, x_t) -> dist_t`` callable returning the distribution
        of the next latent (see `Transition`); the wrapper owns the
        ``numpyro.sample`` statement.
    xs
        Optional exogenous inputs over the full horizon: a PyTree of arrays
        with time at axis ``-2`` and ``duration`` rows (a single array, a
        tuple, a dict, ...), moved leaf by leaf into scan layout internally;
        ``None`` for autonomous dynamics. Slice each leaf from ``covariates``,
        which has ``duration`` rows in every model call (training included),
        or extend an in-sample input with
        `~~numpyro_forecast.arrays.pad_future()`.
    advance
        Optional ``(carry, z_t, x_t) -> carry`` (see `Advance`) that builds the
        next carry from the sampled latent; ``None`` means the carry *is* the
        latent, ``carry_{t+1} = z_t``, so ``init_carry`` must be a single array
        shaped like one draw. A vector autoregression with ``p`` lags keeps a
        ``(p, obs)`` window here.
    plates
        ``(name, size)`` pairs opened **inside** the scan body around the sample
        statement (the only placement NumPyro supports for scan + plate).
    reparam_config
        Site-name -> `numpyro.infer.reparam.Reparam` mapping applied
        **inside** the scan body.

    Returns
    -------
    Array
        The latent over the full horizon in package layout
        ``(*plate_batch, duration, obs)``.

    Raises
    ------
    ValueError
        If forecasting without observed data (only reachable with a hand-built
        `Horizon`: `Horizon.from_data()` never sets ``future > 0`` without
        data), if an ``xs`` leaf lacks the time or observation axis or does
        not span ``h.duration`` rows, if the per-step shape lacks the
        observation dimension, if an enclosing plate is detected, or if
        ``advance`` is omitted and ``init_carry`` is not a single array with
        the shape and dtype of a draw.
    """
    if h.future > 0 and h.data is None:
        msg = "markov_series requires observed data when forecasting"
        raise ValueError(msg)
    _reject_enclosing_plates()
    _validate_xs_leaves(
        h,
        xs,
        "markov_series",
        "Slice each xs leaf from covariates (h.duration rows in every model call, training "
        "too) or extend an in-sample input with pad_future(x, h.future); do not pass the "
        "observed window alone or a longer array.",
    )

    def _body(site_name: str) -> Callable[[Carry, PyTree[Array] | None], tuple[Carry, Array]]:
        def body(carry: Carry, x_t: PyTree[Array] | None) -> tuple[Carry, Array]:
            dist_t = transition(carry, x_t)
            _validate_markov_step_dist(dist_t)
            ctx = (
                numpyro.handlers.reparam(config=dict(reparam_config))
                if reparam_config
                else nullcontext()
            )
            with ctx, _plate_stack(plates):
                z = jnp.asarray(numpyro.sample(site_name, dist_t))
            if advance is not None:
                return advance(carry, z, x_t), z
            return _validate_markov_default_carry(carry, z), z

        return body

    xs_scan = None if xs is None else jax.tree.map(lambda leaf: jnp.moveaxis(leaf, -2, 0), xs)
    final_carry, zs = scan(
        _body(name),
        init_carry,
        None if xs_scan is None else jax.tree.map(lambda leaf: leaf[: h.t_obs], xs_scan),
        length=h.t_obs if xs_scan is None else None,
    )
    if h.future == 0:
        return jnp.moveaxis(zs, 0, -2)
    _, zf = scan(
        _body(f"{name}_future"),
        final_carry,
        None if xs_scan is None else jax.tree.map(lambda leaf: leaf[h.t_obs :], xs_scan),
        length=h.future if xs_scan is None else None,
    )
    return jnp.moveaxis(jnp.concatenate([zs, zf], axis=0), 0, -2)


type SSOEMean[Carry] = Callable[[Carry, PyTree[Array] | None], Float[Array, " *batch obs"]]
"""``(carry, x_t) -> mu_t``: the one-step-ahead mean of the current row, shape
``(*batch, obs)`` (a scalar state emits ``mu[None]``). `ssoe()` owns the error
site: ``mean`` must not call ``numpyro.sample`` (that is `markov_series()`).

``Carry`` is the user's carry type (any PyTree), bound per `ssoe()` call;
``x_t`` is one row of the ``xs`` PyTree (``None`` when ``xs`` is ``None``)."""

type SSOEUpdate[Carry] = Callable[[Carry, Array, Array, PyTree[Array] | None], Carry]
"""``(carry, y_t, eps_t, x_t) -> carry``: the next carry from the row's value and
error. In-sample ``eps_t = y_t - mu_t``; over the horizon ``eps_t`` is the drawn
error and ``y_t = mu_t + eps_t``. An update that needs the mean calls
``mean(carry, x_t)`` again (bit-identical, computed once by XLA) rather than
reconstructing it as ``y_t - eps_t``, which can differ by an ulp. Must preserve
the carry's tree structure, shapes and dtypes. Like ``mean``, it must not call
``numpyro.sample``."""


class SSOEResult(NamedTuple):
    """The means and sampled future values produced by `ssoe()`.

    A named tuple, hence a JAX pytree with three array leaves: it unpacks as
    ``mu, mu_future, y_future = ssoe(...)`` and passes through ``jax.tree.map``,
    ``jax.jit`` and ``jax.vmap`` unchanged.

    Attributes
    ----------
    mu
        In-sample one-step-ahead means, shape ``(*batch, t_obs, obs)``: the
        predictor the caller writes its likelihood against.
    mu_future
        Forecast-horizon one-step-ahead means, shape ``(*batch, future, obs)``
        (a size-0 time axis while training).
    y_future
        Sampled future values ``mu_future + eps``, shape ``(*batch, future, obs)``
        (a size-0 time axis while training); the caller registers them as the
        ``"forecast"`` deterministic when ``h.future > 0``.
    """

    mu: Float[Array, " *batch t_obs obs"]
    mu_future: Float[Array, " *batch future obs"]
    y_future: Float[Array, " *batch future obs"]


def _validate_xs_leaves(h: Horizon, xs: PyTree[Array] | None, block: str, hint: str) -> None:
    """Require every ``xs`` leaf to carry time at axis ``-2`` and span ``h.duration`` rows.

    Shared by `markov_series()` and `ssoe()`, which both split ``xs`` at
    ``h.t_obs``; in `markov_series()` a leaf of another length silently changes
    the number of forecast steps. ``block`` names the caller in the message and
    ``hint`` ends it.
    """
    if xs is None:
        return
    for path, leaf in jax.tree_util.tree_leaves_with_path(xs):
        leaf_name = jax.tree_util.keystr(path) or "<root>"
        if leaf.ndim < 2:
            msg = (
                f"{block} expects every xs leaf to carry time at axis -2 and a trailing "
                f"observation axis; leaf {leaf_name} has shape {leaf.shape}. Add the axis "
                "with [..., None]."
            )
            raise ValueError(msg)
        if leaf.shape[-2] != h.duration:
            msg = (
                f"{block} expects every xs leaf to span the full horizon (duration = "
                f"{h.duration} rows at axis -2); leaf {leaf_name} has {leaf.shape[-2]}. {hint}"
            )
            raise ValueError(msg)


def _validate_ssoe_inputs(h: Horizon, y: Array | None, xs: PyTree[Array] | None) -> Array:
    """Check the driving series and the ``xs`` leaves against the horizon."""
    if y is None:
        msg = (
            "ssoe requires the driving series y: pass the observed series from covariates "
            "(or compute it in the model), never h.data, because predict_in_sample and "
            "to_datatree call the model with data=None."
        )
        raise ValueError(msg)
    if y.ndim < 2:
        msg = (
            "ssoe expects y with time at axis -2 and a trailing observation axis, shape "
            f"(*batch, t_obs, obs); got shape {y.shape}. Slice the series as "
            "covariates[..., :h.t_obs, :] rather than [..., 0]."
        )
        raise ValueError(msg)
    if y.shape[-2] != h.t_obs:
        msg = (
            f"ssoe expects y to cover exactly the observed window: y.shape[-2] = {y.shape[-2]} "
            f"but h.t_obs = {h.t_obs}. Slice the driving series as covariates[..., :h.t_obs, :]."
        )
        raise ValueError(msg)
    _validate_xs_leaves(
        h, xs, "ssoe", "Freeze an in-sample gate over the horizon with pad_future(gate, h.future)."
    )
    return y


def _validate_ssoe_mean(mu_t: Array, y_t: Array) -> None:
    """Require a floating per-step mean shaped exactly like the series rows."""
    if mu_t.ndim == 0 or mu_t.shape != y_t.shape:
        msg = (
            "mean must return a per-step mean shaped exactly like the series rows "
            f"((*batch, obs)); got mean shape {mu_t.shape} for rows of shape {y_t.shape}. "
            "Add the trailing axis with mu[..., None] and broadcast init_carry to the rows "
            "(a wider or narrower mean would silently broadcast the likelihood)."
        )
        raise ValueError(msg)
    if not jnp.issubdtype(mu_t.dtype, jnp.floating):
        msg = (
            f"mean must return a floating per-step mean, got dtype {mu_t.dtype}. Cast the "
            "carry that produces it, e.g. init_carry = y[0].astype(float)."
        )
        raise ValueError(msg)


def _validate_ssoe_carry[Carry](carry: Carry, new_carry: Carry) -> Carry:
    """Require ``update`` to preserve the carry's tree structure, shapes and dtypes."""
    new_carry = jax.tree.map(jnp.asarray, new_carry)
    old_structure = jax.tree.structure(carry)
    new_structure = jax.tree.structure(new_carry)
    if old_structure != new_structure:
        msg = (
            "update must return a carry with the same tree structure as init_carry; got "
            f"{new_structure} instead of {old_structure}."
        )
        raise ValueError(msg)
    old_leaves = jax.tree_util.tree_leaves_with_path(carry)
    for (path, old), new in zip(old_leaves, jax.tree.leaves(new_carry), strict=True):
        old_arr = jnp.asarray(old)
        if old_arr.shape != new.shape or old_arr.dtype != new.dtype:
            leaf_name = jax.tree_util.keystr(path) or "<root>"
            msg = (
                f"update changed carry leaf {leaf_name} from {old_arr.dtype}{old_arr.shape} to "
                f"{new.dtype}{new.shape}; init_carry must already match what update returns "
                "(broadcast it to the (*batch, obs) rows, and align dtypes with e.g. "
                "jax.tree.map(lambda c: c.astype(y.dtype), init_carry))."
            )
            raise ValueError(msg)
    return new_carry


def _future_plate_dim(noise_dist: dist.Distribution) -> int:
    """Return the ``time_future`` plate dim for ``noise_dist`` (``-2``, or ``-1`` for a row event).

    An elementwise family (event rank 0) is batched over ``(*batch, obs)``, so the
    time plate goes at ``dim=-2``; a multivariate family over the observation
    axis (event rank 1, e.g. `numpyro.distributions.MultivariateNormal`) already
    owns the trailing ``obs`` axis as its event, so the plate goes at ``dim=-1``.
    Either way the draw is ``(*batch, future, obs)``.
    """
    event_rank = len(noise_dist.event_shape)
    if event_rank > 1:
        msg = (
            f"noise_dist has event shape {tuple(noise_dist.event_shape)}, event rank "
            f"{event_rank}; ssoe accepts event rank 0 (an elementwise family with batch shape "
            "(obs,)) or 1 (a multivariate family over the observation axis with event shape "
            "(obs,), e.g. MultivariateNormal). Use .to_event(1) on a (*batch, obs)-batched "
            "family, not .to_event(2)."
        )
        raise ValueError(msg)
    if event_rank == 1:
        for handler in _PYRO_STACK:
            if isinstance(handler, numpyro.plate) and handler.dim == -1:
                msg = (
                    "ssoe with a multivariate noise_dist opens time_future at dim=-1, which "
                    f"collides with the enclosing plate {handler.name!r} at dim=-1; batch the "
                    "series to the left (noise batch (B, 1), series (B, t_obs, obs)) instead "
                    "of an enclosing plate at dim=-1."
                )
                raise ValueError(msg)
    return -2 + event_rank


def _validate_future_errors(eps: Array, mu: Array, future: int) -> None:
    """Require the drawn errors to line up exactly with the in-sample means."""
    expected = (*mu.shape[:-2], future, mu.shape[-1])
    if eps.shape != expected:
        msg = (
            f"ssoe expects noise_dist to draw errors of shape {expected} under the time_future "
            f"plate, got {eps.shape}: for a (t_obs, obs) series noise_dist needs batch shape "
            "(obs,) (or () when obs == 1) for an elementwise family, or batch shape () with "
            "event shape (obs,) for a multivariate family such as MultivariateNormal; a "
            "batched (B, t_obs, obs) series needs (B, 1, obs) or (B, 1) respectively."
        )
        raise ValueError(msg)
    if eps.dtype != mu.dtype:
        msg = (
            f"ssoe expects noise_dist to draw errors with the dtype of the means ({mu.dtype}), "
            f"got {eps.dtype}; the forecast scan feeds y_t = mu_t + eps_t back into the carry, "
            "so match the dtypes (e.g. build noise_dist's parameters with y.dtype)."
        )
        raise ValueError(msg)


def ssoe[Carry](
    h: Horizon,
    name: str,
    y: Array | None,
    init_carry: Carry,
    mean: SSOEMean[Carry],
    update: SSOEUpdate[Carry],
    noise_dist: dist.Distribution,
    xs: PyTree[Array] | None = None,
) -> SSOEResult:
    """Run a single-source-of-error recursion over the full horizon.

    The building block for innovations state-space models (ARMA, exponential
    smoothing, Croston/TSB levels, censored autoregressions): a deterministic
    filter whose state is driven by the one-step-ahead *error*
    ``eps_t = y_t - mu_t``. In-sample it runs ``mean`` and ``update`` in a raw
    ``jax.lax.scan`` over the observed series ``y`` (no sample sites inside);
    when forecasting it draws iid future errors at the site
    ``f"{name}_future"`` from ``noise_dist`` under a
    ``plate("time_future", h.future)`` and runs a second scan from the final
    in-sample carry with ``y_t = mu_t + eps_t`` fed back through ``update``.
    The guide never sees the future site, because fitting always happens with
    ``future == 0``. Linear-Gaussian members (ARMA, additive exponential
    smoothing) can be marginalized exactly by a Kalman filter; the
    error-feedback form is the one that also covers the nonlinear members.

    The block registers nothing but the error site. The caller writes the
    likelihood against ``r.mu`` and registers ``numpyro.deterministic("forecast",
    r.y_future)`` when ``h.future > 0`` (an unconditional registration is
    harmless to `~~numpyro_forecast.predictive.forecast()`, but lands a
    size-0 variable in every posterior). Driver contract:
    `~~numpyro_forecast.predictive.predict_in_sample()` and
    `~~numpyro_forecast.convert.to_datatree()` call the model with
    ``data=None`` and read ``"obs"``, so ``y`` must come from ``covariates`` or be
    computed in the model, never from ``h.data``;
    `~~numpyro_forecast.predictive.forecast()` reads ``"forecast"``.

    **Frozen gates.** Route an update gate (Croston's demand indicator, an
    availability mask) through an ``xs`` leaf frozen over the horizon with
    `~~numpyro_forecast.arrays.pad_future()`, and read it from ``x_t``, never
    from ``y_t`` (over the horizon ``y_t = mu_t + eps_t`` is nonzero). With the
    gate off, ``update`` returns the carry unchanged and the forecast is the
    last level plus iid errors. `~~numpyro_forecast.evaluate.backtest()` and
    `~~numpyro_forecast.predictive.forecast()` hand the model *real* future
    covariate rows, so a gate sliced from the full covariates keeps updating on
    sampled values and leaks the test window; scenario inputs such as a future
    availability mask are the only thing to read from those rows.

    **Shapes.** Rows are ``(*batch, obs)``: a scalar state emits ``mu[None]``
    and starts from ``init[None]`` (the block refuses a scalar or a wider mean
    because either would silently broadcast the likelihood into a
    ``(t, t)`` log-prob); a tuple carry with scalar leaves reads ``eps_t[0]``
    (the ETS idiom). A panel puts the series on the observation axis: ``y`` is
    ``(t_obs, series)``, the carry ``(series,)``, and a ``noise`` sampled under
    ``plate("series")`` has exactly the batch shape ``(series,)`` the block
    needs. Batch dims to the left of time (``(B, t_obs, obs)``) take a
    ``(B, obs)`` carry and a ``(B, 1, obs)`` noise batch. Errors correlated
    across the observation axis are a multivariate ``noise_dist`` with event
    shape ``(obs,)``, which is how a vector autoregression enters the block
    (see `~~numpyro_forecast.var.var_step()`). Inputs are jax
    Arrays (the import hook rejects NumPy). With ``obs == 1`` a noise batch
    shape ``(future, 1)`` is indistinguishable from time and is consumed as
    such: per-step error scales, if that is what you meant.

    **Composition.** Two channels are two calls sharing the same ``Horizon``;
    each opens its own ``time_future`` plate. Scoping a helper that contains the
    call is fine (``handlers.scope`` prefixes the error site and the plate; use
    ``name="eps"`` inside a scoped helper so the site reads ``z/eps_future``,
    or ``z_eps_future`` with ``divider="_"``);
    register ``"obs"`` and ``"forecast"`` outside any scope and build the
    forecast from the channels' ``y_future``.

    Parameters
    ----------
    h
        The horizon for the current model call (see `Horizon`).
    name
        Base name of the error site; the future errors are drawn at
        ``f"{name}_future"``.
    y
        The driving series over the observed window, shape
        ``(*batch, t_obs, obs)`` with time at axis ``-2`` (integer counts are
        fine as long as the carry, hence the mean, is floating; the error
        promotes). Sliced from ``covariates`` or computed in the model;
        ``None`` (the value of ``h.data`` under ``data=None``) raises.
    init_carry
        Initial carry, any PyTree, already broadcast to the ``(*batch, obs)``
        rows: a scalar level is ``init[None]``, a panel level ``(series,)``.
        Every leaf must keep its shape and dtype through ``update``.
    mean
        ``(carry, x_t) -> mu_t`` (see `SSOEMean`): the mean for the current row
        (shape ``(*batch, obs)``, so a scalar state emits ``mu[None]``).
    update
        ``(carry, y_t, eps_t, x_t) -> carry`` (see `SSOEUpdate`): the next carry
        from the row's value and error. Over the horizon it receives the drawn
        ``eps_t`` (not a recomputed ``y_t - mu_t``, which can differ by an ulp);
        when the update needs the mean, call ``mean(carry, x_t)`` inside it.
    noise_dist
        Per-step error distribution with location ``0``, either an elementwise family
        (event rank 0) or a multivariate family over the observation axis (event
        rank 1, e.g. ``dist.MultivariateNormal(jnp.zeros(obs), scale_tril=L)``
        for shocks correlated across series, the VAR case). For a
        ``(t_obs, obs)`` series the batch shape is ``(obs,)`` (``()`` is fine
        when ``obs == 1``) for the elementwise form and ``()`` for the
        multivariate form; a batched ``(B, t_obs, obs)`` series takes
        ``(B, 1, obs)`` and ``(B, 1)`` respectively. Either way the draw under
        the time plate (``dim=-2`` for event rank 0, ``dim=-1`` for event rank 1)
        is exactly ``(*batch, future, obs)`` with the dtype of the means; event
        rank 2 or higher is rejected. Over the horizon ``y_t = mu_t + eps_t``,
        so ``mu_t`` is the mean of ``y_t`` only when the errors have mean zero:
        ``Gumbel`` errors with ``loc = 0`` have a nonzero mean, and
        ``AsymmetricLaplace`` errors with ``loc = 0`` have a nonzero mean unless
        ``asymmetry = 1`` (see `~~numpyro_forecast.surgery.shift_loc()` for the
        offsets). That mean shifts each ``y_t`` away from ``mu_t``, and ``update``
        carries the shift into the later means, so the forecast bias can grow
        with the horizon. ``Cauchy`` errors have no mean.
    xs
        Optional exogenous inputs over the full horizon: a PyTree of arrays with
        time at axis ``-2`` and ``duration`` rows (a single array, a tuple, a
        dict, ...), split at ``h.t_obs`` and handed to ``mean`` and ``update``
        row by row as ``x_t``; ``None`` for autonomous dynamics.

    Returns
    -------
    SSOEResult
        ``mu`` (in-sample means), ``mu_future`` and ``y_future`` (forecast
        means and sampled values; size-0 time axis while training).

    Raises
    ------
    ValueError
        If ``y`` is ``None``, lacks the time or observation axis, or does not
        cover exactly ``h.t_obs`` rows; if an ``xs`` leaf lacks the axes or does
        not span ``h.duration`` rows; if ``mean`` returns a value without the
        observation axis or ``update`` a carry with a different tree structure,
        shape or dtype; if ``mean`` or ``update`` calls ``numpyro.sample``; if
        ``noise_dist`` has event rank 2 or higher, or has event rank 1 inside an
        enclosing plate at ``dim=-1``; or if it draws errors of the wrong shape
        or dtype.

    Examples
    --------
    ARMA(1,1) (``y`` is the observed series routed through ``covariates``):

    >>> def mean(carry, _):
    ...     y_prev, eps_prev = carry
    ...     return mu + phi * y_prev + theta * eps_prev
    >>> def update(carry, y_t, eps_t, _):
    ...     return y_t, eps_t
    >>> r = ssoe(h, "eps", y, (mu[None], jnp.zeros((1,))), mean, update, dist.Normal(0.0, sigma))
    >>> numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    >>> if h.future > 0:
    ...     numpyro.deterministic("forecast", r.y_future)

    A gated level (Croston, TSB) with the gate frozen over the horizon:

    >>> def mean(level, _):
    ...     return level
    >>> def update(level, y_t, _, gate_t):
    ...     return jnp.where(gate_t, alpha * y_t + (1 - alpha) * level, level)
    >>> gate_full = pad_future(gate, h.future)
    >>> r = ssoe(h, "eps", y, init[None], mean, update, dist.Normal(0.0, noise), xs=gate_full)
    """
    y = _validate_ssoe_inputs(h, y, xs)
    y_scan = jnp.moveaxis(y, -2, 0)
    xs_scan = None if xs is None else jax.tree.map(lambda leaf: jnp.moveaxis(leaf, -2, 0), xs)
    xs_obs = None if xs_scan is None else jax.tree.map(lambda leaf: leaf[: h.t_obs], xs_scan)
    xs_future = None if xs_scan is None else jax.tree.map(lambda leaf: leaf[h.t_obs :], xs_scan)

    def filter_body(
        carry: Carry, inputs: tuple[Array, PyTree[Array] | None]
    ) -> tuple[Carry, Array]:
        y_t, x_t = inputs
        mu_t = jnp.asarray(mean(carry, x_t))
        _validate_ssoe_mean(mu_t, y_t)
        new_carry = _validate_ssoe_carry(carry, update(carry, y_t, y_t - mu_t, x_t))
        return new_carry, mu_t

    with numpyro.handlers.trace() as filter_trace:
        final_carry, mu_scan = jax.lax.scan(filter_body, init_carry, (y_scan, xs_obs))
    if filter_trace:
        msg = (
            "mean and update must not call numpyro.sample or numpyro.deterministic (found sites: "
            f"{sorted(filter_trace)}); a sampled transition is markov_series."
        )
        raise ValueError(msg)
    mu = jnp.moveaxis(mu_scan, 0, -2)
    if h.future == 0:
        empty = jnp.zeros((*mu.shape[:-2], 0, mu.shape[-1]), mu.dtype)
        return SSOEResult(mu=mu, mu_future=empty, y_future=empty)

    with numpyro.plate(PlateName.TIME_FUTURE, h.future, dim=_future_plate_dim(noise_dist)):
        eps = jnp.asarray(numpyro.sample(f"{name}_future", noise_dist))
    _validate_future_errors(eps, mu, h.future)
    eps_scan = jnp.moveaxis(eps, -2, 0)

    def forecast_body(
        carry: Carry, inputs: tuple[Array, PyTree[Array] | None]
    ) -> tuple[Carry, tuple[Array, Array]]:
        eps_t, x_t = inputs
        mu_t = mean(carry, x_t)
        y_t = mu_t + eps_t
        return update(carry, y_t, eps_t, x_t), (mu_t, y_t)

    _, (mu_future, y_future) = jax.lax.scan(forecast_body, final_carry, (eps_scan, xs_future))
    return SSOEResult(
        mu=mu,
        mu_future=jnp.moveaxis(mu_future, 0, -2),
        y_future=jnp.moveaxis(y_future, 0, -2),
    )


def _has_discrete_support(d: dist.Distribution) -> bool:
    """Whether ``d`` has discrete support, treating an undetermined support as continuous.

    `numpyro.distributions.Distribution` defaults ``support`` to
    ``constraints.dependent``, whose ``is_discrete`` raises ``NotImplementedError``
    when it cannot be determined statically (``ImproperUniform``, or a custom
    subclass that never sets ``support``). Those are not count families, so
    they take the continuous path instead of surfacing that error from `predict()`.
    """
    # numpyro's annotation reaches ty as ``Constraint | None``; no distribution
    # actually sets ``support = None``, so the short-circuit is for the checker.
    support = d.support
    try:
        return support is not None and bool(support.is_discrete)
    except NotImplementedError:
        return False


def _require_predictor_axes(h: Horizon, prediction: Array) -> None:
    """Reject a 1-D predictor of ``h.duration`` steps: time without the observation axis.

    Distributions broadcast from the right, so the ``(duration,)`` result of
    ``covariates @ beta`` puts its time steps on the observation axis of the
    data: ``(time, 1)`` data is scored on a ``(time, time)`` grid and a square
    panel gets a per-series effect, both without an error (issue #166). It is
    checked on the predictor because a ``(duration, 1)`` noise scale gives the
    distribution a valid-looking ``(duration, duration)`` shape and ``data=None``
    leaves nothing to compare against. Every other shape is checked on the
    distribution by `_require_obs_layout()`: a 0-d or per-series predictor is
    fine when a noise parameter carries the time axis.
    """
    shape = tuple(prediction.shape)
    if shape != (h.duration,):
        return
    msg = (
        "predict expects prediction to have shape (*batch, duration, obs), with time at axis "
        f"-2 and the observation axis at -1, but it has shape {shape}: its time steps would "
        "broadcast onto the observation axis of the data. Add the observation axis with "
        "prediction[..., None]: a univariate predictor has shape (duration, 1) and a panel "
        "predictor (duration, n)."
    )
    raise ValueError(msg)


def _require_obs_layout(h: Horizon, full_dist: dist.Distribution) -> None:
    """Require ``(time, obs)`` axes on ``full_dist`` and a log density shaped like ``h.data``.

    The ``"obs"`` site scores ``h.data`` against ``full_dist`` (its first
    ``h.t_obs`` rows while forecasting, as `~~numpyro_forecast.surgery.slice_time()`
    takes them), and the log density broadcasts the two shapes. A shape that
    grows the data counts each observation more than once: a ``(duration,)``
    term added to a ``(duration, 1)`` predictor or used as a noise scale gives a
    ``(time, time)`` grid, a second ``[..., None]`` a ``(time, time, 1)`` one,
    and a predictor with more rows than a single observed step repeats it
    (issue #166). Shapes alone cannot tell a ``(duration,)`` term from a
    per-series one when the data has ``h.duration`` series. ``MultivariateNormal``
    noise is skipped: its event is the time axis, without the observation axis
    (issue #160).
    """
    if isinstance(full_dist, dist.MultivariateNormal):
        return
    shape = (*full_dist.batch_shape, *full_dist.event_shape)
    data = h.data
    width = None if data is None else data.shape[-1]
    if len(shape) < 2:
        problem = "has fewer than two axes, so it has no time axis at -2"
    elif data is None:
        return
    else:
        data_shape = tuple(data.shape)
        rows = min(shape[-2], h.t_obs) if h.future > 0 else shape[-2]
        try:
            grid = tuple(jnp.broadcast_shapes((*shape[:-2], rows, shape[-1]), data_shape))
        except ValueError:
            grid = None
        if grid == data_shape:
            return
        problem = f"does not match the data (shape {data_shape}): " + (
            "the two do not broadcast"
            if grid is None
            else f"the obs site would score a {grid} grid, counting each observation more "
            "than once"
        )
    if shape and shape[-1] == h.duration and shape[-1] not in (1, width):
        hint = (
            f"Its last axis has the h.duration = {h.duration} time steps: give a (duration,) "
            "term (for example covariates @ beta in level + covariates @ beta, or a "
            "time-varying scale) the observation axis with [..., None], and transpose a "
            "(series, time) predictor to (time, series)."
        )
    elif len(shape) < 2:
        hint = (
            "Give the predictor or a parameter of obs_dist the time axis at -2 over the full "
            f"horizon (h.duration = {h.duration}), for example "
            f"jnp.broadcast_to(prediction, ({h.duration}, {'obs' if width is None else width})) "
            "for a constant or per-series predictor."
        )
    else:
        allowed = "1" if width == 1 else f"1 or {width}"
        hint = (
            "The predictor and every parameter of obs_dist need the time axis at -2 over the "
            f"full horizon (h.duration = {h.duration}), an observation axis of size {allowed} "
            "at -1, and no axis that the data does not have (such as a second [..., None])."
        )
    msg = (
        "predict expects the observation distribution to have shape (*batch, duration, obs), "
        f"but its batch_shape + event_shape {shape} {problem}. {hint}"
    )
    raise ValueError(msg)


def predict(
    h: Horizon,
    obs_dist: dist.Distribution | Callable[[Array], dist.Distribution],
    prediction: Array,
) -> None:
    """Register the observation and forecast sites for the model.

    ``prediction`` is the deterministic predictor over the full horizon (time at
    axis ``-2``, shape ``(*batch, duration, obs)``). ``obs_dist`` takes one of
    two forms. A `numpyro.distributions.Distribution` is observation noise with
    location ``0`` (e.g. ``dist.StudentT(nu, 0.0, sigma)``) whose location is
    shifted onto the predictor with `~~numpyro_forecast.surgery.shift_loc()`,
    which also owns the multivariate-normal layout: noise correlated over time,
    ``dist.MultivariateNormal(jnp.zeros(h.duration), cov)``, gets the event shape
    ``(time, 1)``, so ``"obs"`` scores each series jointly and the forecast
    keeps the observation axis. The predictor is then
    the location of the observation distribution: the mean for ``Normal``,
    ``StudentT`` (``df > 1``), ``Laplace`` and ``MultivariateNormal`` noise,
    but the mode for ``Gumbel`` and ``AsymmetricLaplace`` noise, whose mean is
    offset from it (for ``AsymmetricLaplace`` unless ``asymmetry = 1``), and for
    ``Cauchy`` noise, which has no mean (see
    `~~numpyro_forecast.surgery.shift_loc()`). A callable is a link mapping the
    predictor to the observation distribution directly (the GLM form, e.g.
    ``lambda eta: dist.Poisson(jnp.exp(eta))``). Either way, while training the
    observation site ``"obs"`` is observed; while forecasting the in-sample
    prefix is observed and the forecast suffix is sampled at ``"obs_future"``
    and exposed as the ``"forecast"`` deterministic site that
    `~~numpyro_forecast.predictive.forecast()` reads. The observation
    distribution must support time-axis surgery
    (`~~numpyro_forecast.surgery.slice_time()` /
    `~~numpyro_forecast.surgery.prefix_condition()`), i.e. an elementwise
    family or a registered one. This also holds for a distribution wrapped in
    ``Independent`` (e.g. by ``.to_event(1)``).

    Parameters
    ----------
    h
        The horizon for the current model call (see `Horizon`).
    obs_dist
        Noise distribution with location ``0`` (its location is shifted onto
        ``prediction``) or a link callable from the full-horizon predictor to
        the observation distribution.
    prediction
        The deterministic predictor over the full horizon, shape
        ``(*batch, duration, obs)``: ``h.duration`` steps at axis ``-2`` and the
        observation axis at ``-1``, so ``(duration, 1)`` for a univariate series
        and ``(duration, n)`` for a panel of ``n`` series (or ``(duration, 1)``
        inside a size-``n`` plate at ``dim=-1``). A 0-d or per-series
        predictor also works when a parameter of ``obs_dist`` carries the time
        axis, for example a constant mean with a ``(duration, 1)`` noise scale.

    Raises
    ------
    TypeError
        If ``obs_dist`` is a callable that does not return a distribution (for
        example a link that returns the predictor itself).
    RuntimeError
        If forecasting (``future > 0``) but no observed data is available.
    ValueError
        If ``prediction`` has shape ``(duration,)`` (for example
        ``covariates @ beta``, which needs ``[..., None]``); if the observation
        distribution has fewer than two axes; if ``h.data`` is given and the
        observation distribution would score it on a grid larger than
        ``h.data`` (for example a ``(duration,)`` term added to a
        ``(duration, 1)`` predictor, a ``(duration,)`` noise scale or a second
        ``[..., None]``); or if the observation distribution has discrete
        support but ``h.data`` is not integer-dtyped (the usual mistake for
        count models). The shape checks cannot tell a ``(duration,)`` term
        from a per-series one when ``h.data`` has ``h.duration`` series (a
        square panel in training).
    """
    _require_predictor_axes(h, prediction)
    full_dist = (
        shift_loc(obs_dist, prediction)
        if isinstance(obs_dist, dist.Distribution)
        else obs_dist(prediction)
    )
    if not isinstance(full_dist, dist.Distribution):
        msg = (
            "obs_dist must be a noise Distribution with location 0 or a link callable that "
            f"returns a Distribution, got {type(full_dist).__name__} from the link; e.g. "
            "predict(h, dist.Normal(0.0, sigma), mu) or "
            "predict(h, lambda mu: dist.Normal(mu, sigma), mu)."
        )
        raise TypeError(msg)
    _require_obs_layout(h, full_dist)
    if h.data is not None and _has_discrete_support(full_dist):
        if not jnp.issubdtype(h.data.dtype, jnp.integer):
            msg = (
                "the observation distribution has discrete support, so data must "
                f"be integer-dtyped, got dtype {h.data.dtype}. Cast counts with "
                "e.g. data.astype('int32')."
            )
            raise ValueError(msg)
    if h.future == 0:
        numpyro.sample("obs", full_dist, obs=h.data)
        return
    data = h.data
    if data is None:
        msg = "forecasting requires observed data"
        raise RuntimeError(msg)
    prefix = slice_time(full_dist, slice(None, h.t_obs))
    numpyro.sample("obs", prefix, obs=data)
    forecast = numpyro.sample("obs_future", prefix_condition(full_dist, data))
    numpyro.deterministic("forecast", forecast)
