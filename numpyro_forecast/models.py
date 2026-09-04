"""Model building blocks: plain functions that register the train/forecast sites.

The `Horizon` value carries the train/forecast split, and the building
blocks (`innovations()`, `markov_series()`, `ssoe()`, `predict()`)
sample latents and observation sites against it. A model is a plain NumPyro function
``(covariates, data=None) -> None`` whose first line derives its
`Horizon` from the shapes with `Horizon.from_data()` and which then
calls the blocks directly. They are ordinary Python functions that call
``numpyro.sample`` and ``numpyro.deterministic`` on your behalf: not NumPyro
primitives, and not effect handlers.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Float, PyTree
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import Reparam
from numpyro.primitives import _PYRO_STACK

from numpyro_forecast.arrays import _zeros_like_data, concat_future
from numpyro_forecast.surgery import prefix_condition, shift_loc, slice_time
from numpyro_forecast.typing import Array


@dataclass(frozen=True)
class Horizon:
    """The train/forecast split for a single model call.

    An immutable value derived once per model call from the covariate and data
    shapes by `from_data()`; every building block (`innovations()`,
    `markov_series()`, `ssoe()`, `predict()`) takes it as its first
    argument.

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
    t_obs: int
    future: int
    duration: int

    def __post_init__(self) -> None:
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
        return _zeros_like_data(self.data, self.duration)

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
            If ``data`` is longer than ``covariates`` along the time axis.
        """
        duration = covariates.shape[-2]
        t_obs = duration if data is None else data.shape[-2]
        if t_obs > duration:
            msg = "data must not be longer than covariates along the time axis"
            raise ValueError(msg)
        return cls(data=data, t_obs=t_obs, future=duration - t_obs, duration=duration)


def _sample_time_block(
    site: str,
    size: int,
    plate_name: str,
    dist_fn: Callable[[], dist.Distribution],
    reparam: Reparam | None,
) -> Array:
    """Sample a single time block of ``size`` steps under a time plate at axis ``-2``."""
    with ExitStack() as stack:
        if reparam is not None:
            stack.enter_context(numpyro.handlers.reparam(config={site: reparam}))
        stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
        return cast(Array, numpyro.sample(site, dist_fn()))


def innovations(
    h: Horizon,
    name: str,
    dist_fn: Callable[[], dist.Distribution],
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
    dist_fn
        Zero-argument callable returning the per-step prior distribution.
    reparam
        Optional reparameterization (e.g. ``LocScaleReparam``) applied to both
        the in-sample and forecast sites.

    Returns
    -------
    Array
        The latent over the full horizon with time at axis ``-2``.
    """
    prefix = _sample_time_block(name, h.t_obs, "time", dist_fn, reparam)
    if h.future <= 0:
        return prefix
    suffix = _sample_time_block(f"{name}_future", h.future, "time_future", dist_fn, reparam)
    return concat_future(prefix, suffix, axis=-2)


type Transition[Carry] = Callable[
    [Carry, PyTree[Array] | None],
    tuple[dist.Distribution, Callable[[Array], Carry]],
]
"""``(carry, x_t) -> (dist_t, carry_fn)`` where ``carry_fn(z_t)`` builds the next
carry from the *sampled* latent. The wrapper owns the sample statement.

``Carry`` is the user's carry type (any PyTree), bound per `markov_series()`
call; ``x_t`` is one row of the ``xs`` PyTree (``None`` for autonomous dynamics)."""


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


def markov_series[Carry](
    h: Horizon,
    name: str,
    init_carry: Carry,
    transition: Transition[Carry],
    xs: PyTree[Array] | None = None,
    *,
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
        Per-step ``(carry, x_t) -> (dist_t, carry_fn)`` callable; the wrapper
        owns the ``numpyro.sample`` statement.
    xs
        Optional exogenous inputs over the full horizon: a PyTree of arrays
        with time at axis ``-2`` (a single array, a tuple, a dict, ...), moved
        leaf by leaf into scan layout internally; ``None`` for autonomous
        dynamics.
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
        data), if the per-step shape lacks the observation dimension, or if an
        enclosing plate is detected.
    """
    if h.future > 0 and h.data is None:
        msg = "markov_series requires observed data when forecasting"
        raise ValueError(msg)
    _reject_enclosing_plates()

    def _body(site_name: str) -> Callable[[Carry, PyTree[Array] | None], tuple[Carry, Array]]:
        def body(carry: Carry, x_t: PyTree[Array] | None) -> tuple[Carry, Array]:
            dist_t, carry_fn = transition(carry, x_t)
            _validate_markov_step_dist(dist_t)
            ctx = (
                numpyro.handlers.reparam(config=dict(reparam_config))
                if reparam_config
                else nullcontext()
            )
            with ctx, _plate_stack(plates):
                z = cast(Array, numpyro.sample(site_name, dist_t))
            return carry_fn(z), z

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


type SSOEStep[Carry] = Callable[
    [Carry, PyTree[Array] | None],
    tuple[Float[Array, " *batch obs"], Callable[[Array, Array], Carry]],
]
"""``(carry, x_t) -> (mu_t, carry_fn)`` where ``mu_t`` is the one-step-ahead mean
of the current row (shape ``(*batch, obs)``) and ``carry_fn(y_t, eps_t)`` builds
the next carry from the row's value and error. `ssoe()` owns the error
site: ``step`` must not call ``numpyro.sample`` (that is `markov_series()`).

``Carry`` is the user's carry type (any PyTree), bound per `ssoe()` call;
``x_t`` is one row of the ``xs`` PyTree (``None`` when ``xs`` is ``None``)."""


@dataclass(frozen=True)
class SSOEResult:
    """The means and sampled future values produced by `ssoe()`.

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
    if xs is not None:
        for path, leaf in jax.tree_util.tree_leaves_with_path(xs):
            leaf_name = jax.tree_util.keystr(path) or "<root>"
            if leaf.ndim < 2:
                msg = (
                    "ssoe expects every xs leaf to carry time at axis -2 and a trailing "
                    f"observation axis; leaf {leaf_name} has shape {leaf.shape}. Add the axis "
                    "with [..., None]."
                )
                raise ValueError(msg)
            if leaf.shape[-2] != h.duration:
                msg = (
                    "ssoe expects every xs leaf to span the full horizon (duration = "
                    f"{h.duration} rows at axis -2); leaf {leaf_name} has {leaf.shape[-2]}. "
                    "Freeze an in-sample gate over the horizon with pad_future(gate, h.future)."
                )
                raise ValueError(msg)
    return y


def _validate_ssoe_mean(mu_t: Array, y_t: Array) -> None:
    """Require a floating per-step mean shaped exactly like the series rows."""
    if mu_t.ndim == 0 or mu_t.shape != y_t.shape:
        msg = (
            "step must return a per-step mean shaped exactly like the series rows "
            f"((*batch, obs)); got mean shape {mu_t.shape} for rows of shape {y_t.shape}. "
            "Add the trailing axis with mu[..., None] and broadcast init_carry to the rows "
            "(a wider or narrower mean would silently broadcast the likelihood)."
        )
        raise ValueError(msg)
    if not jnp.issubdtype(mu_t.dtype, jnp.floating):
        msg = (
            f"step must return a floating per-step mean, got dtype {mu_t.dtype}. Cast the "
            "carry that produces it, e.g. init_carry = y[0].astype(float)."
        )
        raise ValueError(msg)


def _validate_ssoe_carry[Carry](carry: Carry, new_carry: Carry) -> Carry:
    """Require ``carry_fn`` to preserve the carry's tree structure, shapes and dtypes."""
    new_carry = jax.tree.map(jnp.asarray, new_carry)
    old_structure = jax.tree.structure(carry)
    new_structure = jax.tree.structure(new_carry)
    if old_structure != new_structure:
        msg = (
            "carry_fn must return a carry with the same tree structure as init_carry; got "
            f"{new_structure} instead of {old_structure}."
        )
        raise ValueError(msg)
    old_leaves = jax.tree_util.tree_leaves_with_path(carry)
    for (path, old), new in zip(old_leaves, jax.tree.leaves(new_carry), strict=True):
        old_arr = jnp.asarray(old)
        if old_arr.shape != new.shape or old_arr.dtype != new.dtype:
            leaf_name = jax.tree_util.keystr(path) or "<root>"
            msg = (
                f"carry_fn changed carry leaf {leaf_name} from {old_arr.dtype}{old_arr.shape} to "
                f"{new.dtype}{new.shape}; init_carry must already match what carry_fn returns "
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
    step: SSOEStep[Carry],
    noise_dist: dist.Distribution,
    xs: PyTree[Array] | None = None,
) -> SSOEResult:
    """Run a single-source-of-error recursion over the full horizon.

    The building block for innovations state-space models (ARMA, exponential
    smoothing, Croston/TSB levels, censored autoregressions): a deterministic
    filter whose state is driven by the one-step-ahead *error*
    ``eps_t = y_t - mu_t``. In-sample it runs ``step`` in a raw ``jax.lax.scan``
    over the observed series ``y`` (no sample sites inside); when forecasting it
    draws iid future errors at the site ``f"{name}_future"`` from ``noise_dist``
    under a ``plate("time_future", h.future)`` and runs a second scan from
    the final in-sample carry with ``y_t = mu_t + eps_t`` fed back through
    ``carry_fn``. The guide never sees the future site, because fitting always
    happens with ``future == 0``. Linear-Gaussian members (ARMA, additive
    exponential smoothing) can be marginalized exactly by a Kalman filter; the
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
    gate off, ``carry_fn`` is the identity and the forecast is the last level
    plus iid errors. `~~numpyro_forecast.evaluate.backtest()` and
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
    ``name="eps"`` inside a scoped helper so the site reads ``z_eps_future``);
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
        Every leaf must keep its shape and dtype through ``carry_fn``.
    step
        ``(carry, x_t) -> (mu_t, carry_fn)`` (see `SSOEStep`): ``mu_t`` is
        the mean for the current row (shape ``(*batch, obs)``, so a scalar
        state emits ``mu[None]``) and ``carry_fn(y_t, eps_t)`` the next carry.
        ``carry_fn`` receives the drawn ``eps_t`` over the horizon (not a
        recomputed ``y_t - mu_t``, which can differ by an ulp), so close over
        ``mu_t`` when the update needs it.
    noise_dist
        Zero-centered per-step error distribution, either an elementwise family
        (event rank 0) or a multivariate family over the observation axis (event
        rank 1, e.g. ``dist.MultivariateNormal(jnp.zeros(obs), scale_tril=L)``
        for shocks correlated across series, the VAR case). For a
        ``(t_obs, obs)`` series the batch shape is ``(obs,)`` (``()`` is fine
        when ``obs == 1``) for the elementwise form and ``()`` for the
        multivariate form; a batched ``(B, t_obs, obs)`` series takes
        ``(B, 1, obs)`` and ``(B, 1)`` respectively. Either way the draw under
        the time plate (``dim=-2`` for event rank 0, ``dim=-1`` for event rank 1)
        is exactly ``(*batch, future, obs)`` with the dtype of the means; event
        rank 2 or higher is rejected.
    xs
        Optional exogenous inputs over the full horizon: a PyTree of arrays with
        time at axis ``-2`` and ``duration`` rows (a single array, a tuple, a
        dict, ...), split at ``h.t_obs`` and handed to ``step`` row by row as
        ``x_t``; ``None`` for autonomous dynamics.

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
        not span ``h.duration`` rows; if ``step`` returns a mean without the
        observation axis or a carry with a different tree structure, shape or
        dtype; if ``step`` calls ``numpyro.sample``; if ``noise_dist`` has event
        rank 2 or higher, or has event rank 1 inside an enclosing plate at
        ``dim=-1``; or if it draws errors of the wrong shape or dtype.

    Examples
    --------
    ARMA(1,1) with the lambda form of ``carry_fn`` (``y`` is the observed series
    routed through ``covariates``):

    >>> def step(carry, _):
    ...     y_prev, eps_prev = carry
    ...     mu_t = mu + phi * y_prev + theta * eps_prev
    ...     return mu_t, lambda y_t, eps_t: (y_t, eps_t)
    >>> r = ssoe(h, "eps", y, (mu[None], jnp.zeros((1,))), step, dist.Normal(0.0, sigma))
    >>> numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    >>> if h.future > 0:
    ...     numpyro.deterministic("forecast", r.y_future)

    A gated level (Croston, TSB) with the gate frozen over the horizon:

    >>> def step(level, gate_t):
    ...     def carry_fn(y_t, _):
    ...         return jnp.where(gate_t, alpha * y_t + (1 - alpha) * level, level)
    ...
    ...     return level, carry_fn
    >>> gate_full = pad_future(gate, h.future)
    >>> r = ssoe(h, "eps", y, init[None], step, dist.Normal(0.0, noise), xs=gate_full)
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
        mu_raw, carry_fn = step(carry, x_t)
        mu_t = jnp.asarray(mu_raw)
        _validate_ssoe_mean(mu_t, y_t)
        new_carry = _validate_ssoe_carry(carry, carry_fn(y_t, y_t - mu_t))
        return new_carry, mu_t

    with numpyro.handlers.trace() as filter_trace:
        final_carry, mu_scan = jax.lax.scan(filter_body, init_carry, (y_scan, xs_obs))
    if filter_trace:
        msg = (
            "step must not call numpyro.sample or numpyro.deterministic (found sites: "
            f"{sorted(filter_trace)}); a sampled transition is markov_series."
        )
        raise ValueError(msg)
    mu = jnp.moveaxis(mu_scan, 0, -2)
    if h.future == 0:
        empty = jnp.zeros((*mu.shape[:-2], 0, mu.shape[-1]), mu.dtype)
        return SSOEResult(mu=mu, mu_future=empty, y_future=empty)

    with numpyro.plate("time_future", h.future, dim=_future_plate_dim(noise_dist)):
        eps = cast(Array, numpyro.sample(f"{name}_future", noise_dist))
    _validate_future_errors(eps, mu, h.future)
    eps_scan = jnp.moveaxis(eps, -2, 0)

    def forecast_body(
        carry: Carry, inputs: tuple[Array, PyTree[Array] | None]
    ) -> tuple[Carry, tuple[Array, Array]]:
        eps_t, x_t = inputs
        mu_t, carry_fn = step(carry, x_t)
        y_t = mu_t + eps_t
        return carry_fn(y_t, eps_t), (mu_t, y_t)

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


def predict(
    h: Horizon,
    obs_dist: dist.Distribution | Callable[[Array], dist.Distribution],
    prediction: Array,
) -> None:
    """Register the observation and forecast sites for the model.

    ``prediction`` is the deterministic predictor over the full horizon (time at
    axis ``-2``, shape ``(*batch, duration, obs)``). ``obs_dist`` takes one of
    two forms. A `numpyro.distributions.Distribution` is zero-centered
    observation noise (e.g. ``dist.StudentT(nu, 0.0, sigma)``) shifted onto the
    predictor with `~~numpyro_forecast.surgery.shift_loc()`, which also owns
    the multivariate-normal layout check. A callable is a link mapping the
    predictor to the observation distribution directly (the GLM form, e.g.
    ``lambda eta: dist.Poisson(jnp.exp(eta))``). Either way, while training the
    observation site ``"obs"`` is observed; while forecasting the in-sample
    prefix is observed and the forecast suffix is sampled at ``"obs_future"``
    and exposed as the ``"forecast"`` deterministic site that
    `~~numpyro_forecast.predictive.forecast()` reads. The observation
    distribution must support time-axis surgery
    (`~~numpyro_forecast.surgery.slice_time()` /
    `~~numpyro_forecast.surgery.prefix_condition()`), i.e. an elementwise
    family or a registered one.

    Parameters
    ----------
    h
        The horizon for the current model call (see `Horizon`).
    obs_dist
        Zero-centered noise distribution (shifted onto ``prediction``) or a
        link callable from the full-horizon predictor to the observation
        distribution.
    prediction
        The deterministic predictor over the full horizon, time at axis ``-2``.

    Raises
    ------
    TypeError
        If ``obs_dist`` is a callable that does not return a distribution (for
        example a link that returns the predictor itself).
    RuntimeError
        If forecasting (``future > 0``) but no observed data is available.
    ValueError
        If the observation distribution has discrete support but ``h.data`` is
        not integer-dtyped (the usual mistake for count models).
    """
    full_dist = (
        shift_loc(obs_dist, prediction)
        if isinstance(obs_dist, dist.Distribution)
        else obs_dist(prediction)
    )
    if not isinstance(full_dist, dist.Distribution):
        msg = (
            "obs_dist must be a zero-centered noise Distribution or a link callable that "
            f"returns a Distribution, got {type(full_dist).__name__} from the link; e.g. "
            "predict(h, dist.Normal(0.0, sigma), mu) or "
            "predict(h, lambda mu: dist.Normal(mu, sigma), mu)."
        )
        raise TypeError(msg)
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
