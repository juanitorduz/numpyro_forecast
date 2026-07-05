"""Functional forecasting API: the pure core shared with the OOP classes.

This module is the functional counterpart of :mod:`numpyro_forecast.forecaster`.
Where the OOP API carries the train/forecast split as mutable state on a
:class:`~numpyro_forecast.forecaster.ForecastingModel` instance, here it is an
explicit, immutable :class:`Horizon` value threaded into the primitives. The
class-based API is a thin shim over the functions defined here, so the two
styles are fully interchangeable: both produce a NumPyro model callable
``(covariates, data=None)`` and consume a posterior dict of latent draws.
"""

import functools
import inspect
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial, singledispatch
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.tree_util import tree_map
from jaxtyping import Num
from numpyro.contrib.control_flow import scan
from numpyro.infer import AIES, ESS, MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoGuide, AutoNormal
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.reparam import Reparam
from numpyro.optim import Adam, _NumPyroOptim

from numpyro_forecast.typing import (
    Array,
    ForecastModel,
    GuideLike,
    KernelLike,
    OptimizerLike,
)
from numpyro_forecast.util import (
    _zeros_like_data,
    concat_future,
    prefix_condition,
    shift_loc,
    slice_time,
)


@dataclass(frozen=True)
class Horizon:
    """The train/forecast split for a single model call.

    Replaces the mutable ``self._*`` state of the OOP base class with an
    immutable value derived from the covariate and data shapes via
    :meth:`from_data`. The functional primitives (:func:`time_series`,
    :func:`predict`) take it as their first argument.

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

        Mirrors Pyro's ``zero_data`` (and :func:`numpyro_forecast.util.zero_data_like`):
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
        """Derive the horizon from covariate and data shapes.

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


def time_series(
    h: Horizon,
    name: str,
    dist_fn: Callable[[], dist.Distribution],
    *,
    reparam: Reparam | None = None,
) -> Array:
    """Sample a time-varying latent over the full horizon.

    The in-sample portion is sampled under ``plate("time", t)`` with the fixed
    site ``name``; when forecasting, the horizon portion is sampled under a
    separate site ``f"{name}_future"`` and concatenated. The separate site keeps
    the guide shape fixed and lets ``Predictive`` draw the forecast suffix from
    the prior.

    Parameters
    ----------
    h
        The horizon for the current model call (see :class:`Horizon`).
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


Transition = Callable[
    [Any, Array | None],
    tuple[dist.Distribution, Callable[[Array], Any]],
]
"""(carry, x_t) -> (dist_t, carry_fn) where carry_fn(z_t) builds the next carry
from the *sampled* latent. The wrapper owns the sample statement.

``carry`` is an arbitrary PyTree (hence ``Any``): typing it as ``object`` would,
by function-parameter contravariance, reject every concretely-typed transition a
user might write (e.g. ``carry: Array``)."""


@contextmanager
def _plate_stack(plates: Sequence[tuple[str, int]]):
    """Open nested ``numpyro.plate`` contexts (innermost last)."""
    with ExitStack() as stack:
        for plate_name, size in plates:
            stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
        yield


def _reject_enclosing_plates() -> None:
    """Raise if ``markov_time_series`` is called inside an enclosing plate."""
    try:
        from numpyro.primitives import _PYRO_STACK

        for msg in _PYRO_STACK:
            if type(msg).__name__ == "plate":
                msg_text = (
                    "markov_time_series opens plates internally via the plates= "
                    "argument; do not wrap the call in an enclosing numpyro.plate."
                )
                raise ValueError(msg_text)
    except (ImportError, AttributeError, TypeError):
        pass


def _validate_markov_step_dist(dist_t: dist.Distribution) -> None:
    """Require a non-degenerate per-step shape with an observation axis (C7)."""
    if len(dist_t.event_shape) == 0 and len(dist_t.batch_shape) == 0:
        msg = (
            "markov_time_series requires the transition distribution to carry "
            "the trailing observation dimension; add it, e.g. loc=...[..., None]."
        )
        raise ValueError(msg)


def markov_time_series(
    h: Horizon,
    name: str,
    init_carry: Any,
    transition: Transition,
    xs: Array | None = None,
    *,
    plates: Sequence[tuple[str, int]] = (),
    reparam_config: Mapping[str, Reparam] | None = None,
) -> Array:
    """Sample a Markov (state-space) latent over the full horizon.

    In-sample steps run in a ``numpyro.contrib.control_flow.scan`` with site
    ``name``; when forecasting, horizon steps run in a second scan with site
    ``f"{name}_future"`` **seeded by the final in-sample carry**. The guide
    never sees the future site (same invariant as :func:`time_series`), and
    under posterior replay the carry is a deterministic function of the replayed
    draws, so the forecast is conditioned through the state.

    Parameters
    ----------
    h
        The horizon for the current model call (see :class:`Horizon`).
    name
        Base sample-site name for the in-sample latent scan.
    init_carry
        Initial carry passed to the first transition.
    transition
        Per-step ``(carry, x_t) -> (dist_t, carry_fn)`` callable; the wrapper
        owns the ``numpyro.sample`` statement.
    xs
        Optional exogenous inputs over the full horizon with time at axis ``-2``,
        moved into scan layout internally; ``None`` for autonomous dynamics.
    plates
        ``(name, size)`` pairs opened **inside** the scan body around the sample
        statement (the only placement NumPyro supports for scan + plate).
    reparam_config
        Site-name -> :class:`~numpyro.infer.reparam.Reparam` mapping applied
        **inside** the scan body.

    Returns
    -------
    Array
        The latent over the full horizon in package layout
        ``(*plate_batch, duration, obs)``.

    Raises
    ------
    ValueError
        If forecasting without observed data, if the per-step shape lacks the
        observation dimension (C7), or if an enclosing plate is detected.
    """
    if h.future > 0 and h.data is None:
        msg = "markov_time_series requires observed data when forecasting"
        raise ValueError(msg)
    _reject_enclosing_plates()

    def _body(site_name: str) -> Callable[[object, Array | None], tuple[object, Array]]:
        def body(carry: object, x_t: Array | None) -> tuple[object, Array]:
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

    xs_scan = None if xs is None else jnp.moveaxis(xs, -2, 0)
    final_carry, zs = scan(
        _body(name),
        init_carry,
        None if xs_scan is None else xs_scan[: h.t_obs],
        length=h.t_obs if xs_scan is None else None,
    )
    if h.future == 0:
        return jnp.moveaxis(zs, 0, -2)
    _, zf = scan(
        _body(f"{name}_future"),
        final_carry,
        None if xs_scan is None else xs_scan[h.t_obs :],
        length=h.future if xs_scan is None else None,
    )
    return jnp.moveaxis(jnp.concatenate([zs, zf], axis=0), 0, -2)


def predict_glm(
    h: Horizon,
    obs_dist_fn: Callable[[Array], dist.Distribution],
    latent: Array,
) -> None:
    """Register GLM-style observation/forecast sites from a latent predictor.

    The generalized-linear counterpart of :func:`predict`: instead of a
    zero-centered noise distribution shifted by a mean, the caller supplies a link
    ``obs_dist_fn`` that maps the full-horizon ``latent`` predictor to the
    observation distribution directly (e.g. ``lambda eta: Poisson(jnp.exp(eta))``).
    The prefix/suffix mirroring is identical to :func:`predict`: while training the
    observation is observed; while forecasting the in-sample prefix is observed and
    the forecast suffix is sampled and exposed as the ``"forecast"`` deterministic
    site. The observation distribution must support time-axis surgery
    (:func:`~numpyro_forecast.util.slice_time` /
    :func:`~numpyro_forecast.util.prefix_condition`), i.e. an elementwise family.

    Parameters
    ----------
    h
        The horizon for the current model call (see :class:`Horizon`).
    obs_dist_fn
        Link mapping the full-horizon ``latent`` to the observation distribution
        (time at axis ``-2``, shape ``(*batch, duration, obs)``).
    latent
        The deterministic latent predictor over the full horizon, time at axis
        ``-2``.

    Raises
    ------
    RuntimeError
        If forecasting (``future > 0``) but no observed data is available.
    ValueError
        If the observation distribution has discrete support but ``h.data`` is
        not integer-dtyped (the usual mistake for count models).
    """
    obs_dist = obs_dist_fn(latent)
    support = obs_dist.support
    if support is not None and support.is_discrete and h.data is not None:
        if not jnp.issubdtype(h.data.dtype, jnp.integer):
            msg = (
                "the observation distribution has discrete support, so data must "
                f"be integer-dtyped, got dtype {h.data.dtype}. Cast counts with "
                "e.g. data.astype('int32')."
            )
            raise ValueError(msg)
    if h.future == 0:
        numpyro.sample("obs", obs_dist, obs=h.data)
        return
    data = h.data
    if data is None:
        msg = "forecasting requires observed data"
        raise RuntimeError(msg)
    prefix = slice_time(obs_dist, slice(None, h.t_obs))
    numpyro.sample("obs", prefix, obs=data)
    forecast = numpyro.sample("obs_future", prefix_condition(obs_dist, data))
    numpyro.deterministic("forecast", forecast)


def predict(h: Horizon, noise_dist: dist.Distribution, prediction: Array) -> None:
    """Register the observation/forecast sites for the model.

    ``noise_dist`` is a zero-centered observation noise distribution and
    ``prediction`` the deterministic mean over the full horizon. While training
    the residual is observed; while forecasting the in-sample prefix is observed
    and the forecast suffix is sampled and exposed as the ``"forecast"``
    deterministic site. A thin wrapper over :func:`predict_glm` with the
    location-shift link ``lambda mu: shift_loc(noise_dist, mu)``.

    Parameters
    ----------
    h
        The horizon for the current model call (see :class:`Horizon`).
    noise_dist
        Zero-centered observation noise (e.g. ``Normal(0, sigma)``).
    prediction
        Deterministic mean with time at axis ``-2``, shape
        ``(*batch, duration, obs)``.

    Raises
    ------
    RuntimeError
        If forecasting (``future > 0``) but no observed data is available.
    """
    predict_glm(h, lambda mu: shift_loc(noise_dist, mu), prediction)


def forecasting_model(model_fn: Callable[[Horizon, Array], None]) -> ForecastModel:
    """Build a NumPyro model from a functional model body.

    The functional analogue of subclassing
    :class:`~numpyro_forecast.forecaster.ForecastingModel`. ``model_fn`` is a
    pure function ``(Horizon, covariates) -> None`` that calls :func:`time_series`
    and :func:`predict`; this wraps it into the standard NumPyro model callable
    ``(covariates, data=None)``, deriving the :class:`Horizon` from the shapes.

    Parameters
    ----------
    model_fn
        The model body. It receives the per-call :class:`Horizon` (use
        ``h.zero_data`` for the Pyro-style ``zero_data``) and the covariates with
        time at axis ``-2``.

    Returns
    -------
    ForecastModel
        A callable ``(covariates, data=None) -> None`` usable with ``SVI``,
        ``MCMC``, ``Predictive``, :func:`fit_svi`, :func:`fit_mcmc`, and the
        OOP forecaster classes.
    """

    def numpyro_model(covariates: Array, data: Array | None = None) -> None:
        model_fn(Horizon.from_data(covariates, data), covariates)

    return numpyro_model


_DEFAULT_LEARNING_RATE: float = 0.01
"""Default Adam learning rate used when ``optim`` is ``None``."""


def resolve_optimizer(optim: "OptimizerLike") -> _NumPyroOptim:
    """Normalize an optimizer specification into a NumPyro optimizer.

    Accepted forms: ``None`` (``Adam(0.01)``); a finite positive scalar learning
    rate (``float``/``int``/NumPy scalar/0-d array) giving ``Adam(lr)``; an
    ``optax.GradientTransformation`` (wrapped via
    ``numpyro.optim.optax_to_numpyro``, imported lazily so optax stays a soft
    dependency); a ``_NumPyroOptim`` (returned unchanged).

    Parameters
    ----------
    optim
        The optimizer specification (see :data:`~numpyro_forecast.typing.OptimizerLike`).

    Returns
    -------
    _NumPyroOptim
        The resolved NumPyro optimizer.

    Raises
    ------
    TypeError
        For ``bool`` (``bool`` is an ``int`` subclass, so it would silently mean
        ``Adam(1.0)``) and for any other unrecognized type; the message lists
        the accepted forms.
    ValueError
        For a non-finite or non-positive learning rate.
    """
    if optim is None:
        return Adam(_DEFAULT_LEARNING_RATE)
    if isinstance(optim, _NumPyroOptim):
        return optim
    if isinstance(optim, bool):
        msg = (
            "resolve_optimizer() does not accept bool; pass a positive float "
            "learning rate, an optax.GradientTransformation, a numpyro optimizer, "
            "or None."
        )
        raise TypeError(msg)
    is_array_scalar = getattr(optim, "ndim", None) == 0
    if isinstance(optim, (int, float)) or is_array_scalar:
        lr = float(cast("float", optim))
        if not math.isfinite(lr) or lr <= 0.0:
            msg = f"learning rate must be finite and positive, got {lr}"
            raise ValueError(msg)
        return Adam(lr)
    if hasattr(optim, "init") and hasattr(optim, "update"):
        from numpyro.optim import optax_to_numpyro

        return optax_to_numpyro(optim)
    msg = (
        f"resolve_optimizer() does not support {type(optim).__name__}; pass None, "
        "a positive float learning rate, an optax.GradientTransformation, or a "
        "numpyro optimizer (_NumPyroOptim)."
    )
    raise TypeError(msg)


_HANDWRITTEN_GUIDE_FACTORY_MSG = (
    "resolve_guide() received a callable taking a single required positional "
    "argument and no defaults. This is ambiguous: it looks like a guide *factory* "
    "(e.g. `lambda model: AutoNormal(model)`), which must be passed as the class "
    "or a functools.partial of it, not a lambda; or it is a hand-written guide, "
    "which must use the model signature `(covariates, data=None)`."
)

_HANDWRITTEN_GUIDE_NEEDS_ARGS_MSG = (
    "drawing from a hand-written guide requires the in-sample covariates/data, "
    "which this SVIFit was constructed without. Fit via fit_svi (which records "
    "them) or provide them explicitly."
)


def _probe_handwritten_guide(guide: Callable[..., object]) -> None:
    """Reject callables whose signature matches a guide *factory*.

    Uses :func:`inspect.signature`: exactly one required positional parameter,
    no defaults, and no var-args raises :class:`TypeError` with the
    dual-interpretation message. Signatures :mod:`inspect` cannot resolve
    (builtins, some partials) pass the probe: it is a tripwire for the common
    mistyped-factory mistake, not a gatekeeper.

    Parameters
    ----------
    guide
        The candidate hand-written guide callable.

    Raises
    ------
    TypeError
        If ``guide`` has the single-required-positional-argument factory shape.
    """
    try:
        sig = inspect.signature(guide)
    except (ValueError, TypeError):
        return
    params = list(sig.parameters.values())
    required_positional = [
        p
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    has_default = any(p.default is not p.empty for p in params)
    has_var = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)
    if len(required_positional) == 1 and not has_default and not has_var:
        raise TypeError(_HANDWRITTEN_GUIDE_FACTORY_MSG)


def resolve_guide(
    guide: "GuideLike",
    model: ForecastModel,
) -> "AutoGuide | Callable[..., None]":
    """Normalize a guide specification against ``model``.

    Resolution: ``None`` -> ``AutoNormal(model)``; an ``AutoGuide`` instance ->
    returned unchanged; an ``AutoGuide`` subclass or a ``functools.partial`` of
    one -> called with ``model``; any other callable -> a hand-written guide,
    after :func:`_probe_handwritten_guide`. Anything else -> ``TypeError``.

    Parameters
    ----------
    guide
        The guide specification (see :data:`~numpyro_forecast.typing.GuideLike`).
    model
        The model the guide is built against.

    Returns
    -------
    AutoGuide | Callable[..., None]
        The resolved guide.

    Raises
    ------
    TypeError
        If ``guide`` is neither an ``AutoGuide`` (instance/subclass/partial) nor
        a callable, or if it has the mistyped-factory shape.
    """
    if guide is None:
        return AutoNormal(model)
    if isinstance(guide, AutoGuide):
        return guide
    if isinstance(guide, type) and issubclass(guide, AutoGuide):
        return guide(model)
    if isinstance(guide, functools.partial):
        target = guide.func
        if isinstance(target, type) and issubclass(target, AutoGuide):
            return cast("AutoGuide", guide(model))
    if callable(guide):
        _probe_handwritten_guide(guide)
        return cast("Callable[..., None]", guide)
    msg = (
        f"resolve_guide() does not support {type(guide).__name__}; pass None, an "
        "AutoGuide instance, an AutoGuide subclass (or functools.partial of one), "
        "or a hand-written guide function `(covariates, data=None)`."
    )
    raise TypeError(msg)


def _ensure_sample_axis_for_delta(samples: dict[str, Array], num_samples: int) -> dict[str, Array]:
    """Tile ``AutoDelta`` point estimates to a leading sample axis.

    Called only when the fit's guide is an ``AutoDelta`` (guide-type dispatch,
    never shape inspection): every leaf is broadcast to
    ``(num_samples, *leaf.shape)`` unconditionally. For all other Auto guides
    ``sample_posterior`` already returns the axis and this is never invoked.

    Parameters
    ----------
    samples
        The MAP point estimate, one leaf per latent site.
    num_samples
        The leading sample-axis size to tile to.

    Returns
    -------
    dict[str, Array]
        Each leaf broadcast to ``(num_samples, *leaf.shape)``.
    """
    return {
        name: jnp.broadcast_to(leaf, (num_samples, *jnp.shape(leaf)))
        for name, leaf in samples.items()
    }


@dataclass(frozen=True)
class SVIFit:
    """The result of fitting a forecasting model with SVI.

    Attributes
    ----------
    guide
        The fitted variational guide.
    params
        The learned variational parameters.
    losses
        The ELBO loss per SVI step (shape ``(num_steps,)``).
    data
        The in-sample data the model was fit on (needed to draw from
        hand-written guides and by :func:`~numpyro_forecast.convert.to_datatree`).
        ``None`` for fits constructed without it.
    covariates
        The in-sample covariates the model was fit on. ``None`` for fits
        constructed without it.
    """

    guide: "AutoGuide | Callable[..., None]"
    params: dict[str, Array]
    losses: Array
    data: Array | None = None
    covariates: Array | None = None


def _require_positive_num_samples(num_samples: int) -> None:
    """Raise ``ValueError`` if ``num_samples`` is not positive."""
    if num_samples <= 0:
        msg = "num_samples must be positive"
        raise ValueError(msg)


def _require_equal_duration(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` if ``data`` and ``covariates`` differ in duration."""
    if data.shape[-2] != covariates.shape[-2]:
        msg = "fit expects data and covariates of equal duration"
        raise ValueError(msg)


def _require_covariates_extend_data(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` unless ``covariates`` is longer than ``data`` in time."""
    if data.shape[-2] >= covariates.shape[-2]:
        msg = "covariates must extend beyond data along the time axis"
        raise ValueError(msg)


def _index_tree(tree: dict[str, Array], index: Array | slice) -> dict[str, Array]:
    """Index every leaf of a posterior-sample pytree along its sample axis."""
    return tree_map(lambda leaf: leaf[index], tree)


def _pad_posterior(posterior: dict[str, Array], batch_size: int) -> tuple[dict[str, Array], int]:
    """Pad a posterior's sample axis up to a whole multiple of ``batch_size``.

    Padding lets the chunk loop slice fixed-size ``batch_size`` blocks so
    ``_predict`` compiles exactly once regardless of ``num_samples`` (invariant
    I3). The pad rows are wrapped-around copies of existing draws and are
    discarded by the caller's final ``[:num]`` slice, so they never affect the
    result.

    Parameters
    ----------
    posterior
        Posterior samples, sample axis leading (all leaves agree on that axis).
    batch_size
        Positive chunk size.

    Returns
    -------
    tuple[dict[str, Array], int]
        The padded posterior and the original (pre-pad) sample count.

    Raises
    ------
    ValueError
        If ``posterior`` is empty, ``batch_size`` is not positive, or the leaves
        disagree on the sample-axis length (the message names the offending sites).
    """
    if not posterior:
        msg = "_pad_posterior() requires a non-empty posterior"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    sizes = {name: leaf.shape[0] for name, leaf in posterior.items()}
    distinct = set(sizes.values())
    if len(distinct) != 1:
        msg = f"posterior leaves disagree on the sample axis: {sizes}"
        raise ValueError(msg)
    num = distinct.pop()
    remainder = num % batch_size
    if remainder == 0:
        return posterior, num
    pad = batch_size - remainder
    pad_indices = jnp.arange(pad) % num
    padded = {
        name: jnp.concatenate([leaf, leaf[pad_indices]], axis=0)
        for name, leaf in posterior.items()
    }
    return padded, num


def fit_svi(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    guide: "GuideLike" = None,
    optim: "OptimizerLike" = None,
    num_steps: int = 1_001,
    num_particles: int = 1,
    progress_bar: bool = False,
    stable_update: bool = False,
) -> SVIFit:
    """Fit a forecasting model with stochastic variational inference.

    PRNG: consumes ``rng_key`` once for the SVI run; nothing is retained.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    guide
        Guide specification resolved by :func:`resolve_guide`: ``None``
        (``AutoNormal``), an ``AutoGuide`` instance, an ``AutoGuide`` subclass or
        ``functools.partial`` factory of one, or a hand-written guide function.
    optim
        Optimizer specification resolved by :func:`resolve_optimizer`: ``None``
        (``Adam(0.01)``), a positive scalar learning rate, an
        ``optax.GradientTransformation``, or a ``_NumPyroOptim``. For example, a
        cosine-decayed, gradient-clipped Adam::

            import optax

            schedule = optax.cosine_decay_schedule(1e-2, decay_steps=1_000)
            optim = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))

    num_steps
        Number of SVI steps.
    num_particles
        Number of ELBO particles.
    progress_bar
        Whether to display the SVI progress bar.
    stable_update
        Whether SVI skips parameter updates whose new value is non-finite
        (NumPyro's ``stable_update``).

    Returns
    -------
    SVIFit
        The fitted guide, variational parameters, loss history, and the
        in-sample ``data``/``covariates`` (kept by identity, not copied).

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` have different durations.
    """
    _require_equal_duration(data, covariates)
    resolved_guide = resolve_guide(guide, model)
    optimizer = resolve_optimizer(optim)
    svi = SVI(model, resolved_guide, optimizer, Trace_ELBO(num_particles=num_particles))
    result = svi.run(
        rng_key,
        num_steps,
        covariates,
        data,
        progress_bar=progress_bar,
        stable_update=stable_update,
    )
    return SVIFit(
        guide=resolved_guide,
        params=result.params,
        losses=result.losses,
        data=data,
        covariates=covariates,
    )


@singledispatch
def _draw_posterior_impl(fit: object, num_samples: int, rng_key: Array) -> dict[str, Array]:
    """Dispatch on the fit type to draw posterior samples (see :func:`draw_posterior`)."""
    msg = f"draw_posterior() does not support {type(fit).__name__}"
    raise NotImplementedError(msg)


@_draw_posterior_impl.register
def _(fit: SVIFit, num_samples: int, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    if isinstance(fit.guide, AutoDelta):
        point = fit.guide.sample_posterior(rng_key, fit.params)
        return _ensure_sample_axis_for_delta(point, num_samples)
    if isinstance(fit.guide, AutoGuide):
        return fit.guide.sample_posterior(rng_key, fit.params, sample_shape=(num_samples,))
    if fit.covariates is None:
        raise ValueError(_HANDWRITTEN_GUIDE_NEEDS_ARGS_MSG)
    predictive = Predictive(fit.guide, params=fit.params, num_samples=num_samples)
    return predictive(rng_key, fit.covariates, fit.data)


def draw_posterior(rng_key: Array, fit: object, num_samples: int) -> dict[str, Array]:
    """Draw ``num_samples`` posterior samples of the latent sites from a fit.

    Dispatches on the fit type (e.g. :class:`SVIFit`, :class:`MCMCFit`). The
    returned dict has the sample axis leading and is ready to pass to
    :func:`forecast` or NumPyro's ``Predictive``.

    Parameters
    ----------
    rng_key
        PRNG key.
    fit
        A fit result produced by :func:`fit_svi` or :func:`fit_mcmc`.
    num_samples
        Number of posterior draws.

    Returns
    -------
    dict[str, Array]
        Posterior samples of the latent sites, sample axis leading.

    Raises
    ------
    NotImplementedError
        If ``fit`` is of an unsupported type.

    Notes
    -----
    For an :class:`MCMCFit`, when ``num_samples`` does not exceed the number of
    draws in the chain the draws are thinned on an evenly spaced grid (no
    duplicates); only when more samples are requested than the chain holds are
    they resampled with replacement. For an :class:`SVIFit` the draws are sampled
    afresh from the fitted guide.
    """
    return _draw_posterior_impl(fit, num_samples, rng_key)


@dataclass(frozen=True)
class MCMCFit:
    """The result of fitting a forecasting model with MCMC.

    Attributes
    ----------
    samples
        The posterior samples of the latent sites, sample axis leading and
        stored flattened (``group_by_chain=False``).
    num_chains
        The number of chains the fit was run with (frozen; default ``1``), so
        chain structure survives into :func:`~numpyro_forecast.convert.to_datatree`.
    """

    samples: dict[str, Array]
    num_chains: int = 1


def resolve_kernel(
    kernel: "KernelLike",
    model: ForecastModel,
    kernel_kwargs: Mapping[str, Any] | None,
) -> MCMCKernel:
    """Normalize a kernel specification.

    ``None`` -> ``NUTS(model, **kernel_kwargs)`` (kwargs tune the default kernel
    without naming it); a kernel class -> ``kernel(model, **kernel_kwargs)``; a
    kernel instance -> returned unchanged, and combining an instance with
    non-empty ``kernel_kwargs`` raises ``ValueError`` (ambiguous). Anything else
    -> ``TypeError``.

    Parameters
    ----------
    kernel
        The kernel specification (see :data:`~numpyro_forecast.typing.KernelLike`).
    model
        The model the kernel is built against.
    kernel_kwargs
        Extra keyword arguments forwarded to the kernel constructor (ignored,
        and rejected, for an already-constructed instance).

    Returns
    -------
    MCMCKernel
        The resolved kernel.

    Raises
    ------
    ValueError
        If a kernel instance is combined with non-empty ``kernel_kwargs``.
    TypeError
        If ``kernel`` is neither ``None``, an ``MCMCKernel`` subclass, nor an
        ``MCMCKernel`` instance.
    """
    kwargs = dict(kernel_kwargs or {})
    if kernel is None:
        return NUTS(model, **kwargs)
    if isinstance(kernel, type) and issubclass(kernel, MCMCKernel):
        factory = cast("Callable[..., MCMCKernel]", kernel)
        return factory(model, **kwargs)
    if isinstance(kernel, MCMCKernel):
        if kwargs:
            msg = (
                "kernel_kwargs cannot be combined with an already-constructed "
                "kernel instance; pass the kernel class instead, or set the "
                "options on the instance."
            )
            raise ValueError(msg)
        return kernel
    msg = (
        f"resolve_kernel() does not support {type(kernel).__name__}; pass None, "
        "an MCMCKernel subclass, or an MCMCKernel instance."
    )
    raise TypeError(msg)


def _is_blackjax_kernel(kernel: MCMCKernel) -> bool:
    """Return whether ``kernel`` is a ``_BlackjaxKernel`` (by MRO name, import-free)."""
    return any(base.__name__ == "_BlackjaxKernel" for base in type(kernel).__mro__)


def _validate_kernel_run_config(
    kernel: MCMCKernel, num_chains: int, chain_method: str, num_warmup: int
) -> None:
    """Entry-point checks for constraints NumPyro surfaces late or never.

    - ``AIES``/``ESS``: require ``num_chains > 1`` and ``chain_method ==
      "vectorized"`` (the ensemble is the chain batch).
    - ``_BlackjaxKernel`` subclasses: require ``chain_method == "sequential"``
      (instance-held step/postprocess functions capture tracers under vmap/pmap
      tracing, risk K5); warn when ``num_warmup > 0`` (adaptation lives in
      ``kernel.init``, so warmup steps are discarded work).

    Parameters
    ----------
    kernel
        The resolved kernel.
    num_chains
        The requested number of chains.
    chain_method
        The requested chain method (``"sequential"``/``"parallel"``/``"vectorized"``).
    num_warmup
        The requested number of warmup steps.

    Raises
    ------
    ValueError
        For each violated constraint, naming the constraint and the fix.
    """
    if isinstance(kernel, (AIES, ESS)):
        name = type(kernel).__name__
        if num_chains <= 1 or chain_method != "vectorized":
            msg = (
                f"{name} is an ensemble sampler: it requires num_chains > 1 and "
                f'chain_method="vectorized" (got num_chains={num_chains}, '
                f'chain_method="{chain_method}").'
            )
            raise ValueError(msg)
    if _is_blackjax_kernel(kernel):
        if chain_method != "sequential":
            msg = (
                f'{type(kernel).__name__} requires chain_method="sequential"; '
                "its step/postprocess functions capture tracers under "
                f'vmap/pmap (got chain_method="{chain_method}").'
            )
            raise ValueError(msg)
        if num_warmup > 0:
            warnings.warn(
                f"{type(kernel).__name__} performs adaptation in kernel.init; "
                f"num_warmup={num_warmup} warmup steps are discarded work, pass "
                "num_warmup=0.",
                stacklevel=2,
            )


def fit_mcmc(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    kernel: "KernelLike" = None,
    kernel_kwargs: Mapping[str, Any] | None = None,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 1,
    chain_method: str = "sequential",
    progress_bar: bool = False,
) -> MCMCFit:
    """Fit a forecasting model with MCMC.

    PRNG: consumed entirely by :class:`~numpyro.infer.MCMC`.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    kernel
        Kernel specification resolved by :func:`resolve_kernel`: ``None``
        (``NUTS``), an ``MCMCKernel`` instance, or an ``MCMCKernel`` subclass.
    kernel_kwargs
        Extra keyword arguments for the kernel constructor (only with ``None``
        or a kernel class; rejected with an instance).
    num_warmup
        Number of warmup steps.
    num_samples
        Number of posterior samples.
    num_chains
        Number of MCMC chains (stored on the returned :class:`MCMCFit`).
    chain_method
        NumPyro chain method (``"sequential"``/``"parallel"``/``"vectorized"``).
    progress_bar
        Whether to display the MCMC progress bar.

    Returns
    -------
    MCMCFit
        The posterior samples (flattened) and ``num_chains``.

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` have different durations, or a run-config
        constraint is violated (see :func:`_validate_kernel_run_config`).
    """
    _require_equal_duration(data, covariates)
    resolved_kernel = resolve_kernel(kernel, model, kernel_kwargs)
    _validate_kernel_run_config(resolved_kernel, num_chains, chain_method, num_warmup)
    mcmc = MCMC(
        resolved_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, covariates, data)
    return MCMCFit(samples=mcmc.get_samples(), num_chains=num_chains)


@_draw_posterior_impl.register
def _(fit: MCMCFit, num_samples: int, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    leaves = list(fit.samples.values())
    available = leaves[0].shape[0]
    if num_samples <= available:
        # Thin the genuine posterior draws on an evenly spaced grid: this is
        # deterministic, order-preserving, and free of the duplicate draws that
        # sampling with replacement would otherwise inject.
        indices = jnp.linspace(0, available - 1, num_samples).round().astype(int)
    else:
        # More draws requested than the chain holds: fall back to resampling
        # with replacement (the only way to grow the sample count).
        indices = random.choice(rng_key, available, shape=(num_samples,), replace=True)
    return _index_tree(fit.samples, indices)


@partial(jax.jit, static_argnums=(1,), static_argnames=("parallel",))
def _predict(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    data: Array,
    covariates: Array,
    *,
    parallel: bool = True,
) -> Array:
    """Run ``Predictive`` over the full horizon and return the ``forecast`` site.

    Jitted with ``model`` (and ``parallel``) static: each
    ``(model, parallel, shape)`` combination compiles once and is reused, which
    is what makes the chunked :func:`forecast` loop cheap (the per-call
    ``Predictive`` tracing cost is paid a single time). ``parallel`` selects
    ``Predictive``'s sample-axis mapping (``vmap`` when ``True``, serial
    ``lax.map`` when ``False``).
    """
    predictive = Predictive(
        model, posterior_samples=posterior, return_sites=["forecast"], parallel=parallel
    )
    return predictive(rng_key, covariates, data)["forecast"]


def forecast(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    data: Array,
    covariates: Array,
    *,
    batch_size: int | None = None,
    parallel: bool = True,
) -> Num[Array, " sample *batch future obs"]:
    """Sample forecasts for the steps in ``[t, duration)`` from a posterior.

    Runs ``Predictive`` with full-horizon ``covariates`` and the in-sample
    ``data``: the in-sample latent sites are drawn from ``posterior`` while the
    ``_future`` suffix is drawn from the prior, and the ``"forecast"`` site is
    returned. The number of forecast samples equals the leading (sample) axis of
    ``posterior`` (see :func:`draw_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading.
    data
        Observed data with time at axis ``-2`` and length ``t``.
    covariates
        Covariates with time at axis ``-2`` and length ``duration > t``.
    batch_size
        Optional chunk size for sampling (caps peak memory).
    parallel
        Whether ``Predictive`` vectorizes over the sample axis with ``vmap``
        (``True``, faster, higher peak memory) or maps it serially with
        ``lax.map`` (``False``). With ``parallel=True`` the samples in each
        ``batch_size`` chunk are vectorized while the chunks are looped over, so
        ``batch_size`` remains the peak-memory governor. The two settings produce
        the same draws up to floating-point reduction order.

    Returns
    -------
    Num[Array, " sample *batch future obs"]
        Forecast samples over the ``future = duration - t`` horizon (floating
        point for continuous observations, integer for discrete/count models
        built with :func:`predict_glm`).

    Raises
    ------
    ValueError
        If ``covariates`` does not extend beyond ``data`` along the time axis.

    Notes
    -----
    Chunking is a memory knob, not a reproducibility knob: reproducibility is
    per ``(rng_key, batch_size)``. Each chunk is padded to a whole multiple of
    ``batch_size`` so the underlying ``_predict`` compiles exactly once for a
    fixed shape (the pad draws are discarded), but changing ``batch_size``
    changes the PRNG stream layout and therefore the exact draws.
    """
    _require_covariates_extend_data(data, covariates)
    num_samples = next(iter(posterior.values())).shape[0]
    if batch_size is None or batch_size >= num_samples:
        return _predict(rng_key, model, posterior, data, covariates, parallel=parallel)
    padded, num = _pad_posterior(posterior, batch_size)
    num_padded = next(iter(padded.values())).shape[0]
    keys = random.split(rng_key, num_padded // batch_size)
    chunks = [
        _predict(
            keys[i],
            model,
            _index_tree(padded, slice(start, start + batch_size)),
            data,
            covariates,
            parallel=parallel,
        )
        for i, start in enumerate(range(0, num_padded, batch_size))
    ]
    return jnp.concatenate(chunks, axis=0)[:num]


@partial(jax.jit, static_argnums=(1,), static_argnames=("parallel",))
def _predict_obs(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    covariates: Array,
    *,
    parallel: bool = True,
) -> Array:
    """Run ``Predictive`` over the observed window and return the ``obs`` site.

    Jitted with ``model`` (and ``parallel``) static (see :func:`_predict`): each
    ``(model, parallel, shape)`` combination compiles once and is reused, keeping
    the chunked :func:`predict_in_sample` loop cheap.
    """
    predictive = Predictive(
        model, posterior_samples=posterior, return_sites=["obs"], parallel=parallel
    )
    return predictive(rng_key, covariates)["obs"]


def predict_in_sample(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    covariates: Array,
    *,
    batch_size: int | None = None,
    parallel: bool = True,
) -> Num[Array, " sample *batch time obs"]:
    """Sample the in-sample posterior predictive of the ``obs`` site.

    Runs ``Predictive`` with the in-sample ``covariates`` and the supplied posterior
    latent draws. Unlike :func:`forecast` there is no forecast horizon: ``covariates``
    span only the observed window, so the model's ``obs`` site is sampled at every
    step. The number of predictive samples equals the leading (sample) axis of
    ``posterior`` (see :func:`draw_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading.
    covariates
        Covariates with time at axis ``-2`` spanning the observed window. Its time
        length must match the data the ``posterior`` was fit on, since the in-sample
        latent sites are sized to that window.
    batch_size
        Optional chunk size for sampling (caps peak memory).
    parallel
        Whether ``Predictive`` vectorizes over the sample axis with ``vmap``
        (``True``, faster, higher peak memory) or maps it serially with
        ``lax.map`` (``False``). See :func:`forecast` for how this interacts with
        ``batch_size``.

    Returns
    -------
    Num[Array, " sample *batch time obs"]
        In-sample posterior-predictive draws of the ``obs`` site.
    """
    num_samples = next(iter(posterior.values())).shape[0]
    if batch_size is None or batch_size >= num_samples:
        return _predict_obs(rng_key, model, posterior, covariates, parallel=parallel)
    padded, num = _pad_posterior(posterior, batch_size)
    num_padded = next(iter(padded.values())).shape[0]
    keys = random.split(rng_key, num_padded // batch_size)
    chunks = [
        _predict_obs(
            keys[i],
            model,
            _index_tree(padded, slice(start, start + batch_size)),
            covariates,
            parallel=parallel,
        )
        for i, start in enumerate(range(0, num_padded, batch_size))
    ]
    return jnp.concatenate(chunks, axis=0)[:num]
