"""Model building blocks: plain functions that register the train/forecast sites.

The :class:`Horizon` value carries the train/forecast split, and the building
blocks (:func:`innovations`, :func:`markov_series`, :func:`predict`) sample
latents and observation sites against it. A model is a plain NumPyro function
``(covariates, data=None) -> None`` whose first line derives its
:class:`Horizon` from the shapes with :func:`horizon` and which then calls the
blocks directly. They are ordinary Python functions that call
``numpyro.sample`` and ``numpyro.deterministic`` on your behalf: not NumPyro
primitives, and not effect handlers.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import PyTree
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import Reparam

from numpyro_forecast.arrays import _zeros_like_data, concat_future
from numpyro_forecast.surgery import prefix_condition, shift_loc, slice_time
from numpyro_forecast.typing import Array


@dataclass(frozen=True)
class Horizon:
    """The train/forecast split for a single model call.

    An immutable value derived once per model call from the covariate and data
    shapes by :func:`horizon`; every building block (:func:`innovations`,
    :func:`markov_series`, :func:`predict`) takes it as its first argument.

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

        Mirrors Pyro's ``zero_data`` (and :func:`numpyro_forecast.arrays.zero_data_like`):
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


def horizon(covariates: Array, data: Array | None) -> Horizon:
    """Derive the :class:`Horizon` from the covariate and data shapes.

    The first line of every model: ``h = horizon(covariates, data)``.

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
    return Horizon(data=data, t_obs=t_obs, future=duration - t_obs, duration=duration)


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
    depends on the previous state is :func:`markov_series`.

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


type Transition[Carry] = Callable[
    [Carry, PyTree[Array] | None],
    tuple[dist.Distribution, Callable[[Array], Carry]],
]
"""``(carry, x_t) -> (dist_t, carry_fn)`` where ``carry_fn(z_t)`` builds the next
carry from the *sampled* latent. The wrapper owns the sample statement.

``Carry`` is the user's carry type (any PyTree), bound per :func:`markov_series`
call; ``x_t`` is one row of the ``xs`` PyTree (``None`` for autonomous dynamics)."""


@contextmanager
def _plate_stack(plates: Sequence[tuple[str, int]]):
    """Open nested ``numpyro.plate`` contexts (innermost last)."""
    with ExitStack() as stack:
        for plate_name, size in plates:
            stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
        yield


def _reject_enclosing_plates() -> None:
    """Raise if ``markov_series`` is called inside an enclosing plate."""
    try:
        from numpyro.primitives import _PYRO_STACK

        for msg in _PYRO_STACK:
            if type(msg).__name__ == "plate":
                msg_text = (
                    "markov_series opens plates internally via the plates= "
                    "argument; do not wrap the call in an enclosing numpyro.plate."
                )
                raise ValueError(msg_text)
    except (ImportError, AttributeError, TypeError):
        pass


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
    never sees the future site (same invariant as :func:`innovations`), and
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
        Optional exogenous inputs over the full horizon: a PyTree of arrays
        with time at axis ``-2`` (a single array, a tuple, a dict, ...), moved
        leaf by leaf into scan layout internally; ``None`` for autonomous
        dynamics.
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
        observation dimension, or if an enclosing plate is detected.
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


def predict(
    h: Horizon,
    obs_dist: dist.Distribution | Callable[[Array], dist.Distribution],
    prediction: Array,
) -> None:
    """Register the observation and forecast sites for the model.

    ``prediction`` is the deterministic predictor over the full horizon (time at
    axis ``-2``, shape ``(*batch, duration, obs)``). ``obs_dist`` takes one of
    two forms. A :class:`~numpyro.distributions.Distribution` is zero-centered
    observation noise (e.g. ``dist.StudentT(nu, 0.0, sigma)``) shifted onto the
    predictor with :func:`~numpyro_forecast.surgery.shift_loc`, which also owns
    the multivariate-normal layout check. A callable is a link mapping the
    predictor to the observation distribution directly (the GLM form, e.g.
    ``lambda eta: dist.Poisson(jnp.exp(eta))``). Either way, while training the
    observation site ``"obs"`` is observed; while forecasting the in-sample
    prefix is observed and the forecast suffix is sampled at ``"obs_future"``
    and exposed as the ``"forecast"`` deterministic site that
    :func:`~numpyro_forecast.predictive.forecast` reads. The observation
    distribution must support time-axis surgery
    (:func:`~numpyro_forecast.surgery.slice_time` /
    :func:`~numpyro_forecast.surgery.prefix_condition`), i.e. an elementwise
    family or a registered one.

    Parameters
    ----------
    h
        The horizon for the current model call (see :class:`Horizon`).
    obs_dist
        Zero-centered noise distribution (shifted onto ``prediction``) or a
        link callable from the full-horizon predictor to the observation
        distribution.
    prediction
        The deterministic predictor over the full horizon, time at axis ``-2``.

    Raises
    ------
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
    support = full_dist.support
    if support is not None and support.is_discrete and h.data is not None:
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
