"""Model-building primitives for the functional API.

The :class:`Horizon` value carries the train/forecast split, and the
site-registration functions (:func:`time_series`, :func:`markov_time_series`,
:func:`predict`, :func:`predict_glm`) sample latents and observation sites
against it. :func:`forecasting_model` wraps a pure model body into the
standard NumPyro model callable ``(covariates, data=None)``.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import Reparam

from numpyro_forecast.arrays import _zeros_like_data, concat_future
from numpyro_forecast.surgery import prefix_condition, shift_loc, slice_time
from numpyro_forecast.typing import Array, ForecastModel


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
    """Require a non-degenerate per-step shape with an observation axis."""
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
        observation dimension, or if an enclosing plate is detected.
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
    (:func:`~numpyro_forecast.surgery.slice_time` /
    :func:`~numpyro_forecast.surgery.prefix_condition`), i.e. an elementwise family.

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

    ``model_fn`` is a pure function ``(Horizon, covariates) -> None`` that calls
    :func:`time_series` and :func:`predict`; this wraps it into the standard
    NumPyro model callable ``(covariates, data=None)``, deriving the
    :class:`Horizon` from the shapes.

    Parameters
    ----------
    model_fn
        The model body. It receives the per-call :class:`Horizon` (use
        ``h.zero_data`` for the Pyro-style ``zero_data``) and the covariates with
        time at axis ``-2``.

    Returns
    -------
    ForecastModel
        A callable ``(covariates, data=None) -> None`` usable directly with
        NumPyro's ``SVI``, ``MCMC``, and ``Predictive``.
    """

    def numpyro_model(covariates: Array, data: Array | None = None) -> None:
        model_fn(Horizon.from_data(covariates, data), covariates)

    return numpyro_model
