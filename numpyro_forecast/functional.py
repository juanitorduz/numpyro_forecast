"""Functional forecasting API: the pure core shared with the OOP classes.

This module is the functional counterpart of :mod:`numpyro_forecast.forecaster`.
Where the OOP API carries the train/forecast split as mutable state on a
:class:`~numpyro_forecast.forecaster.ForecastingModel` instance, here it is an
explicit, immutable :class:`Horizon` value threaded into the primitives. The
class-based API is a thin shim over the functions defined here, so the two
styles are fully interchangeable: both produce a NumPyro model callable
``(covariates, data=None)`` and consume a posterior dict of latent draws.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import singledispatch
from typing import cast

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.tree_util import tree_map
from jaxtyping import Float
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.infer.reparam import Reparam
from numpyro.optim import Adam, _NumPyroOptim

from numpyro_forecast.typing import Array, ForecastModel
from numpyro_forecast.util import (
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
        shape = (*self.data.shape[:-2], self.duration, self.data.shape[-1])
        return jnp.zeros(shape, dtype=self.data.dtype)

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


def predict(h: Horizon, noise_dist: dist.Distribution, prediction: Array) -> None:
    """Register the observation/forecast sites for the model.

    ``noise_dist`` is a zero-centered observation noise distribution and
    ``prediction`` the deterministic mean over the full horizon. While training
    the residual is observed; while forecasting the in-sample prefix is observed
    and the forecast suffix is sampled and exposed as the ``"forecast"``
    deterministic site.

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
    obs_dist = shift_loc(noise_dist, prediction)
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
    """

    guide: AutoGuide
    params: dict[str, Array]
    losses: Array


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


def _index_tree(tree: dict[str, Array], index: Array | slice) -> dict[str, Array]:
    """Index every leaf of a posterior-sample pytree along its sample axis."""
    return tree_map(lambda leaf: leaf[index], tree)


def fit_svi(
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    guide: AutoGuide | None = None,
    optim: _NumPyroOptim | None = None,
    num_steps: int = 1001,
    num_particles: int = 1,
    rng_key: Array,
    progress_bar: bool = False,
) -> SVIFit:
    """Fit a forecasting model with stochastic variational inference.

    Parameters
    ----------
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    guide
        Variational guide; defaults to ``AutoNormal(model)``.
    optim
        NumPyro optimizer; defaults to ``Adam(0.01)``.
    num_steps
        Number of SVI steps.
    num_particles
        Number of ELBO particles.
    rng_key
        PRNG key for inference.
    progress_bar
        Whether to display the SVI progress bar.

    Returns
    -------
    SVIFit
        The fitted guide, variational parameters, and loss history.

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` have different durations.
    """
    _require_equal_duration(data, covariates)
    guide = AutoNormal(model) if guide is None else guide
    optimizer = Adam(0.01) if optim is None else optim
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=num_particles))
    result = svi.run(rng_key, num_steps, covariates, data, progress_bar=progress_bar)
    return SVIFit(guide=guide, params=result.params, losses=result.losses)


@singledispatch
def draw_posterior(fit: object, num_samples: int, *, rng_key: Array) -> dict[str, Array]:
    """Draw ``num_samples`` posterior samples of the latent sites from a fit.

    Dispatches on the fit type (e.g. :class:`SVIFit`, :class:`MCMCFit`). The
    returned dict has the sample axis leading and is ready to pass to
    :func:`forecast` or NumPyro's ``Predictive``.

    Parameters
    ----------
    fit
        A fit result produced by :func:`fit_svi` or :func:`fit_mcmc`.
    num_samples
        Number of posterior draws.
    rng_key
        PRNG key.

    Returns
    -------
    dict[str, Array]
        Posterior samples of the latent sites, sample axis leading.

    Raises
    ------
    NotImplementedError
        If ``fit`` is of an unsupported type.
    """
    msg = f"draw_posterior() does not support {type(fit).__name__}"
    raise NotImplementedError(msg)


@draw_posterior.register
def _(fit: SVIFit, num_samples: int, *, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    return fit.guide.sample_posterior(rng_key, fit.params, sample_shape=(num_samples,))


@dataclass(frozen=True)
class MCMCFit:
    """The result of fitting a forecasting model with MCMC (NUTS).

    Attributes
    ----------
    samples
        The posterior samples of the latent sites, sample axis leading.
    """

    samples: dict[str, Array]


def fit_mcmc(
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    rng_key: Array,
    progress_bar: bool = False,
) -> MCMCFit:
    """Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).

    Parameters
    ----------
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    num_warmup
        Number of warmup steps.
    num_samples
        Number of posterior samples.
    num_chains
        Number of MCMC chains.
    rng_key
        PRNG key for inference.
    progress_bar
        Whether to display the MCMC progress bar.

    Returns
    -------
    MCMCFit
        The posterior samples.

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` have different durations.
    """
    _require_equal_duration(data, covariates)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, covariates, data)
    return MCMCFit(samples=mcmc.get_samples())


@draw_posterior.register
def _(fit: MCMCFit, num_samples: int, *, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    leaves = list(fit.samples.values())
    available = leaves[0].shape[0]
    indices = random.choice(rng_key, available, shape=(num_samples,), replace=True)
    return _index_tree(fit.samples, indices)


def _predict(
    model: ForecastModel,
    posterior: dict[str, Array],
    data: Array,
    covariates: Array,
    rng_key: Array,
) -> Array:
    """Run ``Predictive`` over the full horizon and return the ``forecast`` site."""
    predictive = Predictive(model, posterior_samples=posterior, return_sites=["forecast"])
    return predictive(rng_key, covariates, data)["forecast"]


def forecast(
    model: ForecastModel,
    posterior: dict[str, Array],
    data: Array,
    covariates: Array,
    *,
    rng_key: Array,
    batch_size: int | None = None,
) -> Float[Array, " sample *batch future obs"]:
    """Sample forecasts for the steps in ``[t, duration)`` from a posterior.

    Runs ``Predictive`` with full-horizon ``covariates`` and the in-sample
    ``data``: the in-sample latent sites are drawn from ``posterior`` while the
    ``_future`` suffix is drawn from the prior, and the ``"forecast"`` site is
    returned. The number of forecast samples equals the leading (sample) axis of
    ``posterior`` (see :func:`draw_posterior`).

    Parameters
    ----------
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading.
    data
        Observed data with time at axis ``-2`` and length ``t``.
    covariates
        Covariates with time at axis ``-2`` and length ``duration > t``.
    rng_key
        PRNG key.
    batch_size
        Optional chunk size for sampling (caps peak memory).

    Returns
    -------
    Float[Array, " sample *batch future obs"]
        Forecast samples over the ``future = duration - t`` horizon.

    Raises
    ------
    ValueError
        If ``covariates`` does not extend beyond ``data`` along the time axis.
    """
    if data.shape[-2] >= covariates.shape[-2]:
        msg = "covariates must extend beyond data along the time axis"
        raise ValueError(msg)
    num_samples = next(iter(posterior.values())).shape[0]
    if batch_size is None or batch_size >= num_samples:
        return _predict(model, posterior, data, covariates, rng_key)
    outputs = []
    key = rng_key
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        key, sub_key = random.split(key)
        chunk = _index_tree(posterior, slice(start, stop))
        outputs.append(_predict(model, chunk, data, covariates, sub_key))
    return jnp.concatenate(outputs, axis=0)
