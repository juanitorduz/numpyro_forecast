"""Forecasting model base class and SVI/MCMC forecasters.

This is the JAX/NumPyro port of Pyro's ``pyro.contrib.forecast.forecaster``.
The classes here are thin object-oriented shims over the functional core in
:mod:`numpyro_forecast.functional`: :class:`ForecastingModel` threads the
train/forecast :class:`~numpyro_forecast.functional.models.Horizon` for you, and the
forecaster classes wrap :func:`~numpyro_forecast.functional.svi.fit_svi` /
:func:`~numpyro_forecast.functional.mcmc.fit_mcmc` plus
:func:`~numpyro_forecast.functional.prediction.forecast`. The two styles are fully
interchangeable: both consume a model callable ``(covariates, data=None)`` and a
posterior dict of latent draws.
"""

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax
import numpyro.distributions as dist
from jax import random
from jaxtyping import Num
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.reparam import Reparam

from numpyro_forecast.functional import (
    Horizon,
    Transition,
    draw_posterior,
    fit_mcmc,
    fit_svi,
)
from numpyro_forecast.functional import forecast as _forecast
from numpyro_forecast.functional import markov_time_series as _markov_time_series
from numpyro_forecast.functional import predict as _predict
from numpyro_forecast.functional import predict_glm as _predict_glm
from numpyro_forecast.functional import predict_in_sample as _predict_in_sample
from numpyro_forecast.functional import time_series as _time_series
from numpyro_forecast.functional._validation import (
    _require_covariates_extend_data,
    _require_positive_num_samples,
)
from numpyro_forecast.typing import (
    Array,
    ForecastModel,
    GuideLike,
    KernelLike,
    OptimizerLike,
)


class ForecastingModel(abc.ABC):
    """Abstract base class for forecasting models.

    Subclasses implement :meth:`model`, which must call :meth:`predict` exactly
    once. The instance itself is the (pure) NumPyro model function with signature
    ``model_instance(covariates, data=None)``: the forecast horizon is inferred
    from the shapes (``future = covariates.shape[-2] - data.shape[-2]``).

    This is the object-oriented façade over the functional API: :meth:`time_series`
    and :meth:`predict` delegate to the free functions in
    :mod:`numpyro_forecast.functional`, passing the current
    :class:`~numpyro_forecast.functional.models.Horizon`.
    """

    def __init__(self) -> None:
        self._horizon: Horizon | None = None

    @abc.abstractmethod
    def model(self, zero_data: Array | None, covariates: Array) -> None:
        """Define the generative model and call :meth:`predict` exactly once.

        Parameters
        ----------
        zero_data
            Zeros shaped like the data extended to the covariate duration
            (shape/dtype only; ``None`` during pure prior sampling).
        covariates
            Covariates with time at axis ``-2`` and shape
            ``(*batch, duration, cov)``.
        """
        raise NotImplementedError

    def _require_horizon(self) -> Horizon:
        if self._horizon is None:
            msg = "model state is only available during a model call"
            raise RuntimeError(msg)
        return self._horizon

    @property
    def duration(self) -> int:
        """Total horizon length ``t + future`` (in time steps)."""
        return self._require_horizon().duration

    @property
    def t_obs(self) -> int:
        """Number of observed (in-sample) time steps ``t``."""
        return self._require_horizon().t_obs

    @property
    def future(self) -> int:
        """Number of forecast time steps ``f`` (``0`` while training)."""
        return self._require_horizon().future

    def time_series(
        self,
        name: str,
        dist_fn: Callable[[], dist.Distribution],
        *,
        reparam: Reparam | None = None,
    ) -> Array:
        """Sample a time-varying latent over the full horizon.

        Thin wrapper over :func:`numpyro_forecast.functional.models.time_series`.

        Parameters
        ----------
        name
            Base sample-site name for the in-sample latent.
        dist_fn
            Zero-argument callable returning the per-step prior distribution.
        reparam
            Optional reparameterization (e.g. ``LocScaleReparam``) applied to
            both the in-sample and forecast sites.

        Returns
        -------
        Array
            The latent over the full horizon with time at axis ``-2``.
        """
        return _time_series(self._require_horizon(), name, dist_fn, reparam=reparam)

    def predict(self, noise_dist: dist.Distribution, prediction: Array) -> None:
        """Register the observation/forecast sites for the model.

        Thin wrapper over :func:`numpyro_forecast.functional.models.predict`.

        Parameters
        ----------
        noise_dist
            Zero-centered observation noise (e.g. ``Normal(0, sigma)``).
        prediction
            Deterministic mean with time at axis ``-2``, shape
            ``(*batch, duration, obs)``.
        """
        _predict(self._require_horizon(), noise_dist, prediction)

    def predict_glm(
        self,
        obs_dist_fn: Callable[[Array], dist.Distribution],
        latent: Array,
    ) -> None:
        """Register GLM-style observation/forecast sites from a latent predictor.

        Thin wrapper over :func:`numpyro_forecast.functional.models.predict_glm`.

        Parameters
        ----------
        obs_dist_fn
            Link mapping the full-horizon ``latent`` predictor to the observation
            distribution (e.g. ``lambda eta: Poisson(jnp.exp(eta))``).
        latent
            The deterministic latent predictor over the full horizon, time at
            axis ``-2``.
        """
        _predict_glm(self._require_horizon(), obs_dist_fn, latent)

    def markov_time_series(
        self,
        name: str,
        init_carry: Any,
        transition: Transition,
        xs: Array | None = None,
        *,
        plates: Sequence[tuple[str, int]] = (),
        reparam_config: Mapping[str, Reparam] | None = None,
    ) -> Array:
        """Sample a Markov (state-space) latent over the full horizon.

        Thin wrapper over :func:`numpyro_forecast.functional.models.markov_time_series`
        that threads this model's train/forecast horizon. In-sample steps run in a
        ``scan`` under site ``name``; the forecast horizon runs in a second scan
        under ``f"{name}_future"`` seeded by the final in-sample carry, so the guide
        never sees the future site.

        Parameters
        ----------
        name
            Sample-site name for the in-sample scan; the forecast scan uses
            ``f"{name}_future"``.
        init_carry
            Initial carry PyTree fed to the first transition step.
        transition
            Callable ``(carry, x_t) -> (dist_t, carry_fn)``: ``dist_t`` is the
            per-step observation distribution (its per-step shape must carry the
            trailing observation dimension) and ``carry_fn(z_t)`` builds the next
            carry from the sampled latent ``z_t``. The wrapper owns the ``sample``
            statement, so the Markov structure cannot be broken by resampling.
        xs
            Optional exogenous inputs spanning the full horizon with time at axis
            ``-2``; split and moved into scan layout internally. ``None`` for
            autonomous dynamics.
        plates
            ``(name, size)`` pairs opened inside the scan body around the sample
            statement (the only placement NumPyro accepts around a scan).
        reparam_config
            Optional site-name to :class:`~numpyro.infer.reparam.Reparam` mapping
            applied inside the scan body.

        Returns
        -------
        Array
            The latent over the full horizon in package layout
            ``(*plate_batch, duration, obs)`` (time at axis ``-2``).
        """
        return _markov_time_series(
            self._require_horizon(),
            name,
            init_carry,
            transition,
            xs,
            plates=plates,
            reparam_config=reparam_config,
        )

    def __call__(self, covariates: Array, data: Array | None = None) -> None:
        """Run the model as a NumPyro model function.

        Parameters
        ----------
        covariates
            Covariates with time at axis ``-2`` spanning the full horizon.
        data
            Observed data with time at axis ``-2`` (``None`` for prior sampling).
        """
        horizon = Horizon.from_data(covariates, data)
        self._horizon = horizon
        try:
            self.model(horizon.zero_data, covariates)
        finally:
            self._horizon = None


class _BaseForecaster(abc.ABC):
    """Shared forecasting logic over a fitted posterior."""

    def __init__(self, model: ForecastModel, t_obs: int) -> None:
        self.model = model
        self._t_obs = t_obs

    @abc.abstractmethod
    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        """Return ``num_samples`` posterior draws of the latent sites."""
        raise NotImplementedError

    def __call__(
        self,
        rng_key: Array,
        data: Array,
        covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
        parallel: bool = True,
        device: jax.Device | str | None = None,
    ) -> Num[Array, " sample *batch future obs"]:
        """Sample forecasts for the steps in ``[t, duration)``.

        Parameters
        ----------
        rng_key
            PRNG key.
        data
            Observed data with time at axis ``-2`` and length ``t``.
        covariates
            Covariates with time at axis ``-2`` and length ``duration > t``.
        num_samples
            Number of forecast samples to draw.
        batch_size
            Optional chunk size for sampling (caps peak memory).
        parallel
            Whether to vectorize the sample axis with ``vmap`` (``True``, faster,
            higher peak memory) or map it serially (``False``). See
            :func:`numpyro_forecast.functional.prediction.forecast`.
        device
            Optional device (or platform name like ``"cpu"``) where each chunk
            of draws is placed as soon as it is drawn and where the stitched
            result lives. See :func:`numpyro_forecast.functional.prediction.forecast`.

        Returns
        -------
        Num[Array, " sample *batch future obs"]
            Forecast samples over the ``future = duration - t`` horizon (integer
            for discrete/count models built with ``predict_glm``).
        """
        _require_covariates_extend_data(data, covariates)
        _require_positive_num_samples(num_samples)
        key_post, key_pred = random.split(rng_key)
        posterior = self._draw_posterior(key_post, num_samples)
        return _forecast(
            key_pred,
            self.model,
            posterior,
            data,
            covariates,
            batch_size=batch_size,
            parallel=parallel,
            device=device,
        )

    def predict_in_sample(
        self,
        rng_key: Array,
        covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
        parallel: bool = True,
        device: jax.Device | str | None = None,
    ) -> Num[Array, " sample *batch time obs"]:
        """Sample the in-sample posterior predictive of the ``obs`` site.

        Draws ``num_samples`` posterior latent samples from the fitted forecaster and
        runs the model forward over ``covariates`` with no forecast horizon, returning
        the ``obs`` site. This is the in-sample counterpart of :meth:`__call__`, which
        instead forecasts the future horizon.

        Parameters
        ----------
        rng_key
            PRNG key.
        covariates
            Covariates with time at axis ``-2`` spanning the observed window. Its time
            length must match the data the forecaster was fit on.
        num_samples
            Number of posterior-predictive samples to draw.
        batch_size
            Optional chunk size for sampling (caps peak memory).
        parallel
            Whether to vectorize the sample axis with ``vmap`` (``True``, faster,
            higher peak memory) or map it serially (``False``). See
            :func:`numpyro_forecast.functional.prediction.predict_in_sample`.
        device
            Optional device (or platform name like ``"cpu"``) where each chunk
            of draws is placed as soon as it is drawn and where the stitched
            result lives. See
            :func:`numpyro_forecast.functional.prediction.predict_in_sample`.

        Returns
        -------
        Num[Array, " sample *batch time obs"]
            In-sample posterior-predictive draws of the ``obs`` site.

        Raises
        ------
        ValueError
            If ``num_samples`` is not positive, or if ``covariates`` does not
            match the in-sample duration the forecaster was fit on.
        """
        if covariates.shape[-2] != self._t_obs:
            msg = (
                "predict_in_sample expects covariates of the same duration as the "
                f"fitted data ({self._t_obs}), got {covariates.shape[-2]}"
            )
            raise ValueError(msg)
        _require_positive_num_samples(num_samples)
        key_post, key_pred = random.split(rng_key)
        posterior = self._draw_posterior(key_post, num_samples)
        return _predict_in_sample(
            key_pred,
            self.model,
            posterior,
            covariates,
            batch_size=batch_size,
            parallel=parallel,
            device=device,
        )


class Forecaster(_BaseForecaster):
    """Fit a forecasting model with stochastic variational inference.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model to fit (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    guide
        Guide specification resolved by
        :func:`~numpyro_forecast.functional.svi.resolve_guide`: ``None``
        (``AutoNormal``), an ``AutoGuide`` instance, an ``AutoGuide`` subclass or
        ``functools.partial`` factory of one, or a hand-written guide function.
    optim
        Optimizer specification resolved by
        :func:`~numpyro_forecast.functional.svi.resolve_optimizer`: ``None``
        (``Adam(0.01)``), a positive scalar learning rate, an
        ``optax.GradientTransformation``, or a ``_NumPyroOptim``.
    num_steps
        Number of SVI steps.
    num_particles
        Number of ELBO particles.
    progress_bar
        Whether to display the SVI progress bar.
    stable_update
        Whether SVI skips non-finite parameter updates.
    """

    def __init__(
        self,
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
    ) -> None:
        super().__init__(model, data.shape[-2])
        self._fit = fit_svi(
            rng_key,
            model,
            data,
            covariates,
            guide=guide,
            optim=optim,
            num_steps=num_steps,
            num_particles=num_particles,
            progress_bar=progress_bar,
            stable_update=stable_update,
        )
        # Typed as AutoGuide for the common OOP path; hand-written guides
        # (callables) are supported at runtime but rare through this class.
        self.guide: AutoGuide = cast("AutoGuide", self._fit.guide)
        self.params: dict[str, Array] = self._fit.params
        self.losses: Array = self._fit.losses

    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        return draw_posterior(rng_key, self._fit, num_samples)


class HMCForecaster(_BaseForecaster):
    """Fit a forecasting model with MCMC (NUTS by default).

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model to fit (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    kernel
        Kernel specification resolved by
        :func:`~numpyro_forecast.functional.mcmc.resolve_kernel`: ``None`` (``NUTS``),
        an ``MCMCKernel`` instance, or an ``MCMCKernel`` subclass.
    kernel_kwargs
        Extra keyword arguments for the kernel constructor (only with ``None``
        or a kernel class).
    num_warmup
        Number of warmup steps.
    num_samples
        Number of posterior samples.
    num_chains
        Number of MCMC chains.
    chain_method
        NumPyro chain method (``"sequential"``/``"parallel"``/``"vectorized"``).
    progress_bar
        Whether to display the MCMC progress bar.
    """

    def __init__(
        self,
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
    ) -> None:
        super().__init__(model, data.shape[-2])
        self._fit = fit_mcmc(
            rng_key,
            model,
            data,
            covariates,
            kernel=kernel,
            kernel_kwargs=kernel_kwargs,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )
        self.posterior_samples: dict[str, Array] = self._fit.samples

    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        return draw_posterior(rng_key, self._fit, num_samples)


class PathfinderForecaster(_BaseForecaster):
    """Fit a forecasting model with BlackJAX Pathfinder variational inference.

    A thin shim over :func:`numpyro_forecast.contrib.blackjax.fit_pathfinder`.
    BlackJAX is an optional dependency (``pip install numpyro_forecast[blackjax]``)
    imported lazily here, so constructing this class is the opt-in that pulls it
    in; importing :mod:`numpyro_forecast` never does. The
    constructor mirrors :class:`Forecaster` without ``guide``/``optim`` (Pathfinder
    has neither) and adds ``num_elbo_samples``/``ftol``.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model to fit (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    num_elbo_samples
        Number of Monte Carlo samples used to estimate the ELBO along the
        L-BFGS path.
    ftol
        L-BFGS relative function-value tolerance (convergence criterion).
    maxiter
        Maximum number of L-BFGS iterations (see
        :func:`~numpyro_forecast.contrib.blackjax.fit_pathfinder`; models with
        per-step time latents typically need far more than the default ``30``).
    """

    def __init__(
        self,
        rng_key: Array,
        model: ForecastModel,
        data: Array,
        covariates: Array,
        *,
        num_elbo_samples: int = 200,
        ftol: float = 1e-5,
        maxiter: int = 30,
    ) -> None:
        super().__init__(model, data.shape[-2])
        # Lazy import: registers the PathfinderFit draw dispatch and defers the
        # blackjax dependency to first use (I8).
        from numpyro_forecast.contrib.blackjax import fit_pathfinder

        self._fit = fit_pathfinder(
            rng_key,
            model,
            data,
            covariates,
            num_elbo_samples=num_elbo_samples,
            ftol=ftol,
            maxiter=maxiter,
        )
        self.elbo: float = self._fit.elbo

    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        return draw_posterior(rng_key, self._fit, num_samples)
