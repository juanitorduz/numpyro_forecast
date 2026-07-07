"""BlackJAX MCMC kernels exposed as NumPyro :class:`~numpyro.infer.mcmc.MCMCKernel`.

This module adapts `BlackJAX <https://blackjax-devs.github.io/blackjax/>`_ sampling
algorithms to the NumPyro ``MCMCKernel`` interface so they can be passed straight to
:func:`numpyro_forecast.functional.mcmc.fit_mcmc` (and thus
:class:`~numpyro_forecast.forecaster.HMCForecaster`).

BlackJAX is an optional dependency (``pip install numpyro_forecast[blackjax]``) and is
never imported at package import time: it is pulled in lazily, via
:func:`numpyro_forecast.optional.require`, the first time a kernel is initialized. All three
kernels run one blackjax step per NumPyro sampling step; adaptation happens once inside
:meth:`_BlackjaxKernel.init`, so these kernels require ``chain_method="sequential"`` and
``num_warmup=0`` (enforced/warned by
:func:`numpyro_forecast.functional.mcmc._validate_kernel_run_config`).
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, cast

import jax
from jax import random
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity

from numpyro_forecast.functional._validation import _require_positive_num_samples
from numpyro_forecast.functional.posterior import _draw_posterior_impl
from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array, BlackjaxBuildFn, ForecastModel

_BJState = namedtuple("_BJState", ["position", "inner", "rng_key"])
"""MCMC state threaded by every :class:`_BlackjaxKernel`.

``position`` is the unconstrained sample dict (the NumPyro ``sample_field``),
``inner`` is the opaque blackjax algorithm state, and ``rng_key`` is the PRNG key
carried between steps.
"""


class _BlackjaxKernel(MCMCKernel):
    """Base adapter turning a blackjax sampler into a NumPyro ``MCMCKernel``.

    Subclasses implement :meth:`_build`, which runs adaptation and returns the
    initial blackjax state together with a one-step transition function. This base
    owns the shared plumbing: it builds the model's log density via
    :func:`~numpyro.infer.util.initialize_model`, drives the blackjax step in
    :meth:`sample`, exposes ``sample_field = "position"``, and validates that the
    built state exposes a ``position`` with the same key set as the initial one (a
    tripwire for a malformed custom ``build_fn``).

    Parameters
    ----------
    model
        The NumPyro forecasting model the kernel samples from. Bound by
        :func:`~numpyro_forecast.functional.mcmc.resolve_kernel` when a kernel class
        is passed to :func:`~numpyro_forecast.functional.mcmc.fit_mcmc`.
    """

    sample_field = "position"

    def __init__(self, model: ForecastModel | None = None) -> None:
        self._model = model
        self._postprocess_fn: Any = None
        self._step_fn: Any = None

    @property
    def model(self) -> ForecastModel | None:
        """The model the kernel samples from (``None`` until bound)."""
        return self._model

    def _build(
        self,
        rng_key: Array,
        logdensity_fn: "Any",
        position: dict[str, Array],
        num_warmup: int,
    ) -> tuple[Any, Any]:
        """Run adaptation and return ``(initial_inner_state, step_fn)``.

        Parameters
        ----------
        rng_key
            PRNG key for adaptation/initialization.
        logdensity_fn
            The unnormalized log density over the unconstrained ``position``.
        position
            Initial unconstrained parameter dictionary.
        num_warmup
            The MCMC warmup count (blackjax kernels adapt here, so this is
            informational only).

        Returns
        -------
        tuple
            ``(inner_state, step_fn)`` where ``step_fn`` is
            ``(rng_key, inner_state) -> (inner_state, info)``.
        """
        raise NotImplementedError

    def init(
        self,
        rng_key: Array,
        num_warmup: int,
        init_params: dict[str, Array] | None,
        model_args: tuple[Any, ...],
        model_kwargs: dict[str, Any] | None,
    ) -> _BJState:
        """Initialize the model, run adaptation, and return the first state.

        PRNG: ``rng_key`` is split into model-initialization, adaptation, and the
        carried sampling stream.

        Parameters
        ----------
        rng_key
            PRNG key for initialization and adaptation.
        num_warmup
            Number of warmup steps (forwarded to :meth:`_build`; blackjax kernels
            adapt in ``init`` instead).
        init_params
            Optional initial unconstrained parameters; defaults to NumPyro's
            initialization strategy.
        model_args
            Positional arguments to the model (``(covariates, data)``).
        model_kwargs
            Keyword arguments to the model.

        Returns
        -------
        _BJState
            The initial sampler state.

        Raises
        ------
        ValueError
            If the kernel has no bound model.
        TypeError
            If :meth:`_build` returns a state whose ``position`` key set differs
            from the initial one (a malformed ``build_fn``).
        """
        require("blackjax", extra="blackjax")
        if self._model is None:
            msg = f"{type(self).__name__} has no bound model; pass the kernel class to fit_mcmc."
            raise ValueError(msg)
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, init_key, build_key = random.split(rng_key, 3)
        param_info, potential_fn_gen, postprocess_fn, _ = initialize_model(
            init_key,
            self._model,
            dynamic_args=True,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )
        self._postprocess_fn = postprocess_fn
        potential_fn = potential_fn_gen(*model_args, **model_kwargs)

        def logdensity_fn(position: dict[str, Array]) -> Array:
            return -potential_fn(position)

        position = param_info.z if init_params is None else init_params
        inner_state, step_fn = self._build(build_key, logdensity_fn, position, num_warmup)
        self._step_fn = step_fn

        built_position = inner_state.position
        if set(built_position) != set(position):
            msg = (
                f"{type(self).__name__} build produced a state whose position keys "
                f"{sorted(built_position)} differ from the model's "
                f"{sorted(position)}; a build_fn must return a state over the same sites."
            )
            raise TypeError(msg)
        return _BJState(position=built_position, inner=inner_state, rng_key=rng_key)

    def sample(
        self,
        state: _BJState,
        model_args: tuple[Any, ...],
        model_kwargs: dict[str, Any] | None,
    ) -> _BJState:
        """Advance the chain by one blackjax step.

        Parameters
        ----------
        state
            The current sampler state.
        model_args
            Positional arguments to the model (unused; the log density is closed
            over in :meth:`init`).
        model_kwargs
            Keyword arguments to the model (unused).

        Returns
        -------
        _BJState
            The next sampler state.
        """
        rng_key, step_key = random.split(state.rng_key)
        new_inner, _info = self._step_fn(step_key, state.inner)
        return _BJState(position=new_inner.position, inner=new_inner, rng_key=rng_key)

    def postprocess_fn(
        self, model_args: tuple[Any, ...], model_kwargs: dict[str, Any] | None
    ) -> "Any":
        """Return the transform from unconstrained samples to constrained sites.

        Parameters
        ----------
        model_args
            Positional arguments to the model.
        model_kwargs
            Keyword arguments to the model.

        Returns
        -------
        Callable
            NumPyro's postprocessing function, so ``get_samples()`` yields the same
            constrained/deterministic site set as any built-in kernel.
        """
        if self._postprocess_fn is None:
            return identity
        kwargs = {} if model_kwargs is None else model_kwargs
        return self._postprocess_fn(*model_args, **kwargs)


class BlackjaxNUTSKernel(_BlackjaxKernel):
    """BlackJAX NUTS with Stan-style window adaptation.

    Adaptation (step size and inverse mass matrix) runs once in
    :meth:`~_BlackjaxKernel.init` via ``blackjax.window_adaptation``; each MCMC step
    is then a single tuned NUTS step.

    Parameters
    ----------
    model
        The NumPyro model to sample from.
    num_adaptation_steps
        Number of window-adaptation steps.
    target_acceptance_rate
        Target acceptance probability for dual-averaging step-size adaptation.
    """

    def __init__(
        self,
        model: ForecastModel | None = None,
        *,
        num_adaptation_steps: int = 500,
        target_acceptance_rate: float = 0.8,
    ) -> None:
        super().__init__(model)
        self.num_adaptation_steps = num_adaptation_steps
        self.target_acceptance_rate = target_acceptance_rate

    def _build(
        self,
        rng_key: Array,
        logdensity_fn: "Any",
        position: dict[str, Array],
        num_warmup: int,
    ) -> tuple[Any, Any]:
        blackjax = require("blackjax", extra="blackjax")
        adapt = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            target_acceptance_rate=self.target_acceptance_rate,
        )
        result, _info = adapt.run(rng_key, position, num_steps=self.num_adaptation_steps)
        kernel = blackjax.nuts(logdensity_fn, **result.parameters)
        return result.state, kernel.step


class BlackjaxMCLMCKernel(_BlackjaxKernel):
    """BlackJAX Microcanonical Langevin Monte Carlo (MCLMC).

    The step size and trajectory length ``L`` are tuned once in
    :meth:`~_BlackjaxKernel.init` via ``blackjax.mclmc_find_L_and_step_size``; each
    MCMC step is then a single tuned MCLMC step.

    Parameters
    ----------
    model
        The NumPyro model to sample from.
    num_tuning_steps
        Number of tuning steps for ``L`` and the step size.
    """

    def __init__(
        self,
        model: ForecastModel | None = None,
        *,
        num_tuning_steps: int = 500,
    ) -> None:
        super().__init__(model)
        self.num_tuning_steps = num_tuning_steps

    def _build(
        self,
        rng_key: Array,
        logdensity_fn: "Any",
        position: dict[str, Array],
        num_warmup: int,
    ) -> tuple[Any, Any]:
        blackjax = require("blackjax", extra="blackjax")
        init_key, tune_key = random.split(rng_key)
        init_state = blackjax.mcmc.mclmc.init(
            position=position, logdensity_fn=logdensity_fn, rng_key=init_key
        )

        def kernel(inverse_mass_matrix: Array) -> Any:
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        tuned_state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=self.num_tuning_steps,
            state=init_state,
            rng_key=tune_key,
        )
        algorithm = blackjax.mclmc(logdensity_fn, L=params.L, step_size=params.step_size)
        return tuned_state, algorithm.step


class BlackjaxCustomKernel(_BlackjaxKernel):
    """Adapt an arbitrary BlackJAX sampler via a user-supplied ``build_fn``.

    The escape hatch for kernels without a dedicated wrapper. ``build_fn`` receives
    ``(rng_key, logdensity_fn, position, num_warmup)`` and must return
    ``(inner_state, step_fn)``, where ``inner_state`` exposes ``.position`` over the
    same sites as the model and ``step_fn`` has signature
    ``(rng_key, inner_state) -> (inner_state, info)``. The base class validates the
    returned state's key set and raises :class:`TypeError` if it is malformed.

    Parameters
    ----------
    model
        The NumPyro model to sample from.
    build_fn
        The blackjax build callable described above.
    """

    def __init__(
        self,
        model: ForecastModel | None = None,
        *,
        build_fn: BlackjaxBuildFn,
    ) -> None:
        super().__init__(model)
        self._build_fn = build_fn

    def _build(
        self,
        rng_key: Array,
        logdensity_fn: "Any",
        position: dict[str, Array],
        num_warmup: int,
    ) -> tuple[Any, Any]:
        result = self._build_fn(rng_key, logdensity_fn, position, num_warmup)
        return cast("tuple[Any, Any]", result)


@dataclass(frozen=True)
class PathfinderFit:
    """The result of fitting a forecasting model with BlackJAX Pathfinder.

    A plain-data (picklable) container: it holds the raw blackjax
    ``PathfinderState``, the model and its data/covariates (needed to rebuild the
    unconstrained-to-constrained transform when drawing), and the fitted ELBO.
    Draws are produced lazily by :func:`~numpyro_forecast.functional.posterior.draw_posterior`.

    Attributes
    ----------
    state
        The blackjax ``PathfinderState`` (the variational approximation).
    model
        The forecasting model that was fit.
    covariates
        In-sample covariates used at fit time (time at axis ``-2``).
    data
        In-sample data used at fit time (time at axis ``-2``).
    elbo
        The evidence lower bound of the fitted approximation.
    """

    state: Any
    model: ForecastModel
    covariates: Array
    data: Array
    elbo: float


def fit_pathfinder(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    num_elbo_samples: int = 200,
    ftol: float = 1e-5,
) -> PathfinderFit:
    """Fit a forecasting model with BlackJAX Pathfinder variational inference.

    PRNG: ``rng_key`` is split into a model-initialization stream and a
    Pathfinder-approximation stream.

    Parameters
    ----------
    rng_key
        PRNG key for initialization and the Pathfinder run.
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    num_elbo_samples
        Number of Monte Carlo samples used to estimate the ELBO along the
        L-BFGS optimization path.
    ftol
        L-BFGS relative function-value tolerance (convergence criterion).

    Returns
    -------
    PathfinderFit
        The fitted variational approximation.
    """
    blackjax = require("blackjax", extra="blackjax")
    init_key, approx_key = random.split(rng_key)
    param_info, potential_fn_gen, _postprocess_fn, _ = initialize_model(
        init_key,
        model,
        dynamic_args=True,
        model_args=(covariates, data),
    )
    potential_fn = potential_fn_gen(covariates, data)

    def logdensity_fn(position: dict[str, Array]) -> Array:
        return -potential_fn(position)

    state, _info = blackjax.vi.pathfinder.approximate(
        approx_key,
        logdensity_fn,
        param_info.z,
        num_samples=num_elbo_samples,
        ftol=ftol,
    )
    return PathfinderFit(
        state=state,
        model=model,
        covariates=covariates,
        data=data,
        elbo=float(state.elbo),
    )


@_draw_posterior_impl.register
def _(fit: PathfinderFit, num_samples: int, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    blackjax = require("blackjax", extra="blackjax")
    # Split so the discarded init draws and the pathfinder draws use distinct
    # streams (the init param draws are unused, but keep the streams isolated).
    key_init, key_sample = random.split(rng_key)
    _param_info, _potential_fn_gen, postprocess_fn, _ = initialize_model(
        key_init,
        fit.model,
        dynamic_args=True,
        model_args=(fit.covariates, fit.data),
    )
    # sample() returns (samples_dict, log_weights); take the unconstrained draws
    # and map the constraining transform over the leading sample axis (no
    # batch_ndims reliance).
    unconstrained, _log_weights = blackjax.vi.pathfinder.sample(key_sample, fit.state, num_samples)
    transform = postprocess_fn(fit.covariates, fit.data)
    return jax.vmap(transform)(unconstrained)
