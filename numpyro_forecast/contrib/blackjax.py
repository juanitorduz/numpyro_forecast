"""BlackJAX MCMC kernels exposed as NumPyro :class:`~numpyro.infer.mcmc.MCMCKernel`.

This module adapts `BlackJAX <https://blackjax-devs.github.io/blackjax/>`_ sampling
algorithms to the NumPyro ``MCMCKernel`` interface so they can be passed straight to
:class:`~numpyro.infer.MCMC` like any built-in kernel.

BlackJAX is an optional dependency (``pip install numpyro_forecast[blackjax]``,
requiring ``blackjax>=1.6`` whose MCLMC tuning API this module targets) and is
never imported at package import time: it is pulled in lazily, via
:func:`numpyro_forecast.optional.require`, the first time a kernel is initialized. All three
kernels run one blackjax step per NumPyro sampling step; adaptation happens once inside
:meth:`_BlackjaxKernel.init`, so every run against one of these kernels must use
``chain_method="sequential"`` and ``num_warmup=0``: see the "Run configuration"
section on :class:`BlackjaxNUTSKernel`, :class:`BlackjaxMCLMCKernel`, and
:class:`BlackjaxCustomKernel` for why, and :exc:`~numpyro_forecast.exceptions.KernelConfigError`
for the one constraint (a kernel with no bound model) these kernels reject outright.

This module also exposes a two-tier Pathfinder variational-inference API. The
single-path tier, :func:`fit_pathfinder` and :func:`pathfinder_samples`, runs one
L-BFGS optimization path and returns a single local normal approximation: cheap,
but a single path can settle in one mode and miss the rest of a multimodal or
otherwise hard posterior. The multi-path tier, :func:`fit_multipathfinder` and
:func:`multipathfinder_samples`, runs several L-BFGS paths in parallel from
diverse starting points and combines them into one approximation that is not
tied to a single mode; it is the recommended entry point over the single-path
functions. How the paths are combined at drawing time is chosen by the
``resample`` argument of :func:`multipathfinder_samples`: Pareto-smoothed
importance sampling (PSIS) over the pooled draws when its ``pareto_k``
diagnostic says the weights are trustworthy, and ELBO-weighted sampling of whole
paths when it does not (PSIS weights degenerate in high dimensions).
"""

import importlib
import math
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax import random
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity

from numpyro_forecast.exceptions import KernelConfigError
from numpyro_forecast.functional._offload import _draw_chunked
from numpyro_forecast.functional._validation import _require_positive_num_samples
from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array, BlackjaxBuildFn, ForecastModel

_BJState = namedtuple("_BJState", ["position", "inner", "rng_key"])
"""MCMC state threaded by every :class:`_BlackjaxKernel`.

``position`` is the unconstrained sample dict (the NumPyro ``sample_field``),
``inner`` is the opaque blackjax algorithm state, and ``rng_key`` is the PRNG key
carried between steps.
"""

_PARETO_K_THRESHOLD = 0.7
"""Pareto shape-parameter value above which PSIS importance weights are unreliable.

The conventional threshold of Vehtari et al. (2017): below ``0.5`` importance
sampling is reliable, ``0.5`` to ``0.7`` is borderline, and above ``0.7`` the
weights are dominated by a handful of draws.
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
        The NumPyro forecasting model the kernel samples from. Pass it directly
        to the kernel constructor (``BlackjaxNUTSKernel(model, ...)``) before
        handing the instance to :class:`~numpyro.infer.MCMC`; a kernel
        constructed without a model is unbound and raises
        :exc:`~numpyro_forecast.exceptions.KernelConfigError` from :meth:`init`.
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
        KernelConfigError
            If the kernel has no bound model.
        TypeError
            If :meth:`_build` returns a state whose ``position`` key set differs
            from the initial one (a malformed ``build_fn``).
        """
        require("blackjax", extra="blackjax")
        if self._model is None:
            msg = (
                f"{type(self).__name__} has no bound model; construct it with the "
                "model as the first argument (e.g. BlackjaxNUTSKernel(model)) before "
                "passing it to MCMC."
            )
            raise KernelConfigError(msg)
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

    Run configuration
    -----------------
    Pass this kernel to :class:`~numpyro.infer.MCMC` with **``chain_method="sequential"``
    only**. With more than one chain, ``chain_method="vectorized"`` hands
    :meth:`~_BlackjaxKernel.init` a stacked ``rng_key`` of shape ``(num_chains, 2)``
    rather than vmapping a per-chain call, and the ``jax.random.split`` inside ``init``
    rejects a non-scalar key with a ``ValueError`` (pinned by the test suite).
    ``chain_method="parallel"`` is likewise unsupported: it is not exercised by this
    package and shares the same single-key assumption.
    Also pass **``num_warmup=0``**: adaptation runs once inside :meth:`~_BlackjaxKernel.init`
    (via ``blackjax.window_adaptation``), so any NumPyro-driven warmup steps on top
    of that are simply discarded work, not additional tuning.

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

    The step size, trajectory length ``L``, and diagonal preconditioner (inverse
    mass matrix) are tuned once in :meth:`~_BlackjaxKernel.init` via
    ``blackjax.mclmc_find_L_and_step_size``; each MCMC step is then a single
    tuned MCLMC step.

    Run configuration
    -----------------
    Pass this kernel to :class:`~numpyro.infer.MCMC` with **``chain_method="sequential"``
    only**. With more than one chain, ``chain_method="vectorized"`` hands
    :meth:`~_BlackjaxKernel.init` a stacked ``rng_key`` of shape ``(num_chains, 2)``
    rather than vmapping a per-chain call, and the ``jax.random.split`` inside ``init``
    rejects a non-scalar key with a ``ValueError`` (pinned by the test suite).
    ``chain_method="parallel"`` is likewise unsupported: it is not exercised by this
    package and shares the same single-key assumption.
    Also pass **``num_warmup=0``**: tuning runs once inside :meth:`~_BlackjaxKernel.init`
    (via ``blackjax.mclmc_find_L_and_step_size``), so any NumPyro-driven warmup steps
    on top of that are simply discarded work, not additional tuning.

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
        kernel = blackjax.mcmc.mclmc.build_kernel(
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )
        tuned_state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=self.num_tuning_steps,
            state=init_state,
            rng_key=tune_key,
            logdensity_fn=logdensity_fn,
        )
        # The tuner adapts step_size and L *for* the diagonal preconditioner it
        # estimates; dropping params.inverse_mass_matrix here would run a
        # mistuned identity-mass kernel on anisotropic posteriors.
        algorithm = blackjax.mclmc(
            logdensity_fn,
            L=params.L,
            step_size=params.step_size,
            inverse_mass_matrix=params.inverse_mass_matrix,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )
        return tuned_state, algorithm.step


class BlackjaxCustomKernel(_BlackjaxKernel):
    """Adapt an arbitrary BlackJAX sampler via a user-supplied ``build_fn``.

    The escape hatch for kernels without a dedicated wrapper. ``build_fn`` receives
    ``(rng_key, logdensity_fn, position, num_warmup)`` and must return
    ``(inner_state, step_fn)``, where ``inner_state`` exposes ``.position`` over the
    same sites as the model and ``step_fn`` has signature
    ``(rng_key, inner_state) -> (inner_state, info)``. The base class validates the
    returned state's key set and raises :class:`TypeError` if it is malformed.

    Run configuration
    -----------------
    Pass this kernel to :class:`~numpyro.infer.MCMC` with **``chain_method="sequential"``
    only**. With more than one chain, ``chain_method="vectorized"`` hands
    :meth:`~_BlackjaxKernel.init` a stacked ``rng_key`` of shape ``(num_chains, 2)``
    rather than vmapping a per-chain call, and the ``jax.random.split`` inside ``init``
    rejects a non-scalar key with a ``ValueError`` (pinned by the test suite).
    ``chain_method="parallel"`` is likewise unsupported: it is not exercised by this
    package and shares the same single-key assumption.
    Also pass **``num_warmup=0``**: whatever adaptation ``build_fn`` performs runs
    once inside :meth:`~_BlackjaxKernel.init`, so any NumPyro-driven warmup steps
    on top of that are simply discarded work, not additional tuning.

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


def _stable_bfgs_sample(
    rng_key: Array,
    num_samples: int | tuple[int, ...],
    position: Array,
    grad_position: Array,
    alpha: Array,
    beta: Array,
    gamma: Array,
) -> tuple[Array, Array]:
    """Numerically stable drop-in for blackjax's ``bfgs_sample``.

    Implements Algorithm 4 of Zhang et al. (2022), like the upstream function, but
    computes the log determinant as ``sum(log(alpha))`` plus twice the log diagonal
    of the Cholesky factor. Upstream (still as of blackjax 1.6) uses
    ``log(prod(alpha)) + 2 * log(det(L))``, whose product/determinant underflow to
    ``0`` once the problem has more than a few hundred dimensions, collapsing every
    ELBO estimate to ``-inf`` and the approximation's log density to ``+inf``.

    Parameters
    ----------
    rng_key
        PRNG key for the draws.
    num_samples
        Number of samples (or a shape tuple of sample axes).
    position
        Flattened current position on the L-BFGS path.
    grad_position
        Gradient of the target log density at ``position``.
    alpha
        Diagonal factor of the inverse-Hessian estimate.
    beta
        Low-rank factor of the inverse-Hessian estimate.
    gamma
        Core matrix of the low-rank factor.

    Returns
    -------
    tuple
        ``(samples, log_density)`` of the local normal approximation.
    """
    if not isinstance(num_samples, tuple):
        num_samples = (num_samples,)

    q_mat, r_mat = jnp.linalg.qr(jnp.diag(jnp.sqrt(1 / alpha)) @ beta)
    param_dims = beta.shape[0]
    identity_mat = jnp.identity(r_mat.shape[0])
    chol = jnp.linalg.cholesky(identity_mat + r_mat @ gamma @ r_mat.T)

    logdet = jnp.sum(jnp.log(alpha)) + 2 * jnp.sum(jnp.log(jnp.diagonal(chol)))
    mu = position + jnp.diag(alpha) @ grad_position + beta @ gamma @ beta.T @ grad_position

    u = jax.random.normal(rng_key, (*num_samples, param_dims, 1))
    phi = mu[..., None] + jnp.diag(jnp.sqrt(alpha)) @ (
        q_mat @ (chol - identity_mat) @ (q_mat.T @ u) + u
    )

    logdensity = -0.5 * (
        logdet + jnp.einsum("...ji,...ji->...", u, u) + param_dims * jnp.log(2.0 * jnp.pi)
    )
    return phi[..., 0], logdensity


def _ensure_stable_bfgs_sample() -> None:
    """Swap blackjax's ``bfgs_sample`` for :func:`_stable_bfgs_sample` (idempotent).

    ``blackjax.vi.pathfinder.approximate`` scores every point on the L-BFGS path
    with an ELBO estimate built on ``bfgs_sample``; the upstream log-determinant
    underflow (see :func:`_stable_bfgs_sample`) makes those estimates ``-inf``
    for any model beyond a few hundred parameters, so Pathfinder silently returns
    a garbage state. Until the fix lands upstream, the pathfinder entry points in
    this module patch both namespaces that hold the symbol before running.
    """
    lbfgs_module = importlib.import_module("blackjax.optimizers.lbfgs")
    pathfinder_module = importlib.import_module("blackjax.vi.pathfinder")
    lbfgs_module.bfgs_sample = _stable_bfgs_sample  # ty: ignore[invalid-assignment]
    pathfinder_module.bfgs_sample = _stable_bfgs_sample  # ty: ignore[invalid-assignment]


@dataclass(frozen=True)
class PathfinderFit:
    """The result of fitting a forecasting model with BlackJAX Pathfinder.

    A plain-data (picklable) container: it holds the raw blackjax
    ``PathfinderState``, the model and its data/covariates (needed to rebuild the
    unconstrained-to-constrained transform when drawing), and the fitted ELBO.
    Draws are produced lazily by :func:`pathfinder_samples`.

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
    maxiter: int = 30,
    maxcor: int = 10,
    maxls: int = 1_000,
    gtol: float = 1e-8,
    init_params: dict[str, Array] | None = None,
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
    maxiter
        Maximum number of L-BFGS iterations (blackjax's default is ``30``).
        Models with per-step time latents typically need far more: the default
        suits tens of parameters, while a few hundred parameters (e.g. a
        400-step random walk) converge around ``1_000``.
    maxcor
        L-BFGS history size; caps the rank of the low-rank-plus-diagonal
        covariance correction at roughly ``2 * maxcor``. High-dimensional
        posteriors need it raised well above the default of ``10``.
    maxls
        Maximum number of line-search steps per L-BFGS iteration.
    gtol
        L-BFGS gradient-norm convergence tolerance.
    init_params
        Optional initial unconstrained position, overriding NumPyro's own
        initialization draw (``param_info.z``); ``initialize_model`` still runs
        regardless, since it also builds the potential/postprocess functions.

    Returns
    -------
    PathfinderFit
        The fitted variational approximation.

    See Also
    --------
    fit_multipathfinder
        The recommended entry point: runs several L-BFGS paths in parallel and
        combines them (see :func:`multipathfinder_samples`), instead of relying
        on this single path's local normal approximation.

    Notes
    -----
    Runs with :func:`_stable_bfgs_sample` patched into blackjax (via
    :func:`_ensure_stable_bfgs_sample`): upstream's ``bfgs_sample`` underflows its
    log-determinant beyond a few hundred parameters, which floors every path ELBO
    at ``-inf`` and makes the returned state garbage.
    """
    blackjax = require("blackjax", extra="blackjax")
    _ensure_stable_bfgs_sample()
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

    position = param_info.z if init_params is None else init_params
    state, _info = blackjax.vi.pathfinder.approximate(
        approx_key,
        logdensity_fn,
        position,
        num_samples=num_elbo_samples,
        maxiter=maxiter,
        maxcor=maxcor,
        maxls=maxls,
        gtol=gtol,
        ftol=ftol,
    )
    return PathfinderFit(
        state=state,
        model=model,
        covariates=covariates,
        data=data,
        elbo=float(state.elbo),
    )


def pathfinder_samples(
    rng_key: Array,
    fit: PathfinderFit,
    num_samples: int,
    *,
    batch_size: int | None = None,
    device: jax.Device | str | None = None,
) -> dict[str, Array]:
    """Draw ``num_samples`` posterior samples from a fitted Pathfinder approximation.

    The returned dict has the sample axis leading and is ready to pass to
    :func:`~numpyro_forecast.functional.prediction.forecast` or NumPyro's
    ``Predictive``, exactly like :func:`~numpyro_forecast.functional.posterior.draw_posterior`.
    Chunking and device offload are delegated to the shared
    :func:`~numpyro_forecast.functional._offload._draw_chunked` driver, so the same
    memory-bounding contract applies.

    PRNG: within each chunk (the whole draw, when unchunked), the chunk key is
    split into a model-initialization stream and a Pathfinder-sampling stream (the
    init param draws are unused, but the streams are kept isolated).

    Parameters
    ----------
    rng_key
        PRNG key.
    fit
        A fit from :func:`fit_pathfinder`.
    num_samples
        Number of posterior draws.
    batch_size
        Optional chunk size for the drawing itself; see
        :func:`~numpyro_forecast.functional.posterior.draw_posterior` (the same
        memory/reproducibility contract applies).
    device
        Where each chunk of draws is moved as soon as it is drawn; see
        :func:`~numpyro_forecast.functional.posterior.draw_posterior`, including
        for the JAX rule that arithmetic mixing a host-committed result with a
        device-resident array raises rather than running on the accelerator.

    Returns
    -------
    dict[str, Array]
        Posterior samples of the latent sites, sample axis leading (leaves
        committed to host memory when ``device`` resolves to ``"host"``).

    Raises
    ------
    ValueError
        If ``num_samples`` or ``batch_size`` is not positive.
    RuntimeError
        If ``device`` resolves to ``"host"`` and the array's device exposes no
        host memory kind (see
        :func:`~numpyro_forecast.functional._offload._host_memory_kind`).
    """
    _require_positive_num_samples(num_samples)
    blackjax = require("blackjax", extra="blackjax")
    _ensure_stable_bfgs_sample()

    def draw_fn(key: Array, n: int) -> dict[str, Array]:
        # Split so the discarded init draws and the pathfinder draws use distinct
        # streams (the init param draws are unused, but keep the streams isolated).
        key_init, key_sample = random.split(key)
        _param_info, _potential_fn_gen, postprocess_fn, _ = initialize_model(
            key_init,
            fit.model,
            dynamic_args=True,
            model_args=(fit.covariates, fit.data),
        )
        # sample() returns (samples_dict, log_weights); take the unconstrained
        # draws and map the constraining transform over the leading sample axis
        # (no batch_ndims reliance).
        unconstrained, _log_weights = blackjax.vi.pathfinder.sample(key_sample, fit.state, n)
        transform = postprocess_fn(fit.covariates, fit.data)
        return jax.vmap(transform)(unconstrained)

    return _draw_chunked(
        rng_key,
        draw_fn,
        num_samples,
        batch_size=batch_size,
        device=device,
        stage="posterior drawing",
    )


@dataclass(frozen=True)
class MultiPathfinderFit:
    """The result of fitting a forecasting model with multi-path BlackJAX Pathfinder.

    A plain-data (picklable) container, mirroring :class:`PathfinderFit`: it holds
    the raw blackjax ``MultipathfinderState`` (the pooled per-path approximations),
    the model and its data/covariates, the per-path ELBOs, and the Pareto-smoothed
    importance sampling (PSIS) weights/diagnostic over the pooled draws. Draws are
    produced lazily by :func:`multipathfinder_samples`.

    Attributes
    ----------
    state
        The blackjax ``MultipathfinderState`` (per-path ``PathfinderState`` objects,
        the pooled samples, and their log target/approximation densities).
    model
        The forecasting model that was fit.
    covariates
        In-sample covariates used at fit time (time at axis ``-2``).
    data
        In-sample data used at fit time (time at axis ``-2``).
    elbos
        The evidence lower bound of each path's fitted approximation, in path
        order; converted eagerly with ``float()`` so pickling this fit never
        carries JAX tracers.
    log_weights
        Normalized PSIS log importance weights over the flattened pool of
        ``num_paths * num_elbo_samples`` draws (path-major, matching
        ``state.samples``); a fit-time diagnostic, valid only for that exact
        stored pool. :func:`multipathfinder_samples` draws fresh samples and
        recomputes its own weights, so it never consumes these.
    pareto_k
        The Pareto shape-parameter diagnostic for the fit-time PSIS weights:
        below ``0.5`` is reliable, ``0.5`` to ``0.7`` is borderline, and above
        ``0.7`` indicates the importance weights are unreliable.
        :func:`multipathfinder_samples` reads it to decide whether
        ``resample="auto"`` uses PSIS or ELBO-weighted path sampling.
    """

    state: Any
    model: ForecastModel
    covariates: Array
    data: Array
    elbos: tuple[float, ...]
    log_weights: Array
    pareto_k: float


def fit_multipathfinder(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    num_paths: int = 4,
    num_elbo_samples: int = 200,
    maxiter: int = 30,
    maxcor: int = 10,
    maxls: int = 1_000,
    gtol: float = 1e-8,
    ftol: float = 1e-5,
    initial_positions: dict[str, Array] | None = None,
) -> MultiPathfinderFit:
    """Fit a forecasting model with multi-path BlackJAX Pathfinder and PSIS resampling.

    Runs ``num_paths`` independent single-path Pathfinder approximations in
    parallel (vmapped L-BFGS runs, by default each from its own fresh
    ``init_to_uniform`` starting point) and scores the pooled per-path draws
    with Pareto-smoothed importance sampling (PSIS), so the returned fit is not
    tied to a single mode. This is the recommended entry point over
    :func:`fit_pathfinder`: a single L-BFGS path can settle in one mode and its
    local normal approximation may not reflect the rest of a multimodal or
    otherwise hard posterior. The PSIS weights computed here are a fit-time
    diagnostic; the draws themselves come from :func:`multipathfinder_samples`,
    which resamples fresh per-path draws and reads ``pareto_k`` to pick between
    PSIS and ELBO-weighted path sampling.

    PRNG: ``rng_key`` is split into a model-initialization stream and a
    multipath-approximation stream; the initialization stream is further split
    into one subkey per path so that (when ``initial_positions`` is not
    supplied) every path starts from its own independent ``init_to_uniform``
    draw, exactly the diverse starting points multipath Pathfinder wants.

    Parameters
    ----------
    rng_key
        PRNG key for initialization and the multipath Pathfinder run.
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    num_paths
        Number of independent L-BFGS paths to run in parallel (vmapped).
    num_elbo_samples
        Number of Monte Carlo samples drawn per path to estimate its ELBO and
        to build the pooled sample used for PSIS resampling.
    maxiter
        Maximum number of L-BFGS iterations per path.
    maxcor
        L-BFGS history size; caps the rank of the low-rank-plus-diagonal
        covariance correction at roughly ``2 * maxcor``. High-dimensional
        posteriors need it raised well above the default of ``10``.
    maxls
        Maximum number of line-search steps per L-BFGS iteration.
    gtol
        L-BFGS gradient-norm convergence tolerance.
    ftol
        L-BFGS relative function-value convergence tolerance.
    initial_positions
        Optional starting positions, one per path, overriding the default
        per-path ``init_to_uniform`` draws. Every leaf must already carry a
        leading axis of size ``num_paths`` (validated; a mismatch raises
        ``ValueError``).

    Returns
    -------
    MultiPathfinderFit
        The fitted multipath approximation together with its per-path ELBOs
        and the fit-time PSIS log weights/``pareto_k`` diagnostic over the
        pooled draws.

    Raises
    ------
    ValueError
        If ``num_paths`` is not positive, or ``initial_positions`` is supplied
        without a leading axis of size ``num_paths`` on every leaf.

    Warns
    -----
    UserWarning
        If one or more per-path ELBOs are non-finite (raise ``maxiter``/``maxcor``
        or inspect ``fit.elbos``), or if the PSIS ``pareto_k`` diagnostic exceeds
        ``0.7``, in which case ``multipathfinder_samples(..., resample="auto")``
        falls back to ELBO-weighted path sampling (increase
        ``num_paths``/``maxiter``/``maxcor``, or fall back to MCMC). Neither
        condition raises: the fit is always returned, carrying everything needed
        to inspect and diagnose it.

    Notes
    -----
    Runs with :func:`_stable_bfgs_sample` patched into blackjax (via
    :func:`_ensure_stable_bfgs_sample`, called before
    ``blackjax.vi.multipathfinder.multi_approximate``): ``multi_approximate``
    calls ``approximate``/``sample`` imported from ``blackjax.vi.pathfinder``,
    whose module globals hold ``bfgs_sample``, so patching that module (as
    :func:`fit_pathfinder` already does) covers the multipath route too.
    """
    if num_paths < 1:
        msg = "num_paths must be positive"
        raise ValueError(msg)
    if initial_positions is not None:
        leaves = jax.tree.leaves(initial_positions)
        if not leaves or any(jnp.ndim(leaf) == 0 or leaf.shape[0] != num_paths for leaf in leaves):
            msg = (
                "initial_positions must carry a leading axis of size "
                f"num_paths={num_paths} on every leaf"
            )
            raise ValueError(msg)

    require("blackjax", extra="blackjax")
    _ensure_stable_bfgs_sample()
    init_key, approx_key = random.split(rng_key)
    init_keys = random.split(init_key, num_paths)

    if initial_positions is None:
        positions = []
        for key in init_keys:
            param_info, potential_fn_gen, _postprocess_fn, _ = initialize_model(
                key,
                model,
                dynamic_args=True,
                model_args=(covariates, data),
            )
            positions.append(param_info.z)
        stacked_positions = jax.tree.map(lambda *xs: jnp.stack(xs), *positions)
    else:
        # potential_fn_gen depends only on the model/data, not the init key or
        # strategy, so a single initialize_model call supplies it here exactly
        # like the per-path loop above does when init_positions is None.
        _param_info, potential_fn_gen, _postprocess_fn, _ = initialize_model(
            init_keys[0],
            model,
            dynamic_args=True,
            model_args=(covariates, data),
        )
        stacked_positions = initial_positions

    potential_fn = potential_fn_gen(covariates, data)

    def logdensity_fn(position: dict[str, Array]) -> Array:
        return -potential_fn(position)

    multipathfinder_module = importlib.import_module("blackjax.vi.multipathfinder")
    state, _info = multipathfinder_module.multi_approximate(
        approx_key,
        logdensity_fn,
        stacked_positions,
        num_elbo_samples,
        maxiter=maxiter,
        maxcor=maxcor,
        maxls=maxls,
        gtol=gtol,
        ftol=ftol,
    )
    log_weights, pareto_k = multipathfinder_module.psis_weights(state)
    pareto_k_value = float(pareto_k)

    elbos = tuple(float(elbo) for elbo in state.path_states.elbo)
    n_failed = sum(1 for elbo in elbos if not math.isfinite(elbo))
    if n_failed:
        msg = (
            f"{n_failed} of {num_paths} path(s) failed (non-finite ELBO); raise "
            "maxiter/maxcor or inspect elbos."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    if pareto_k_value > _PARETO_K_THRESHOLD:
        msg = (
            f"pareto_k={pareto_k_value:.2f} > {_PARETO_K_THRESHOLD}: PSIS importance "
            "weights over the pooled draws are unreliable, so "
            'multipathfinder_samples(..., resample="auto") falls back to '
            "ELBO-weighted path sampling instead of PSIS resampling; increase "
            "num_paths/maxiter/maxcor or fall back to MCMC."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    return MultiPathfinderFit(
        state=state,
        model=model,
        covariates=covariates,
        data=data,
        elbos=elbos,
        log_weights=log_weights,
        pareto_k=pareto_k_value,
    )


def multipathfinder_samples(
    rng_key: Array,
    fit: MultiPathfinderFit,
    num_samples: int,
    *,
    resample: Literal["auto", "psis", "elbo"] = "auto",
    batch_size: int | None = None,
    device: jax.Device | str | None = None,
) -> dict[str, Array]:
    """Draw ``num_samples`` posterior samples from a fitted multipath Pathfinder fit.

    Every call draws fresh unconstrained samples from each path's fitted normal
    approximation (``num_samples`` per path, via ``blackjax.vi.pathfinder.sample``
    vmapped over the paths) and then combines the paths into a single set of
    ``num_samples`` draws. Nothing is recycled from the small pool stored at fit
    time, so asking for more draws than that pool held costs nothing in
    duplication. The output contract is identical to
    :func:`~numpyro_forecast.functional.posterior.draw_posterior` and
    :func:`pathfinder_samples`: :func:`~numpyro_forecast.functional.prediction.forecast`
    and NumPyro's ``Predictive`` consume it unchanged. Chunking and device offload
    are delegated to the shared
    :func:`~numpyro_forecast.functional._offload._draw_chunked` driver, so the same
    memory-bounding contract applies.

    Resampling strategies
    ---------------------
    ``resample="psis"``
        Pool the ``num_paths * num_samples`` fresh draws, score them under the
        model (``logp``) and under their own path's approximation (``logq``),
        Pareto-smooth the importance ratios, and resample ``num_samples`` of them
        with replacement. This is the textbook multipath Pathfinder estimator and
        the most faithful one *when the importance weights behave*.
    ``resample="elbo"``
        Pick a whole path per draw, with probability ``softmax(fit.elbos)`` over
        the paths whose ELBO is finite (non-finite paths get zero weight, and an
        all-non-finite fit degrades to a uniform choice), then take one fresh draw
        from that path. With ELBO gaps of hundreds of nats this collapses onto the
        single best path, which is the right answer when the paths disagree
        strongly: no single draw is ever duplicated.
    ``resample="auto"`` (default)
        ``"psis"`` when ``fit.pareto_k <= 0.7``, ``"elbo"`` otherwise.

    The gate matters because importance sampling degenerates in high dimensions:
    on a posterior with hundreds of parameters the ratio ``logp - logq`` is
    dominated by a handful of draws, ``pareto_k`` climbs far above ``0.7``, and
    PSIS resampling collapses the answer onto those few draws. ELBO-weighted path
    sampling never concentrates like that, because it reweights *paths* (a handful
    of well-separated numbers) rather than individual draws.

    PRNG: within each chunk (the whole draw, when unchunked), the chunk key is
    split into a model-initialization stream (used only to rebuild the
    constraining transform), a per-path sampling stream, and a resampling-index
    stream. The split is the same in every mode, so ``resample="auto"`` produces
    bitwise the same draws as the explicit mode it resolves to.

    Parameters
    ----------
    rng_key
        PRNG key.
    fit
        A fit from :func:`fit_multipathfinder`.
    num_samples
        Number of posterior draws.
    resample
        How to combine the per-path draws: ``"auto"`` (default), ``"psis"``, or
        ``"elbo"``; see "Resampling strategies" above.
    batch_size
        Optional chunk size for the drawing itself; see
        :func:`~numpyro_forecast.functional.posterior.draw_posterior` (the same
        memory/reproducibility contract applies). Note that each chunk draws
        ``num_paths * batch_size`` samples internally.
    device
        Where each chunk of draws is moved as soon as it is drawn; see
        :func:`~numpyro_forecast.functional.posterior.draw_posterior`, including
        for the JAX rule that arithmetic mixing a host-committed result with a
        device-resident array raises rather than running on the accelerator.

    Returns
    -------
    dict[str, Array]
        Posterior samples of the latent sites, sample axis leading (leaves
        committed to host memory when ``device`` resolves to ``"host"``).

    Raises
    ------
    ValueError
        If ``num_samples`` or ``batch_size`` is not positive, or ``resample`` is
        not one of ``"auto"``, ``"psis"``, ``"elbo"``.
    RuntimeError
        If ``device`` resolves to ``"host"`` and the array's device exposes no
        host memory kind (see
        :func:`~numpyro_forecast.functional._offload._host_memory_kind`).

    Warns
    -----
    UserWarning
        In PSIS mode, if the ``pareto_k`` recomputed on this call's fresh draws
        exceeds ``0.7``. This is a separate diagnostic from ``fit.pareto_k``,
        which is computed once over the pool stored at fit time.
    """
    _require_positive_num_samples(num_samples)
    if resample not in ("auto", "psis", "elbo"):
        msg = f"resample must be one of 'auto', 'psis', 'elbo', got {resample!r}"
        raise ValueError(msg)
    blackjax = require("blackjax", extra="blackjax")
    _ensure_stable_bfgs_sample()

    if resample == "auto":
        mode = "psis" if fit.pareto_k <= _PARETO_K_THRESHOLD else "elbo"
    else:
        mode = resample

    num_paths = len(fit.elbos)
    elbos = jnp.asarray(fit.elbos)
    finite = jnp.isfinite(elbos)
    # softmax over the finite-ELBO paths; a fit whose every path failed has no
    # signal to weight by, so it degrades to a uniform choice over the paths.
    path_weights = (
        jax.nn.softmax(jnp.where(finite, elbos, -jnp.inf))
        if bool(jnp.any(finite))
        else jnp.full((num_paths,), 1.0 / num_paths)
    )

    def draw_fn(key: Array, n: int) -> dict[str, Array]:
        # Three isolated streams: the (unused) init param draws, the per-path
        # Pathfinder draws, and the path/index choice.
        key_init, key_paths, key_choice = random.split(key, 3)
        _param_info, potential_fn_gen, postprocess_fn, _ = initialize_model(
            key_init,
            fit.model,
            dynamic_args=True,
            model_args=(fit.covariates, fit.data),
        )
        # sample() returns (samples_dict, logq) with the draws already unravelled
        # into the model's site dict; vmapping it over the stacked per-path states
        # gives leaves of shape (num_paths, n, ...) and logq of shape (num_paths, n).
        path_keys = random.split(key_paths, num_paths)
        draws, logq = jax.vmap(
            lambda path_key, path_state: blackjax.vi.pathfinder.sample(path_key, path_state, n)
        )(path_keys, fit.state.path_states)

        if mode == "elbo":
            path_index = jax.random.choice(
                key_choice, num_paths, shape=(n,), replace=True, p=path_weights
            )
            unconstrained = jax.tree.map(
                lambda x: x[path_index, jnp.arange(n)],
                draws,
            )
        else:
            flat_draws = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), draws)
            potential_fn = potential_fn_gen(fit.covariates, fit.data)
            logp = jax.vmap(lambda position: -potential_fn(position))(flat_draws)
            log_w, pareto_k = blackjax.diagnostics.psis_weights(logp - logq.reshape(-1))
            pareto_k_value = float(pareto_k)
            if pareto_k_value > _PARETO_K_THRESHOLD:
                msg = (
                    f"pareto_k={pareto_k_value:.2f} > {_PARETO_K_THRESHOLD} on this "
                    'call\'s fresh draws: PSIS resampling is unreliable; pass resample="elbo" '
                    '(or resample="auto") to weight whole paths by their ELBO instead.'
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            indices = jax.random.choice(
                key_choice,
                log_w.shape[0],
                shape=(n,),
                replace=True,
                p=jnp.exp(log_w),
            )
            unconstrained = jax.tree.map(lambda x: x[indices], flat_draws)

        transform = postprocess_fn(fit.covariates, fit.data)
        return jax.vmap(transform)(unconstrained)

    return _draw_chunked(
        rng_key,
        draw_fn,
        num_samples,
        batch_size=batch_size,
        device=device,
        stage="posterior drawing",
    )
