r"""dynestyx state space models as a `numpyro_forecast` building block.

[dynestyx](https://github.com/BasisResearch/dynestyx) describes a dynamical
system as a ``DynamicalModel`` (initial condition, state evolution, observation
model) and interprets its ``dsx.sample`` primitive with effect handlers: a
``Filter`` or ``Smoother`` integrates the latent path out and adds the marginal
log likelihood to the NumPyro trace, a ``LatentPathBuilder`` samples the path
explicitly, and a ``Simulator`` rolls the conditioned state forward.

`state_space()` is the building block that connects the two libraries. Like
`~~numpyro_forecast.models.ssoe()`, it is called inside a model function: it
turns the call's `~~numpyro_forecast.models.Horizon` into the dynestyx time
grids, calls ``dsx.sample`` once, and registers the two sites the drivers read
by name:

- **training** (``data`` given, no horizon): nothing beyond dynestyx's own
  marginal-likelihood (or joint) factor;
- **forecast** (``data`` given, a horizon): ``"forecast"`` from the Simulator's
  rollout of the final conditioned state;
- **in-sample predictive** (``data=None``): ``"obs"`` from a draw of the
  in-sample states pushed through the observation model. Under a ``Smoother``
  the states are drawn step by step from the smoothing marginals
  $p(x_t \mid y_{1:T}, \theta)$ (independent across steps: exact for the
  per-step bands, CRPS and coverage that the posterior predictive group is used
  for, not a joint path); under a ``LatentPathBuilder`` the state path is the
  posterior draw itself, a joint path. A ``Filter`` has only the filtering
  marginals $p(x_t \mid y_{1:t}, \theta)$, so the block refuses it for the
  in-sample predictive.

**The handler goes around every call that runs the model**, never inside it:
``mcmc.run``, ``svi.run``, `~~numpyro_forecast.predictive.draw_posterior()`
(an ``AutoGuide`` re-runs the model to compute deterministic sites),
`~~numpyro_forecast.predictive.forecast()`,
`~~numpyro_forecast.predictive.predict_in_sample()`,
`~~numpyro_forecast.convert.to_datatree()`,
`~~numpyro_forecast.evaluate.backtest()`, and plain ``Predictive`` or
``log_density`` calls. This composes because the drivers compile once per call,
under the handlers active at that call. One stack serves every driver:

```python
with DiscreteTimeSimulator(n_simulations=1), Smoother(smoother_config=...):
    tree = to_datatree(rng_key, model, posterior, data, covariates)
```

The block raises when no ``Filter``, ``Smoother`` or ``LatentPathBuilder`` is
active. Without one, dynestyx 0.8 would pick a default itself and only log the
choice, so a forgotten handler would fit or forecast under an inference method
nobody chose. The check reads dynestyx's handler stack through a private
dynestyx query (``dynestyx.handlers._dynestyx_stack_kind``), which a canary test
pins.

Two dynestyx conventions shape the block. The observed window travels in the
covariates (the drivers call the model with ``data=None`` for the in-sample
predictive, as with `~~numpyro_forecast.models.ssoe()`), so the caller passes
``y`` and the likelihood reads it, NaN entries included, never ``h.data``. And
the discrete-time simulator places the rollout's initial state at the first
predict time, so the block asks for one extra step at the last observation and
drops it: the first forecast step is a transition of the final state, not a copy
of it.

The coupling to dynestyx is confined to this module: ``dynestyx.sample``, the
handler-stack query above, the result types ``LatentStateResult``
(``state_path``) and the ``dists`` of a smoother's result, the site suffixes
``_marginal_loglik``, ``_state_path_params``, ``_predicted_observations`` and
``_predicted_states``, and ``DynamicalModel.observation_model(x, u, t)``.
dependency (``pip install numpyro_forecast[dynestyx]``, ``dynestyx>=0.8.0,<0.9``)
imported lazily through `~~numpyro_forecast.optional.require()` at the first
call, which warns when the installed minor version is not the tested one. The
block models one series per model function: it has no batch dims (``dsx.plate``
registers no state sites) and it owns the ``"obs"`` and ``"forecast"`` sites,
which the drivers read by their bare names, so it cannot run under
``handlers.scope`` or be called twice in one model.
"""

import importlib
import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import random
from jaxtyping import Float

from numpyro_forecast.models import Horizon
from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array

_TESTED_DYNESTYX = (0, 8)
"""The dynestyx minor version the block is tested with; other versions warn."""

_PREDICTED_OBSERVATIONS = "_predicted_observations"
"""Suffix of the site a dynestyx Simulator registers with the observation rollout."""

_PREDICTED_STATES = "_predicted_states"
"""Suffix of the site a dynestyx Simulator registers with the state rollout."""

_CONDITIONING_SITES = {
    "FILTER": "_marginal_loglik",
    "SMOOTHER": "_marginal_loglik",
    "LATENT_PATH_BUILDER": "_state_path_params",
}
"""dynestyx handler kinds that condition ``dsx.sample`` on the observed window, each with the
suffix of the site it registers in every mode (the marginal likelihood, or the path)."""

_IDLE_SIMULATOR_WARNING = "Simulator has no predict_times to simulate at."
"""dynestyx's warning on the in-sample legs of the one-stack recipe; silenced when future == 0."""


class StateSpaceResult(NamedTuple):
    """Latent state draws of one `state_space()` call, ``None`` outside their mode.

    The block registers the ``"forecast"`` and ``"obs"`` sites itself; the states
    are returned so that a model can register its latent level (for example
    ``numpyro.deterministic("level", result.x_in_sample)`` when ``h.data is None``)
    under any handler that serves the mode.

    Attributes
    ----------
    x_in_sample
        One draw of the in-sample states, shape ``(t_obs, state)``, when the model
        is called without data: the draw behind the ``"obs"`` site (per-step
        smoothing marginals under a ``Smoother``, the posterior path under a
        ``LatentPathBuilder``). ``None`` otherwise.
    x_future
        The Simulator's state rollout over the horizon, shape ``(future, state)``,
        when forecasting; ``None`` otherwise.
    """

    x_in_sample: Float[Array, " time state"] | None
    x_future: Float[Array, " future state"] | None


def _check_version(dsx: Any) -> None:
    """Warn when the installed dynestyx minor version is not the tested one."""
    version = str(getattr(dsx, "__version__", "unknown"))
    if version.split(".")[:2] != [str(part) for part in _TESTED_DYNESTYX]:
        msg = (
            f"numpyro_forecast.contrib.dynestyx is tested with dynestyx "
            f"{'.'.join(map(str, _TESTED_DYNESTYX))}, found {version}; dynestyx changes its "
            "handler semantics between minor releases"
        )
        warnings.warn(msg, UserWarning)  # noqa: B028 - wrapper frames make stacklevel misleading


def _conditioning_kind() -> str:
    """Return the active conditioning handler kind, or raise when none is active.

    Reads dynestyx's handler stack (a private dynestyx query, pinned by the canary
    test); it works at trace time, so also inside the jitted drivers.
    """
    handlers = importlib.import_module("dynestyx.handlers")
    kinds = [kind.name for kind in handlers._dynestyx_stack_kind()]
    conditioning = [kind for kind in kinds if kind in _CONDITIONING_SITES]
    if not conditioning:
        msg = (
            "state_space needs a dynestyx Filter, Smoother or LatentPathBuilder around the call "
            f"(active dynestyx handlers: {kinds or 'none'}); without one dynestyx picks a default "
            "inference method itself, so pass the handler of the fit explicitly: a Smoother of "
            "the fit's configuration for a Filter or Smoother fit, or the LatentPathBuilder "
            "instance that fitted the model"
        )
        raise ValueError(msg)
    return conditioning[0]


def _validate_window(h: Horizon, y: Array, controls: Array | None) -> None:
    """Check the time axes against the horizon (rank is enforced by the jaxtyping annotations)."""
    if y.shape[-2] != h.t_obs:
        msg = f"y must have shape (t_obs={h.t_obs}, obs), got {y.shape}"
        raise ValueError(msg)
    if controls is not None and controls.shape[-2] != h.duration:
        msg = f"controls must have shape (duration={h.duration}, control), got {controls.shape}"
        raise ValueError(msg)


def _rng_key() -> Array:
    """Return the active seed handler's key, which the in-sample draws consume."""
    with warnings.catch_warnings():  # numpyro warns before returning None; the error says it all
        warnings.simplefilter("ignore")
        key = numpyro.prng_key()
    if key is None:
        msg = (
            "the in-sample predictive of state_space draws the states and observations, so the "
            "model must run under a seed handler (Predictive, predict_in_sample, to_datatree)"
        )
        raise RuntimeError(msg)
    return key


def _in_sample_states(rng_key: Array, result: Any, kind: str, t_obs: int) -> Array:
    """One in-sample state draw ``(t_obs, state)``: the posterior path, or the smoothing marginals.

    Every step's marginal is a numpyro ``Distribution`` whose pytree aux data
    carries its (unbatched) shape, so the marginals cannot be stacked into one
    batched distribution; instead the leaves are stacked and each step is
    unflattened and sampled under ``jax.vmap``.
    """
    if kind == "FILTER":
        msg = (
            "the in-sample predictive needs the smoothing distribution: run predict_in_sample / "
            "to_datatree under a dynestyx Smoother or a LatentPathBuilder (a Filter only has the "
            "filtering marginals; the draws of a Filter fit are valid under the Smoother that "
            "runs the same forward filter, which adds the same marginal likelihood)"
        )
        raise ValueError(msg)
    if kind == "LATENT_PATH_BUILDER":
        x = jnp.asarray(result.state_path)
    else:
        dists = getattr(result, "dists", None)
        if not dists:  # pragma: no cover - defensive: no discrete-time smoother of dynestyx 0.8
            msg = "the dynestyx Smoother returned no per-step marginals to draw from"
            raise ValueError(msg)
        treedef = jax.tree.structure(dists[0])
        leaves = [
            jnp.stack(step) for step in zip(*(jax.tree.leaves(d) for d in dists), strict=True)
        ]

        def sample_step(step_leaves: list[Array], key: Array) -> Array:
            return jnp.asarray(jax.tree.unflatten(treedef, step_leaves).sample(key))

        x = jax.vmap(sample_step)(leaves, random.split(rng_key, len(dists)))
    if x.ndim != 2 or x.shape[0] != t_obs:
        msg = (
            f"state_space needs one state vector per observed step, shape (t_obs={t_obs}, state), "
            f"got a state path of shape {x.shape}: write a scalar state as a length-1 vector "
            "(e.g. dist.Normal(...).expand([1]).to_event(1)); continuous-time paths with an "
            "extra initial row are not supported"
        )
        raise ValueError(msg)
    return x


def _observe(rng_key: Array, dynamics: Any, x: Array, u: Array | None, times: Array) -> Array:
    """Draw ``y_t ~ observation_model(x_t, u_t, t)`` for every in-sample step, ``(t_obs, obs)``.

    A scalar-event observation model gets its event axis appended, as dynestyx
    does for its own rollout, so the site keeps the package's ``(time, obs)`` layout.
    """

    def sample_step(x_t: Array, u_t: Array | None, t: Array, key: Array) -> Array:
        draw = jnp.asarray(dynamics.observation_model(x_t, u_t, t).sample(key))
        return draw[..., None] if draw.ndim == 0 else draw

    keys = random.split(rng_key, x.shape[0])
    return jax.vmap(sample_step, in_axes=(0, None if u is None else 0, 0, 0))(x, u, times, keys)


def _check_site_names(trace: dict[str, Any], name: str, kind: str) -> None:
    """Raise unless the site the active handler registers in every mode keeps its bare name.

    A Filter or Smoother registers ``f"{name}_marginal_loglik"``, a LatentPathBuilder
    ``f"{name}_state_path_params"``. Under ``numpyro.handlers.scope`` every site is
    renamed, including the ``"obs"`` and ``"forecast"`` sites the drivers read by name.
    """
    if f"{name}{_CONDITIONING_SITES[kind]}" not in trace:
        msg = (
            f"dynestyx registered no {name + _CONDITIONING_SITES[kind]!r} site: state_space "
            "reads its sites by their bare names and owns the bare 'obs' and 'forecast' sites, "
            "so it cannot run under numpyro.handlers.scope (which renames them)"
        )
        raise ValueError(msg)


def state_space(
    h: Horizon,
    name: str,
    y: Float[Array, " time obs"],
    dynamics: Any,
    *,
    controls: Float[Array, " duration control"] | None = None,
) -> StateSpaceResult:
    r"""Condition, forecast or predict in-sample a dynestyx model over the horizon.

    One ``dsx.sample`` call on the integer time grid ``0, ..., h.duration - 1``
    (``obs_times`` are the first ``h.t_obs`` steps, ``predict_times`` the rest,
    ``ctrl_times`` the whole grid), interpreted by the dynestyx handlers active
    around the model call. What the block registers depends on the mode of the
    call, exactly as the other building blocks decide it from ``h``:

    - ``h.data`` given and ``h.future == 0`` (training): nothing. The active
      ``Filter``, ``Smoother`` or ``LatentPathBuilder`` adds its likelihood
      factor, which is what ``MCMC`` and ``SVI`` fit.
    - ``h.data`` given and ``h.future > 0`` (forecasting): the site
      ``"forecast"``, shape ``(future, obs)``, read from the rollout of a
      ``DiscreteTimeSimulator(n_simulations=1)`` placed outside the ``Filter``,
      ``Smoother`` or ``LatentPathBuilder`` (the explicit simulator is the
      recommended form; dynestyx 0.8 adds one itself when none is active). The
      rollout starts from the conditioned state at the last observation: the
      filtering or smoothing distribution at the parameters of each posterior
      draw (the two coincide there), or the sampled final state of the path
      under a ``LatentPathBuilder``, which therefore inherits the mixing of the
      path. The block asks the simulator for ``predict_times`` starting at the
      last observation and drops that first row, because dynestyx places the
      rollout's initial state at the first predict time without a transition.
    - ``h.data is None`` (the in-sample predictive of
      `~~numpyro_forecast.predictive.predict_in_sample()` and
      `~~numpyro_forecast.convert.to_datatree()`): the site ``"obs"``, shape
      ``(t_obs, obs)``, one observation draw per step from
      ``dynamics.observation_model(x_t, u_t, t)`` at a draw of the in-sample
      states: step-wise from the smoothing marginals under a ``Smoother``, the
      posterior path itself under a ``LatentPathBuilder``. This is not a prior
      predictive: the block still conditions on ``y`` from the covariates, so
      ``Predictive(model, num_samples=...)(rng_key, covariates)`` returns draws
      conditioned on the observed window at prior parameters. Simulate a prior
      predictive with ``dsx.sample`` and ``predict_times`` under a
      ``Simulator``, without ``obs_values``.

    Parameters
    ----------
    h
        The current call's horizon.
    name
        Site prefix handed to ``dsx.sample`` (dynestyx registers its own sites as
        ``f"{name}_..."``).
    y
        The observed window, shape ``(h.t_obs, obs)``, read from the covariates
        by the caller (``covariates[..., : h.t_obs, :1]`` for a series stored in
        column 0) so that it is present when the drivers call the model with
        ``data=None``. NaN entries are missing observations and must be in ``y``
        (the likelihood never reads ``h.data``). The ``LatentPathBuilder`` and
        the Kalman and ensemble Kalman filters and smoothers of the cuthbert
        backend (``KFConfig``, ``EnKFConfig``, ``KFSmootherConfig``,
        ``EnRTSSmootherConfig`` with ``filter_source="cuthbert"``, and the
        default ``Filter()``, an ensemble Kalman filter) skip them. The other
        configurations reject them, including the cd_dynamax backend (the default
        of ``KFConfig``) and the default ``Smoother()``. The particle filter
        returns a NaN log likelihood (its warning disappears inside the jitted
        drivers), and dynestyx does not check the HMM filters. Under a
        ``LatentPathBuilder``, a partly missing row of an observation family the
        builder cannot marginalize adds a latent ``f"{name}_missing_obs_values"``
        site.
    dynamics
        A ``dynestyx.DynamicalModel`` with a vector state (one row of the path per
        observed step). Its ``observation_model(x, u, t)`` must return a NumPyro
        distribution for one time step (dynestyx's ``LinearGaussianObservation``
        or any callable), which the in-sample predictive samples.
    controls
        Optional control inputs over the whole horizon, shape
        ``(h.duration, control)``; passed as ``ctrl_values`` with ``ctrl_times``
        equal to the full grid, and sliced to the observed window for the
        in-sample predictive.

    Returns
    -------
    StateSpaceResult
        ``x_future`` while forecasting and ``x_in_sample`` when called without
        data, ``None`` otherwise.

    Raises
    ------
    ValueError
        If no ``Filter``, ``Smoother`` or ``LatentPathBuilder`` is active; if
        ``y`` or ``controls`` do not span the horizon; if the forecast runs with
        ``n_simulations != 1``; if the in-sample predictive runs under a
        ``Filter``, receives no per-step marginals, or gets a state path that is
        not ``(t_obs, state)``; if a dynestyx site is missing (for example under
        ``numpyro.handlers.scope``). A fresh ``LatentPathBuilder`` inside a jitted
        driver raises dynestyx's own ``ValueError`` about the missingness pattern.
    RuntimeError
        If the in-sample predictive runs without a seed handler.
    jax.errors.JaxRuntimeError
        From dynestyx's layout check, when a traced call evaluates the likelihood
        under a ``LatentPathBuilder`` whose cached missingness layout comes from a
        later fit with another missingness pattern (see Notes).

    Notes
    -----
    - The ``Smoother`` in-sample draws are per-step marginals
      $p(x_t \mid y_{1:T})$, independent across steps: exact for per-step bands
      and metrics, not a joint path. The ``LatentPathBuilder`` draws are joint.
      The smoother's marginals arrive as a Python list, so building the draws
      costs trace time that grows with the length of the window.
    - A ``Filter`` and a ``Smoother`` that run the same forward filter add the
      same marginal likelihood, so the draws of a ``Filter`` fit can be exported
      under that ``Smoother`` without a refit: the Kalman filter with the Kalman
      (Rauch-Tung-Striebel) smoother, and an approximate filter with the smoother
      of the same configuration and seed. With an approximate filter, the
      posterior is only as good as that filter's likelihood estimate.
    - Under a ``LatentPathBuilder`` the drivers need the posterior draws of
      ``f"{name}_state_path_params"``, which only a fit under a builder has. A
      posterior from a ``Filter`` or ``Smoother`` fit has no path, and the
      builder then draws the path from its prior without an error: export such
      fits under a ``Smoother``.
    - The ``LatentPathBuilder`` around a driver call must be the instance that
      fitted the model: it caches the observation missingness layout per site
      name and shape from the concrete observations of the fit, and the jitted
      drivers hand it traced arrays. A later fit with the same site name and
      shape and a different missingness pattern overwrites that layout. A traced
      call for the earlier model that evaluates the likelihood (a fit,
      ``log_density``) then raises dynestyx's ``JaxRuntimeError``; the predictive
      drivers do not evaluate it and ran without error in our tests. Use one
      builder per fitted model when the patterns differ. A fresh builder needs
      its first model run eager.
    - ``mcmc.get_samples()`` also holds the deterministic sites dynestyx records
      (``f"{name}_filtered_states_mean"`` and similar). The drivers ignore
      deterministic sites, but `~~numpyro_forecast.convert.to_datatree()` stores
      them in the posterior group; drop them for a smaller tree, and keep
      ``f"{name}_state_path_params"`` under a ``LatentPathBuilder``.

    Examples
    --------
    A local level model, fitted with the Kalman filter and exported with the
    smoother around the driver call:

    ```python
    def local_level(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        y = covariates[..., : h.t_obs, :]  # the series travels in the covariates
        q = numpyro.sample("q", dist.HalfNormal(1.0))
        r = numpyro.sample("r", dist.HalfNormal(1.0))
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
            state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=q**2 * jnp.eye(1)),
            observation_model=LinearGaussianObservation(H=jnp.eye(1), R=r**2 * jnp.eye(1)),
        )
        state_space(h, "f", y, dynamics)


    rng_key, key_fit, key_export = random.split(rng_key, 3)
    with Filter(filter_config=KFConfig(filter_source="cuthbert")):
        mcmc.run(key_fit, series[:t_obs], series[:t_obs])

    # the parameters only: the fit also recorded its filtered states (f_filtered_states_*)
    posterior = {k: v for k, v in mcmc.get_samples().items() if not k.startswith("f_")}
    smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert"))
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        tree = to_datatree(key_export, local_level, posterior, series[:t_obs], series)
    ```
    """
    dsx = require("dynestyx", extra="dynestyx")
    _check_version(dsx)
    kind = _conditioning_kind()
    _validate_window(h, y, controls)
    # obs_times is a jax array, because the LatentPathBuilder indexes it inside a scan, and a
    # concrete one even under the jitted drivers, because a Filter's rollout reads it on the
    # host. predict_times and ctrl_times stay NumPy: the Simulator segments them on the host.
    with jax.ensure_compile_time_eval():
        obs_times = jnp.arange(h.t_obs, dtype=jnp.float32)
    predict_times = (  # anchor-inclusive: the rollout's initial state sits at its first time
        np.arange(h.t_obs - 1, h.duration, dtype=np.float32) if h.future > 0 else None
    )
    ctrl = (
        {}
        if controls is None
        else {"ctrl_times": np.arange(h.duration, dtype=np.float32), "ctrl_values": controls}
    )
    with warnings.catch_warnings(), numpyro.handlers.trace() as trace:
        if h.future == 0:  # an idle Simulator around an in-sample call is the recipe
            warnings.filterwarnings("ignore", message=_IDLE_SIMULATOR_WARNING)
        result = dsx.sample(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=y,
            predict_times=predict_times,
            **ctrl,
        )
    _check_site_names(trace, name, kind)
    x_in_sample, x_future = None, None
    if h.future > 0:
        predicted = jnp.asarray(trace[f"{name}{_PREDICTED_OBSERVATIONS}"]["value"])
        if predicted.shape[0] != 1:
            msg = (
                "state_space needs Simulator(n_simulations=1), "
                f"got n_simulations={predicted.shape[0]}"
            )
            raise ValueError(msg)
        numpyro.deterministic("forecast", predicted[0, 1:])  # drop the anchor row
        x_future = jnp.asarray(trace[f"{name}{_PREDICTED_STATES}"]["value"])[0, 1:]
    if h.data is None:
        key_states, key_obs = random.split(_rng_key())
        x_in_sample = _in_sample_states(key_states, result, kind, h.t_obs)
        u = None if controls is None else controls[: h.t_obs]
        numpyro.deterministic("obs", _observe(key_obs, dynamics, x_in_sample, u, obs_times))
    return StateSpaceResult(x_in_sample=x_in_sample, x_future=x_future)
