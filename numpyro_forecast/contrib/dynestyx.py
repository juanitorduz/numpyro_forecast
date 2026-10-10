r"""dynestyx state space models as a `numpyro_forecast` building block.

[dynestyx](https://github.com/BasisResearch/dynestyx) describes a dynamical
system as a ``DynamicalModel`` (initial condition, state evolution, observation
model) and interprets its ``dsx.sample`` primitive with effect handlers: a
``Filter`` or ``Smoother`` integrates the latent path out and adds the marginal
log likelihood to the NumPyro trace, a ``LatentPathBuilder`` samples the path
explicitly, and a ``Simulator`` rolls the conditioned state forward. The
handler goes **around** the call that runs the model (``mcmc.run``,
`~~numpyro_forecast.predictive.forecast()`,
`~~numpyro_forecast.predictive.predict_in_sample()`,
`~~numpyro_forecast.convert.to_datatree()`,
`~~numpyro_forecast.evaluate.backtest()`), which composes because the drivers
compile per call.

`state_space()` is the one piece of glue: a plain model function, like
`~~numpyro_forecast.models.ssoe()`, that turns the call's
`~~numpyro_forecast.models.Horizon` into the dynestyx time grids, calls
``dsx.sample`` once, and registers the two sites the drivers read by name:

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
  in-sample predictive; since a ``Filter`` and a ``Smoother`` add the same
  marginal likelihood, the draws of a ``Filter`` fit can be read under a
  ``Smoother`` without refitting.

Two dynestyx conventions shape the block. The observed window travels in the
covariates (the drivers call the model with ``data=None`` for the in-sample
predictive, as with `~~numpyro_forecast.models.ssoe()`), so the caller passes
``y`` and the likelihood reads it, NaN entries included, never ``h.data``. And
the discrete-time simulator places the rollout's initial state at the first
predict time, so the block asks for one extra step at the last observation and
drops it: the first forecast step is a transition of the final state, not a copy
of it.

The coupling to dynestyx is confined to this module: ``dynestyx.sample``, the
result types ``ConditionedResult`` (``dists``) and ``LatentStateResult``
(``state_path``), the site suffixes ``_predicted_observations`` and
``_smoothed_states_mean``, and ``DynamicalModel.observation_model(x, u, t)``.
dynestyx is an optional dependency (``pip install numpyro_forecast[dynestyx]``,
``dynestyx>=0.7.0``) imported lazily through `~~numpyro_forecast.optional.require()`
at the first call. The block supports no batch dims (``dsx.plate`` registers no
state sites), so panels are one call per series under `handlers.scope`.
"""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
from jax import random
from jaxtyping import Float

from numpyro_forecast.models import Horizon
from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array

_PREDICTED_OBSERVATIONS = "_predicted_observations"
"""Suffix of the site a dynestyx Simulator registers with the horizon rollout."""

_SMOOTHED_MEAN = "_smoothed_states_mean"
"""Suffix of the site a dynestyx Smoother registers; it tells a Smoother from a Filter."""

_IDLE_SIMULATOR_WARNING = "Simulator has no predict_times to simulate at."
"""dynestyx's warning on the in-sample legs of the one-stack recipe; silenced when future == 0."""


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
    key = numpyro.prng_key()
    if key is None:
        msg = (
            "the in-sample predictive of state_space draws the states and observations, so the "
            "model must run under a seed handler (Predictive, predict_in_sample, to_datatree)"
        )
        raise RuntimeError(msg)
    return key


def _in_sample_states(result: Any, trace: dict[str, Any], name: str, dsx: Any) -> Array:
    """One in-sample state draw ``(t_obs, state)``: the posterior path, or the smoothing marginals.

    Every step's marginal is a numpyro ``Distribution`` whose pytree aux data
    carries its (unbatched) shape, so the marginals cannot be stacked into one
    batched distribution; instead the leaves are stacked and each step is
    unflattened and sampled under ``jax.vmap``.
    """
    if isinstance(result, dsx.types.LatentStateResult):
        return jnp.asarray(result.state_path)
    if f"{name}{_SMOOTHED_MEAN}" not in trace:
        msg = (
            "the in-sample predictive needs the smoothing distribution: run predict_in_sample / "
            "to_datatree under a dynestyx Smoother (a Filter only has the filtering marginals; "
            "the draws of a Filter fit are valid under the Smoother of the same model, which "
            "shares the marginal likelihood) or a LatentPathBuilder, and keep the smoothed mean "
            "recorded (record_smoothed_states_mean=True, or record_max_elems above its size)"
        )
        raise ValueError(msg)
    dists = result.dists
    if not dists:
        msg = "the dynestyx Smoother returned no per-step marginals to draw from"
        raise ValueError(msg)
    treedef = jax.tree.structure(dists[0])
    leaves = [jnp.stack(step) for step in zip(*(jax.tree.leaves(d) for d in dists), strict=True)]
    keys = random.split(_rng_key(), len(dists))

    def sample_step(step_leaves: list[Array], key: Array) -> Array:
        return jnp.asarray(jax.tree.unflatten(treedef, step_leaves).sample(key))

    return jax.vmap(sample_step)(leaves, keys)


def _observe(dynamics: Any, x: Array, u: Array | None, times: Array) -> Array:
    """Draw ``y_t ~ observation_model(x_t, u_t, t)`` for every in-sample step, ``(t_obs, obs)``."""
    keys = random.split(_rng_key(), x.shape[0])

    def sample_step(x_t: Array, u_t: Array | None, t: Array, key: Array) -> Array:
        return jnp.asarray(dynamics.observation_model(x_t, u_t, t).sample(key))

    return jax.vmap(sample_step, in_axes=(0, None if u is None else 0, 0, 0))(x, u, times, keys)


def state_space(
    h: Horizon,
    name: str,
    y: Float[Array, " time obs"],
    dynamics: Any,
    *,
    controls: Float[Array, " duration control"] | None = None,
) -> None:
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
      ``DiscreteTimeSimulator(n_simulations=1)`` placed outside the conditioning
      handler. The block asks the simulator for ``predict_times`` starting at
      the last observation and drops that first row, because dynestyx places the
      rollout's initial state at the first predict time without a transition.
    - ``h.data is None`` (the in-sample predictive of
      `~~numpyro_forecast.predictive.predict_in_sample()` and
      `~~numpyro_forecast.convert.to_datatree()`): the site ``"obs"``, shape
      ``(t_obs, obs)``, one observation draw per step from
      ``dynamics.observation_model(x_t, u_t, t)`` at a draw of the in-sample
      states: step-wise from the smoothing marginals under a ``Smoother``, the
      posterior path itself under a ``LatentPathBuilder``.

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
        (the likelihood never reads ``h.data``); the cuthbert Kalman and ensemble
        filters and smoothers and the ``LatentPathBuilder`` accept them, the
        cd_dynamax backend rejects them and the particle filter only warns.
    dynamics
        A ``dynestyx.DynamicalModel``. Its ``observation_model(x, u, t)`` must
        return a NumPyro distribution for one time step (dynestyx's
        ``LinearGaussianObservation`` or any callable), which the in-sample
        predictive samples.
    controls
        Optional control inputs over the whole horizon, shape
        ``(h.duration, control)``; passed as ``ctrl_values`` with ``ctrl_times``
        equal to the full grid, and sliced to the observed window for the
        in-sample predictive.

    Raises
    ------
    ValueError
        If ``y`` or ``controls`` do not span the horizon; if the forecast rollout
        has ``n_simulations != 1``; if the in-sample predictive runs under a
        ``Filter`` (or a ``Smoother`` that does not record its smoothed mean) or
        receives no per-step marginals. A missing conditioning handler or a
        missing Simulator is dynestyx's own ``ValueError``.
    RuntimeError
        If the in-sample predictive runs without a seed handler.

    Notes
    -----
    The ``Smoother`` in-sample draws are per-step marginals $p(x_t \mid y_{1:T})$,
    independent across steps: exact for per-step bands and metrics, not a joint
    path. The ``LatentPathBuilder`` draws are joint; the builder around a driver
    call must be the instance that fitted the model, because it caches the
    observation missingness layout from the concrete observations of the fit
    and the jitted drivers hand it traced arrays. The block has no batch dims:
    a panel is one call per series under ``handlers.scope``.

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


    with Filter(filter_config=KFConfig(filter_source="cuthbert")):
        mcmc.run(key, series[:t_obs], series[:t_obs])

    with DiscreteTimeSimulator(n_simulations=1), Smoother(smoother_config=KFSmootherConfig()):
        tree = to_datatree(key, local_level, mcmc.get_samples(), series[:t_obs], series)
    ```
    """
    dsx = require("dynestyx", extra="dynestyx")
    _validate_window(h, y, controls)
    times = jnp.arange(h.duration, dtype=jnp.float32)
    obs_times = times[: h.t_obs]
    predict_times = times[h.t_obs - 1 :] if h.future > 0 else None  # anchor-inclusive, see above
    ctrl = {} if controls is None else {"ctrl_times": times, "ctrl_values": controls}
    with warnings.catch_warnings(), numpyro.handlers.trace() as trace:
        if h.future == 0:  # an idle Simulator around an in-sample call is the one-stack recipe
            warnings.filterwarnings("ignore", message=_IDLE_SIMULATOR_WARNING)
        result = dsx.sample(
            name, dynamics, obs_times=obs_times, obs_values=y, predict_times=predict_times, **ctrl
        )
    if h.future > 0:
        predicted = jnp.asarray(trace[f"{name}{_PREDICTED_OBSERVATIONS}"]["value"])
        if predicted.shape[0] != 1:
            msg = (
                "state_space needs Simulator(n_simulations=1), "
                f"got n_simulations={predicted.shape[0]}"
            )
            raise ValueError(msg)
        numpyro.deterministic("forecast", predicted[0, 1:])  # drop the anchor row
    if h.data is None:
        x = _in_sample_states(result, trace, name, dsx)
        u = None if controls is None else controls[: h.t_obs]
        numpyro.deterministic("obs", _observe(dynamics, x, u, obs_times))
