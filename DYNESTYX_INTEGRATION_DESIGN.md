# Design: integrating `numpyro_forecast` with `dynestyx`

Status: design with a working example. Every claim about `dynestyx` below was verified with the throwaway probes summarized in Appendix A, against `dynestyx==0.5.0` (A.1 to A.6) and against the `ml-bugfix-controldim-LTI` branch (v0.5.0 plus [dynestyx#361](https://github.com/BasisResearch/dynestyx/pull/361); A.7 onward), installed next to the current `numpyro_forecast` pins (`jax==0.11.0`, `numpyro==0.21.0`, `arviz==1.3.0`). That branch shipped as [dynestyx v0.5.1](https://github.com/BasisResearch/dynestyx/releases/tag/v0.5.1) on 2026-09-15, so no git reference is needed anymore; the probes were not re-run on the release, which also bumps `cd-dynamax` to 0.4.3. The example notebook that implements this design is `docs/examples/dynestyx_integration.ipynb` (Section 14). The second revision of this document incorporates the review by the `dynestyx` maintainers on [numpyro_forecast#125](https://github.com/juanitorduz/numpyro_forecast/pull/125): smoothing for the in-sample predictive, `LatentPathBuilder` as a third inference strategy, explicit time grids instead of a step size, and the `control_dim` fix. The third revision answers their follow-up comments on that PR: how the `Horizon` encodes the forecast times (Section 6.1), why the in-sample predictive is a smoothing task and not a prior predictive (Sections 1, 4 and 6.1), the cached-anchor rollout as the fallback should re-conditioning every draw become too expensive (Section 3, Approach C, Sections 11 and 17), and the v0.5.1 release.

Tracking: [juanitorduz/numpyro_forecast#34](https://github.com/juanitorduz/numpyro_forecast/issues/34) and [BasisResearch/dynestyx#264](https://github.com/BasisResearch/dynestyx/issues/264).

## 1. Summary

A `dynestyx` state space model is written **as a `numpyro_forecast` model function** `(covariates, data=None) -> None`: the model derives its `Horizon` from the shapes, samples its parameters with `numpyro.sample`, builds a `dynestyx.DynamicalModel`, and hands both to one new model building block, `state_space_series(h, name, y, dynamics, conditioner=...)`. The `conditioner` is the `dynestyx` handler stack that interprets `dsx.sample` over the observed window, which is how `dynestyx` separates the model from its inference: a `Filter` or a `Smoother` marginalizes the latent path and adds the marginal log likelihood $\log p(y_{1:T} \mid \theta)$ as a NumPyro factor, a `LatentPathBuilder` samples the path explicitly, and a `Discretizer` after either turns a continuous-time model into the discrete transition they consume. The block adds nf's horizon bookkeeping on top of whichever conditioner is passed: while forecasting it nests the conditioner inside a `Simulator`, whose posterior rollout starts from the conditioned state at the last observed step, and registers the horizon draws as `"forecast"`; when a driver calls the model without `data` (`predict_in_sample` and `to_datatree` pass `data=None` as a mode switch, while the observed window still reaches the model through `covariates`) it draws the in-window states from the conditioner's posterior over the path given the whole window, $p(x_t \mid y_{1:T}, \theta)$ (the smoothing distribution, or the explicit path), and one observation per step, and registers them as `"obs"`, which is what `predict_in_sample` and `to_datatree` read: a smoothing task, not a prior predictive. A model is its horizon, its parameters and one block call. Nothing else changes: `SVI`, `MCMC`, `forecast`, `predict_in_sample`, `to_datatree`, `backtest` (including `eval_train`), the metrics and the ArviZ export all work on the model unchanged.

The block is the entire integration. It has no class hierarchy, no adapter objects and no second driver stack; it is the same shape as the existing building blocks (a plain function that calls NumPyro primitives against a `Horizon`, with the observed series passed in as `ssoe` does), and it composes `dynestyx`'s documented handlers rather than its internals. Switching between marginalized and explicit-path inference is a one-argument change. It lives in the example notebook first and is proposed for `numpyro_forecast/contrib/dynestyx.py` once the maintainers of both libraries have reviewed it (Section 13).

What the user gains: with a `Filter` or `Smoother` the latent path is marginalized (exactly, with the Kalman filter, for linear-Gaussian models; approximately, with the EnKF/EKF/UKF/particle filter, beyond that), so NUTS and SVI see only the parameters; with a `LatentPathBuilder` the path is explicit, as with nf's `innovations`, but built by `dynestyx`, which is the road to discretized continuous-time dynamics and to NaN-aware observation models. On the local level model with $T = 120$ the marginalized NUTS run drew 479 effective samples of the state noise scale on 4 822 leapfrog steps against 140 on 21 992 steps with the explicit path and 76 on 236 568 steps for nf's direct model (Section 11).

## 2. Context

`numpyro_forecast` (nf below) is a functional port of the ideas in `pyro.contrib.forecast`: a model is a plain NumPyro function built from model building blocks (`Horizon.from_data`, `innovations`, `markov_series`, `ssoe`, `predict`) that register the `"obs"` site over the training window and the `"forecast"` site over the horizon; inference is whatever NumPyro or BlackJAX code the user writes; the drivers (`forecast`, `predict_in_sample`, `to_datatree`, `backtest`, `backtest_vectorized`) and the probabilistic metrics (`eval_crps`, `eval_coverage`, ...) read those sites by name. Its latent time series are sampled directly: a random walk is the cumulative sum of `T` sampled innovations, so a fit infers `T` per-step latents plus the parameters.

`dynestyx` (dsx below) is a probabilistic programming layer for dynamical systems on top of NumPyro. A `DynamicalModel` bundles an initial condition, a state evolution (discrete-time transition, or a continuous-time SDE/ODE) and an observation model; `dsx.sample(name, dynamics, obs_times=, obs_values=, predict_times=)` is the single primitive, and effect handlers give it its meaning: `Filter` and `Smoother` marginalize the latent path and register the marginal log likelihood as a NumPyro factor (the smoother also computes $p(x_t \mid y_{1:T})$), `LatentPathBuilder` samples the path explicitly, and the simulators (`Simulator`, `DiscreteTimeSimulator`, `ODESimulator`, `SDESimulator`) generate trajectories, including posterior rollouts when they wrap one of the conditioning handlers. An `Evaluation` handler scores the one-step-ahead predictive observations of a filter with proper scoring rules.

In dynestyx#264 the dsx maintainers suggested two directions: (1) use nf to evaluate a dsx model, and (2) use dsx to fit nf-style models, the benefit being that "your model would be running a non-linear filtering algorithm to connect the noisy data to the underlying model". This document turns both into one mechanism. Direction (2) is the block itself. Direction (1) follows for free: once a dsx model is an nf model, `backtest` and the metrics score its multi-step forecasts on rolling windows, complementing dsx's own in-sample one-step-ahead `Evaluation`.

An earlier draft of this design (branch `dynestyx`) targeted nf's pre-refactor class API (`ForecastingModel`, `Forecaster`, `HMCForecaster`) and was blocked by an ArviZ version conflict. Both are gone: nf is now functional (PR #81) and dynestyx v0.5.0 migrated to ArviZ $\geq 1.0$ (dynestyx#346). That draft's adapter objects (`DsxFilterFit`, `DynestyxForecaster`) are superseded by the block; nothing from it is kept. The first revision of this document had a `filtered_series` block bound to the `Filter` handler with a scalar `dt`; the review on numpyro_forecast#125 led to the conditioner argument, the explicit `times` grid and the in-sample predictive described here.

## 3. Where the two libraries meet

Both libraries are NumPyro effect-handler code, so they compose at the trace level. The seam is small and can be stated exactly:

- nf owns the **horizon bookkeeping** (`Horizon`: `t_obs`, `future`, `duration`), the **site contract** (`"obs"` and `"forecast"`, plus the `_future` suffix convention for sampled latents), the **drivers** (jitted, chunked, device-aware `Predictive` wrappers) and the **evaluation workflow** (rolling windows, CRPS, coverage, MASE, ArviZ export).
- dsx owns the **dynamics** (`DynamicalModel`), the **conditioning** (`Filter`, `Smoother`, `LatentPathBuilder`: the marginal likelihood or the joint path density, plus the conditioned distributions over the path) and the **rollout** (a `Simulator` outside the conditioner starts every prediction segment from the conditioned distribution at the last observation time that precedes it).
- The block translates between them: the observed window `y` becomes `obs_values`, the step index or the user's `times` become `obs_times`/`predict_times`, `covariates` become `ctrl_values`, the simulator's `{name}_predicted_observations` site becomes the `"forecast"` site, and the conditioner's posterior over the path becomes the `"obs"` site when the model is called without data.

Three properties make the composition sound rather than merely possible:

1. `Filter` and `Smoother` register the marginal likelihood as `numpyro.factor`, and `LatentPathBuilder` registers one fixed-size sample site for the path plus the joint density as a factor, so in every case the model's density is a plain NumPyro joint that `SVI` with any autoguide, `MCMC` with any kernel, and the BlackJAX kernels in `numpyro_forecast.contrib.blackjax` consume through `initialize_model`, unchanged.
2. Fitting always happens with `future == 0` (nf's invariant), so the simulator and its `_predicted_*` sites never appear in the guide or in the MCMC trace: the forecast branch of the block is only ever executed under `Predictive`. The path site of the `LatentPathBuilder` has the shape of the observed window whatever the horizon, so posterior substitution works for it exactly as the `_future` suffix makes it work for `innovations` and `markov_series`.
3. Under `Predictive(posterior_samples=...)` the model re-conditions on the window for every posterior draw and then rolls the conditioned state forward, so the forecast is conditioned on the data through the state, with the draw's own parameters, which is the correct posterior predictive $\int p(y_{T+1:T+H} \mid x_T, \theta) \, p(x_T \mid y_{1:T}, \theta) \, p(\theta \mid y_{1:T}) \, dx_T \, d\theta$. The in-sample predictive is the same construction over the window: $\int p(y_t \mid x_t, \theta) \, p(x_t \mid y_{1:T}, \theta) \, p(\theta \mid y_{1:T}) \, dx_t \, d\theta$, per step, which is what nf's direct models produce marginally when `predict_in_sample` replays their sampled path. Re-conditioning costs one pass of the conditioner over the window per draw (`t_obs` filter steps, vectorized by `vmap`: 0.7 s for 300 draws at $T = 120$, Section 11). It is also what lets one posterior be forecast on a different data stream or under counterfactual controls, which is why dsx re-conditions at prediction time instead of replaying the fit, and which nf's drivers expose by taking `data` and `covariates` at prediction time. Should that pass become prohibitively expensive (long windows, particle filters), the fallback the dsx maintainers suggest is to cache the final conditioned state of every draw during the fit and roll out from it with `dsx.simulate`, without re-conditioning (Approach C); the price is a forecast frozen to the window it was fitted on.

## 4. Design principles

- **One model, three interpretations.** A model function is written once; the `Horizon` (and only the `Horizon`) decides whether the block conditions, forecasts, or produces the in-sample predictive. This is nf's existing rule and it matches dsx's own "separation of concerns" (a `DynamicalModel` has no notion of `predict`; handlers interpret `dsx.sample`).
- **The handler stack is the strategy.** Which dsx handlers condition the window is an argument of the block, not a family of blocks or a configuration enum. The three handlers dsx documents for conditioning are the three values the argument takes, a `Discretizer` can follow any of them, and each stack composes with `Simulator` for the rollout exactly as dsx documents it.
- **The series travels through `covariates`.** The block takes the observed window `y` as an argument and uses `h.data` only to detect the mode, as `ssoe` does. This is what lets `predict_in_sample`, `to_datatree` and `backtest(eval_train=True)`, which call the model with `data=None`, reach the observations that a conditioned model needs. `data=None` is therefore a mode switch, not an absence of data: the in-sample predictive conditions on the whole window, which is why a `Smoother` (or the explicit path) is the right conditioner for it and a prior predictive is not what it computes.
- **Pure functions, explicit randomness.** The block is a plain function with no state of its own; randomness comes from the enclosing NumPyro `seed` handler exactly as for `numpyro.sample`, so `Predictive`, `MCMC` and `SVI` control every key. The stateful objects are the handlers the user creates, once, outside the model (Section 6.1).
- **Static shapes, host-side time grids.** `t_obs`, `future` and `duration` are Python integers derived from array shapes, and `times` is a host-side NumPy array, so the grids the block slices from it are constants inside `jax.jit`. This is what keeps the `Filter`/`Smoother` rollout, which does its segment bookkeeping on the host with `np.searchsorted`, safe inside nf's jitted `_predict` driver (verified, Appendix A.3 and A.6). The `LatentPathBuilder` is the opposite case: it indexes its grids inside a `lax.scan` and needs jax arrays (A.8). The block converts per conditioner; a grid derived from a traced argument (a covariate column) would break the first case.
- **Compile once, vectorize over draws.** nf's `forecast()` jits one `Predictive` per `(model, shape)` and vectorizes the sample axis with `vmap`. The Kalman filter and smoother (`lax.scan`, or cuthbert's associative scan), the path reconstruction of the builder and the discrete simulator are all `vmap`-friendly, so the whole forecast for 300 draws compiles and runs in 0.7 s on CPU (A.3). `batch_size` chunking, `parallel=False` and `device="host"` keep working because the driver does not know that dsx is inside.
- **Reuse documented dsx composition, not internals.** The block uses `Simulator` outside a conditioner with `predict_times`, which is the posterior-rollout composition dsx documents for all three handlers, reads the simulator's sites through `numpyro.handlers.trace` (a plain NumPyro idiom; `ssoe` already uses an inner trace), and reads the per-time smoothing distributions and the reconstructed path from the result objects `dsx.sample` returns (`ConditionedResult.dists`, `LatentStateResult.state_path`, both public fields). The pure-JAX alternative that goes further into internals is discussed as Approach C.
- **No new inference code.** Fitting is `svi.run` or `mcmc.run` on the model. The block never wraps either.

## 5. Approaches considered

### A. A model building block inside nf's model contract (recommended)

The dsx model is an nf `ForecastModel`; `state_space_series` is the only new function. Every nf driver, closure and metric works with zero adapter code, `backtest`'s `model_fn`/`forecast_fn`/`in_sample_fn` contract is untouched, and a user who already has an nf model swaps one block for another. The only nf driver contract that needed thought is the `data=None` call of `predict_in_sample` and `to_datatree`, which the `ssoe` idiom (series in `covariates`) already solves.

### B. Adapter functions around a native dsx model

Keep the dsx model in its native signature `(obs_times, obs_values, predict_times)` and write `fit_dsx_filter`, `dsx_forecast` and a `DynestyxForecaster` object that satisfy `backtest`'s closures (the earlier draft). This reimplements what `forecast()` already does (chunking, device placement, compile caching) with a second `Predictive` call site, introduces a second model contract next to `ForecastModel`, and cannot be swapped into an existing nf model. It is only preferable when the same dsx model must also run in pure dsx workflows, and even then the `DynamicalModel` construction can be shared as a helper function while the two thin model functions stay separate. Rejected.

### C. Pure-JAX rollout without handler nesting

Condition with `dsx.condition` under a `Filter` to obtain a `ConditionedResult` (marginal likelihood, filtered `dists` and `times`), register the factor by hand, then build the anchored dynamics with `eqx.tree_at` (initial condition set to `dists[-1]`, `t0` to `times[-1]`) and call `DiscreteTimeSimulator(n_simulations=1).simulate(anchored, rng_key=numpyro.prng_key(), predict_times=...)`. This is the most explicit functional form: the key is passed by hand, no trace is read, no host-side segment search runs, and only the final segment is simulated. The variant the dsx maintainers suggested on #125 goes one step further and skips the conditioning at prediction time altogether: the fit records the final conditioned state of every posterior draw (the last row of `{name}_filtered_states_mean`/`_cov` or `{name}_smoothed_states_mean`/`_cov`, which the `record_*` config flags already put in the posterior dict, Section 7), `Predictive` substitutes those deterministic sites back into the model, and the forecast branch builds the anchored dynamics from the cached `(mean, cov)` and the draw's parameters and calls `dsx.simulate`, vmapped over the pairs. That removes the `t_obs`-step pass per draw and the host-side grid requirement (`dsx.simulate` takes jax grids, A.6), at the price of freezing the forecast to the fitted window: the posterior can no longer be forecast on a different data stream or under counterfactual controls over the window, which is the reason dsx re-conditions by default. Both forms depend on `DynamicalModel` being an Equinox module with mutable-by-`tree_at` fields and on `numpyro.prng_key()`; neither is documented as the way to roll out a posterior, and neither covers the `LatentPathBuilder`. Not exercised in the probes. Either is the natural internal implementation of Approach A's forecast branch if the dsx maintainers endorse it, with the block's public contract unchanged; the cached variant is the answer when re-conditioning turns out to dominate the forecast (Section 11), or becomes redundant if dsx ships a replay mode (Section 17).

### D. One block per handler

`filtered_series`, `smoothed_series`, `latent_series`. Three functions with identical horizon bookkeeping and different one-line handler compositions; the user who wants to compare strategies rewrites the model three times. The dsx maintainers' point in the review, that the explicit-path and marginalized configurations should differ by as little code as possible, argues for the single block with a `conditioner` argument. Rejected in favor of A.

Decision: A, with the rollout implemented through the documented `Simulator` ∘ conditioner composition, and C recorded as the candidate optimization.

## 6. The building block: `state_space_series`

### 6.1 Contract

```python
state_space_series(
    h: Horizon,
    name: str,
    y: Float[Array, " time obs"],
    dynamics: DynamicalModel,
    *,
    conditioner: StateSpaceHandler | Sequence[StateSpaceHandler],
    controls: Float[Array, " duration control"] | None = None,
    times: np.ndarray | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> StateSpaceResult
```

- `h` is the current call's `Horizon`: three Python integers derived from shapes, `duration = covariates.shape[-2]`, `t_obs = data.shape[-2]` (or `duration` when `data` is `None`) and `future = duration - t_obs`. It encodes how many steps are observed and how many are forecast, not when they happen: the rows of `covariates` beyond `t_obs` are the forecast steps, one row per forecast time, and their times are the tail of the `times` grid, `times[t_obs:duration]`, which may be as irregular as the observed part (A.11 forecasts 24 steps whose gaps range from 0.3 to 2.5, and the forecast variance grows with each gap). With `times=None` the forecast times are the step indices `t_obs, ..., duration - 1`. A forecast at arbitrary times is therefore `covariates` with `future` extra rows plus a `times` grid ending in those times. `h.data` selects the mode: training when present and `h.future == 0`, forecasting when `h.future > 0`, in-sample predictive when `None` (a driver convention: `predict_in_sample` and `to_datatree` pass `data=None` as the mode switch while the observed window still arrives through `covariates` and `y`, so this mode is a smoothing task, not a prior predictive). The observations themselves come from `y`.
- `y` is the observed window, shape `(t_obs, obs)` with time at axis `-2`, sliced from `covariates` by the caller (`y = covariates[..., :h.t_obs, :obs]`), exactly the contract of `ssoe`; the block checks that it covers `h.t_obs` steps. It is passed to dsx unchanged as `obs_values`.
- `name` is the dsx site prefix. A `Filter` registers `{name}_marginal_log_likelihood` (the factor), `{name}_marginal_loglik` and the `{name}_filtered_states_*` deterministics selected by its config; a `Smoother` registers the same factor plus the `{name}_smoothed_states_*` deterministics selected by its config; a `LatentPathBuilder` registers the sample site `{name}_state_path_params`, the factor `{name}_joint_log_prob_factor` and the deterministics `{name}_state_path`, `{name}_state_path_times`, `{name}_joint_log_prob`. The simulator registers `{name}_predicted_times`, `{name}_predicted_states`, `{name}_predicted_observations` and one `{name}_{j}_x_0` per rollout segment. The block adds nf's two sites, `"forecast"` and `"obs"`, and one of its own, `{name}_smoothed_states`, the draw of the in-window states from the smoothing distribution, only in the in-sample predictive mode of a `Smoother`.
- `dynamics` is any `DynamicalModel` the conditioner supports. The observation dimension must equal `y.shape[-1]`; a scalar-state model must still be written with a length-1 vector state (`dist.Normal(...).expand([1]).to_event(1)`, transitions returning `.to_event(1)` distributions), because `y` carries an observation axis.
- `conditioner` is the dsx handler stack that conditions the window, entered outermost first: exactly one of `Filter(filter_config=...)` (marginalized, no in-sample predictive), `Smoother(smoother_config=...)` (marginalized, same marginal likelihood, in-sample predictive from the smoothing distribution) or `LatentPathBuilder(...)` (explicit path), alone or as the first element of a sequence followed by a `Discretizer(...)`, which turns a continuous-time model into the discrete transition the conditioning handler consumes (Section 8). `StateSpaceHandler` is the union of the four handler types. Create the handlers once outside the model and close over them: they are dsx objects with state of their own, and the builder in particular caches the observation layout it needs to run under `jit` after it has seen concrete observations once (Section 10, item 12). For linear-Gaussian models pass `KFConfig()`/`KFSmootherConfig(filter_source="cd_dynamax")` explicitly; the dsx defaults (`EnKFConfig()`, `EKFSmootherConfig`) are approximate. The `Smoother` costs nothing extra during fitting (A.10), so it is the recommended conditioner when the in-sample predictive or `to_datatree` is wanted. A malformed stack (no conditioning handler, two of them, or a `Discretizer` first) raises.
- `controls` are the exogenous inputs over the full horizon, shape `(duration, control)`, normally the covariate columns other than the series. They become `ctrl_values` on the full time grid, which covers both the observation times and the prediction times, as the simulator requires. The dynamics consume them through `B`/`D` (linear-Gaussian) or the `u` argument of a callable transition/observation.
- `times` are the observation and forecast times over the full horizon: a host-side NumPy array with at least `duration` strictly increasing entries, of which the block uses the first `duration` (`obs_times = times[:t_obs]`, `predict_times = times[t_obs - 1:duration]`, `ctrl_times = times[:duration]`); the last `future` entries are the forecast times. `None` uses the step index `0, 1, ..., duration - 1`. Irregular spacing is passed through as is, on both sides of the split (A.10, A.11); it matters for continuous-time dynamics and for time-varying parameters, and is ignored by discrete transitions. It is an argument rather than a covariate column because the `Filter`/`Smoother` rollout needs concrete times under `jit` (Section 4).
- `simulator_config` is forwarded to `Simulator`, which auto-selects the discrete, ODE or SDE backend from the dynamics; the discrete backend has no configuration.

Behavior:

- Training (`h.data` present, `h.future == 0`): run `dsx.sample` under the stack; the factor (and the path site, for the builder) enters the trace; return a result whose time axes have size 0.
- Forecasting (`h.future > 0`): run `dsx.sample` under `Simulator(simulator_config, n_simulations=1)` outside the stack with `predict_times = times[t_obs - 1:duration]`, read `{name}_predicted_observations` and `{name}_predicted_states` from an inner trace, drop the simulation axis and the anchor row, register `y_future` of shape `(future, obs)` as the `"forecast"` deterministic site, and return it with `x_future` of shape `(future, state)`.
- In-sample predictive (`h.data is None`, `h.future == 0`): run `dsx.sample` under the stack, take one draw of every in-window state (a `Smoother`: sample `{name}_smoothed_states` from the per-time Gaussian marginals in `ConditionedResult.dists`; a `LatentPathBuilder`: the reconstructed `state_path`, which under `Predictive` is the posterior path), sample one observation per step from `dynamics.observation_model(x_t, u_t, t)` with per-step keys, register the result as the `"obs"` deterministic site and return it as `y_in_sample` of shape `(t_obs, obs)`. This is the smoothing predictive: with $x_t \sim p(x_t \mid y_{1:T}, \theta)$ and $y_t^{\text{rep}} \sim p(y \mid x_t, \theta)$ the draw is from $p(y_t^{\text{rep}} \mid y_{1:T}, \theta)$, the per-step replicate that nf's direct models produce when `predict_in_sample` replays their sampled path. It is not a prior predictive, because the window reaches the model through `y` even though `data` is `None`; a prior predictive check uses `dsx.simulate` instead (below). A `Filter` raises with a message naming the two conditioners that work: the filtering distribution $p(x_t \mid y_{1:t})$ is not the quantity `predict_in_sample` means. The draws are per-time marginals of the smoothing predictive; the joint structure of the path across time is not reproduced (no backward simulation), which only matters for path-wise functionals such as the distribution of a running maximum. This mode is a shim: its target form is the same `Simulator` ∘ stack composition as forecasting with `predict_times = times[:t_obs]`, which dsx supports for `Filter` today and rejects for `Smoother` and `LatentPathBuilder` until in-window prediction lands (Section 12, A.12).
- The block owns the likelihood (the factor), so, like `predict` and unlike `ssoe`, it registers nf's two sites itself: `"forecast"` while forecasting and `"obs"` in the in-sample mode. A model therefore has one `state_space_series` call, exactly as it has one `predict` call, and cannot forget a site. The returned `StateSpaceResult` is for the states and for custom use.
- Prior predictive checks use `dsx.simulate(dynamics, rng_key=..., predict_times=..., n_simulations=...)`, the pure-JAX generator, directly (it needs jax time grids, A.6).

### 6.2 Reference implementation

This is the code the notebook defines and the code proposed for `contrib/dynestyx.py`; the two are identical, docstring included, so the block graduates without edits.

```python
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from dynestyx import Discretizer, DynamicalModel, Filter, LatentPathBuilder, Simulator, Smoother
from dynestyx.inference.configs.simulator import SimulatorConfig
from jax import random
from jaxtyping import Float

from numpyro_forecast import Horizon
from numpyro_forecast.typing import Array

StateSpaceHandler = Filter | Smoother | LatentPathBuilder | Discretizer
"""A dynestyx handler that interprets ``dsx.sample`` over the observed window."""

_CONDITIONING_HANDLERS = (Filter, Smoother, LatentPathBuilder)


@dataclass(frozen=True)
class StateSpaceResult:
    """Draws produced by `state_space_series` (size-0 time axes when not applicable).

    Attributes
    ----------
    y_future
        Observation draws over the horizon, shape ``(future, obs)``; also
        registered as the ``"forecast"`` site.
    x_future
        Latent state draws over the horizon, shape ``(future, state)``.
    y_in_sample
        One draw of the in-sample predictive, shape ``(t_obs, obs)``; filled only
        when the model is called without data, and then registered as ``"obs"``.
    """

    y_future: Float[Array, " future obs"]
    x_future: Float[Array, " future state"]
    y_in_sample: Float[Array, " time obs"]


def _handler_stack(
    conditioner: StateSpaceHandler | Sequence[StateSpaceHandler],
) -> tuple[StateSpaceHandler, ...]:
    """Normalize ``conditioner`` to the tuple of handlers entered outermost first."""
    stack = tuple(conditioner) if isinstance(conditioner, Sequence) else (conditioner,)
    conditioning = [i for i, h in enumerate(stack) if isinstance(h, _CONDITIONING_HANDLERS)]
    if len(conditioning) != 1:
        msg = (
            "conditioner needs exactly one Filter, Smoother or LatentPathBuilder "
            f"(optionally followed by a Discretizer), got {[type(h).__name__ for h in stack]}"
        )
        raise ValueError(msg)
    if conditioning[0] != 0:
        msg = (
            "the Filter, Smoother or LatentPathBuilder must come first (outermost) in conditioner"
        )
        raise ValueError(msg)
    return stack


def _time_grid(times: np.ndarray | None, h: Horizon) -> np.ndarray:
    """Full-horizon float times: ``times[:duration]``, or ``0, 1, ..., duration - 1``."""
    if times is None:
        return np.arange(h.duration, dtype=np.float32)
    grid = np.asarray(times, dtype=np.float32)
    if grid.ndim != 1 or grid.shape[0] < h.duration:
        msg = f"times must be a 1-D array with at least duration={h.duration} entries, got {grid.shape}"
        raise ValueError(msg)
    grid = grid[: h.duration]
    if np.any(np.diff(grid) <= 0):
        msg = "times must be strictly increasing"
        raise ValueError(msg)
    return grid


def _in_sample_states(
    name: str, conditioner: StateSpaceHandler, result: Any
) -> Float[Array, " time state"]:
    """One draw of every in-window state from the conditioner's posterior over the path."""
    if isinstance(conditioner, LatentPathBuilder):
        return result.state_path
    if isinstance(conditioner, Smoother):
        dists = result.dists
        if not dists or not isinstance(dists[0], dist.MultivariateNormal):
            msg = (
                "the in-sample predictive needs a Gaussian smoother (per-time MultivariateNormal)"
            )
            raise TypeError(msg)
        mean = jnp.stack([d.mean for d in dists])
        cov = jnp.stack([d.covariance_matrix for d in dists])
        state_dist = dist.MultivariateNormal(mean, covariance_matrix=cov).to_event(1)
        return jnp.asarray(numpyro.sample(f"{name}_smoothed_states", state_dist))
    msg = (
        "the in-sample predictive (a model call with data=None, as made by predict_in_sample "
        "and to_datatree) needs a Smoother or a LatentPathBuilder conditioner; a Filter only "
        "carries the filtering distribution p(x_t | y_1:t)."
    )
    raise ValueError(msg)


def state_space_series(
    h: Horizon,
    name: str,
    y: Float[Array, " time obs"],
    dynamics: DynamicalModel,
    *,
    conditioner: StateSpaceHandler | Sequence[StateSpaceHandler],
    controls: Float[Array, " duration control"] | None = None,
    times: np.ndarray | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> StateSpaceResult:
    """Condition a dynestyx model on the observed window and predict with it.

    The conditioner is the ``dynestyx`` handler stack that interprets
    ``dsx.sample`` over the observed window, entered outermost first: a
    ``Filter`` or a ``Smoother`` adds the marginal log likelihood of the window as
    a NumPyro factor (the latent path is integrated out), a ``LatentPathBuilder``
    samples the path explicitly, and an optional ``Discretizer`` after it turns a
    continuous-time model into the discrete transition the others consume. The
    block adds the horizon bookkeeping on top and registers the two sites the
    package drivers read. While forecasting it nests the stack inside a
    ``Simulator``, whose posterior rollout starts from the conditioned state at
    the last observed step, and registers the horizon draws as ``"forecast"``.
    When the model is called without data it draws the in-window states from the
    conditioner's posterior over the path (the smoothing distribution, or the
    explicit path), samples one observation per step from the observation
    model, and registers them as ``"obs"``: the in-sample predictive that
    ``predict_in_sample`` and ``to_datatree`` read. The guide never sees the
    rollout because fitting happens with ``h.future == 0``.

    Parameters
    ----------
    h
        The horizon for the current model call. ``h.data`` selects the mode
        (training when present, in-sample predictive when ``None``); the
        observations themselves come from ``y``.
    name
        Prefix of the ``dynestyx`` sites.
    y
        The observed window, shape ``(t_obs, obs)`` with time at axis ``-2``,
        sliced from ``covariates`` by the caller (the contract of ``ssoe``, so
        that model calls without data still reach the observations).
    dynamics
        The ``dynestyx`` model; its observation dimension must match ``y``.
    conditioner
        ``Filter(filter_config=...)``, ``Smoother(smoother_config=...)`` or
        ``LatentPathBuilder(...)``, alone or as the first element of a sequence
        followed by a ``Discretizer(...)``. Create the handlers once outside the
        model and reuse them: the builder caches the observation layout it
        needs under ``jit``. A ``Filter`` cannot serve the in-sample predictive.
    controls
        Exogenous inputs over the full horizon, shape ``(duration, control)``,
        forwarded as ``ctrl_values`` on the full time grid.
    times
        Observation and forecast times over the full horizon, at least
        ``duration`` strictly increasing entries, a host-side NumPy array;
        ``None`` uses the step index. The last ``future`` entries are the
        forecast times. Irregular spacing matters for continuous-time dynamics
        and for time-varying parameters.
    simulator_config
        Forwarded to ``Simulator`` (solver options for continuous-time models
        that are not discretized).

    Returns
    -------
    StateSpaceResult
        ``y_future``, ``x_future`` and ``y_in_sample``.

    Raises
    ------
    ValueError
        If ``y`` does not cover exactly ``h.t_obs`` steps, if ``times`` is not a
        strictly increasing grid covering the horizon, if ``conditioner`` does
        not hold exactly one conditioning handler first, or if the in-sample
        predictive is requested with a ``Filter`` conditioner.
    TypeError
        If the in-sample predictive is requested with a non-Gaussian smoother.
    RuntimeError
        If the in-sample predictive runs outside a NumPyro ``seed`` handler.
    """
    if y.ndim < 2 or y.shape[-2] != h.t_obs:
        msg = f"y must have shape (t_obs={h.t_obs}, obs), got {y.shape}"
        raise ValueError(msg)
    stack = _handler_stack(conditioner)
    grid = _time_grid(times, h)
    # Filter and Smoother rollouts do host-side segment bookkeeping, so their grids are
    # NumPy constants under jit; the LatentPathBuilder indexes its grids inside a scan and
    # takes jax arrays. dynestyx annotates all of them as jax Arrays, hence the untyped dict.
    explicit_path = any(isinstance(handler, LatentPathBuilder) for handler in stack)
    as_grid = jnp.asarray if explicit_path else np.asarray
    kwargs: dict[str, Any] = {"obs_times": as_grid(grid[: h.t_obs])}
    if controls is not None:
        kwargs |= {"ctrl_times": as_grid(grid), "ctrl_values": controls}
    empty_y = jnp.zeros((0, dynamics.observation_dim))
    empty_x = jnp.zeros((0, dynamics.state_dim))
    if h.future > 0:
        # Anchor at the last observed step: the simulator's first predicted state is the
        # conditioned draw at predict_times[0] with no transition, so that row is dropped.
        kwargs["predict_times"] = as_grid(grid[h.t_obs - 1 :])
        with numpyro.handlers.trace() as tr, ExitStack() as handlers:
            handlers.enter_context(Simulator(simulator_config, n_simulations=1))
            for handler in stack:
                handlers.enter_context(handler)
            dsx.sample(name, dynamics, obs_values=y, **kwargs)
        y_future = tr[f"{name}_predicted_observations"]["value"][0, 1:, :]
        x_future = tr[f"{name}_predicted_states"]["value"][0, 1:, :]
        numpyro.deterministic("forecast", y_future)
        return StateSpaceResult(y_future=y_future, x_future=x_future, y_in_sample=empty_y)
    with ExitStack() as handlers:
        for handler in stack:
            handlers.enter_context(handler)
        result = dsx.sample(name, dynamics, obs_values=y, **kwargs)
    if h.data is not None:
        return StateSpaceResult(y_future=empty_y, x_future=empty_x, y_in_sample=empty_y)
    x = _in_sample_states(name, stack[0], result)
    key = numpyro.prng_key()
    if key is None:
        msg = "the in-sample predictive draws observations and needs an active seed handler"
        raise RuntimeError(msg)
    u = None if controls is None else controls[..., : h.t_obs, :]

    def emit(x_t: Array, u_t: Array | None, t: Array, key_t: Array) -> Array:
        return jnp.asarray(dynamics.observation_model(x_t, u_t, t).sample(key_t))

    in_axes = (0, None if u is None else 0, 0, 0)
    keys = random.split(key, h.t_obs)
    y_in_sample = jax.vmap(emit, in_axes=in_axes)(x, u, jnp.asarray(grid[: h.t_obs]), keys)
    numpyro.deterministic("obs", y_in_sample)
    return StateSpaceResult(y_future=empty_y, x_future=empty_x, y_in_sample=y_in_sample)
```

Why each non-obvious line is there:

- `_handler_stack` normalizes one handler or a sequence to a tuple and validates it: exactly one conditioning handler, first. dsx handlers are entered outermost first (`with Simulator, Smoother, Discretizer`), and a `Discretizer` must be inside the conditioner because it rewrites the dynamics and forwards outward; a `Discretizer` first would hand continuous dynamics to a discrete smoother.
- `_time_grid` returns NumPy and `as_grid` converts per stack: the `Filter`/`Smoother` rollout needs host constants (even `jnp.asarray` of a NumPy grid is staged under the jitted driver and fails, A.6), the `LatentPathBuilder` needs jax arrays (A.8). The grids go through an untyped keyword dict because dsx annotates them as `jax.Array` while accepting array-likes; `ty` type-checks the notebooks in this repository.
- The inner `numpyro.handlers.trace()` in the forecast branch: `dsx.sample` returns the innermost handler's result, the conditioner's; the simulator's rollout reaches the model only through the sites it registers. An inner trace records those sites and lets them continue up the handler stack, so the outer `Predictive` trace still sees them (A.2).
- `predict_times` starting at `t_obs - 1`: the anchor semantics were measured, not assumed. Without the anchor, the first forecast row reproduces the conditioned state at `t_obs - 1` instead of stepping to `t_obs`, which is an off-by-one that the marginal variances expose (A.1). The same anchor holds for the builder's rollout, which starts from the final state of the path (A.8).
- A fresh `Simulator` per call: `Simulator` caches the concrete backend it resolves on its instance; a fresh instance per model execution keeps the block free of shared mutable state under `vmap`. The conditioning handlers, by contrast, are deliberately shared (Section 6.1).
- `_in_sample_states` rebuilds one batched `MultivariateNormal` from the per-time means and covariances rather than stacking the distribution objects with `jax.tree.map`: NumPyro distributions keep their `batch_shape` as static metadata through flattening, so a stacked object would sample one noise vector for all steps.
- Observations are sampled per step with `jax.vmap` over per-step keys instead of asking the observation model for one batched distribution, for the same reason.
- `y_future = ...[0, 1:, :]`: drop the `n_simulations` axis (always 1; the posterior sample axis is nf's) and the anchor row.
- `numpyro.deterministic("forecast", ...)` and `numpyro.deterministic("obs", ...)` inside the block: nf's drivers read those two names, the block owns the likelihood, and a model has one block call, as it has one `predict` call.

### 6.3 Models built on the block

The dynamics as a function of the parameters, so the same object serves the generative model, the prior predictive check and every inference strategy:

```python
def local_level_dynamics(q: Array, r: Array) -> DynamicalModel:
    return dsx.LTI_discrete(
        A=jnp.eye(1),
        Q=jnp.eye(1) * q**2,
        H=jnp.eye(1),
        R=jnp.eye(1) * r**2,
        initial_mean=jnp.zeros(1),
        initial_cov=jnp.eye(1) * 100.0,
    )
```

This is the local level model $x_t = x_{t-1} + w_t$, $w_t \sim \text{Normal}(0, q)$, $y_t = x_t + v_t$, $v_t \sim \text{Normal}(0, r)$, with $q$ and $r$ standard deviations squared into dsx's covariance arrays. The model function closes over the conditioner and reads the series from the covariates; the block registers the sites, so the model is the horizon, the parameters and one call:

```python
conditioner = Smoother(smoother_config=KFSmootherConfig(filter_source="cd_dynamax"))


def local_level(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]  # the series doubles as the covariate
    q = jnp.asarray(numpyro.sample("q", dist.HalfNormal(1.0)))  # state noise scale
    r = jnp.asarray(numpyro.sample("r", dist.HalfNormal(1.0)))  # observation noise scale
    state_space_series(h, "f", y, local_level_dynamics(q, r), conditioner=conditioner)
```

Replacing the conditioner by `Filter(filter_config=KFConfig())` or `LatentPathBuilder()` changes nothing else. The same generative process in nf's direct form is `innovations` plus `jnp.cumsum` plus `predict`, and the three fits agree (A.4, A.8).

With covariates in the observation equation, $y_t = H x_t + D u_t + v_t$, the series is the first covariate column and the regressors are the rest:

```python
def seasonal_level_dynamics(q: Array, r: Array, beta: Array) -> DynamicalModel:
    return dsx.LTI_discrete(
        A=jnp.eye(1),
        Q=jnp.eye(1) * q**2,
        H=jnp.eye(1),
        R=jnp.eye(1) * r**2,
        D=beta[None, :],
        initial_mean=jnp.zeros(1),
        initial_cov=jnp.eye(1) * 100.0,
    )


def seasonal_level(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]
    controls = covariates[..., 1:]
    q = jnp.asarray(numpyro.sample("q", dist.HalfNormal(1.0)))
    r = jnp.asarray(numpyro.sample("r", dist.HalfNormal(1.0)))
    beta = jnp.asarray(
        numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([controls.shape[-1]]).to_event(1))
    )
    state_space_series(
        h, "f", y, seasonal_level_dynamics(q, r, beta), conditioner=conditioner, controls=controls
    )
```

`LTI_discrete` infers `control_dim` from `D` when `B` is `None` since dynestyx#361, released in v0.5.1; on dynestyx 0.5.0 the same model needs the explicit `DynamicalModel(..., control_dim=k)` constructor (Section 10, item 7).

An observation-Markov model with the exact-observation idiom, an AR(1) whose state is the observation itself:

```python
def ar1_dynamics(phi: Array, sigma: Array) -> DynamicalModel:
    return dsx.DynamicalModel(
        initial_condition=dist.Normal(0.0, sigma / jnp.sqrt(1 - phi**2)).expand([1]).to_event(1),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(phi * x, sigma).to_event(1),
        observation_model=dsx.DiracIdentityObservation(),
    )
```

Under a `LatentPathBuilder` this model has no latent site at all (the path is the data), its factor is the exact likelihood $\sum_t \log p(y_t \mid y_{t-1}, \theta)$, its rollout is an AR(1) forecast and its in-sample predictive returns the data (A.9).

A continuous-time model on irregularly spaced observations, a mean-reverting (Ornstein-Uhlenbeck) level $dx_t = -\theta x_t \, dt + \sigma \, dW_t$, $y_t = x_t + v_t$, is where the block reaches beyond what nf's index-based blocks can express. Its cheap exact form is the two-handler stack of Section 8:

```python
def ou_dynamics(theta: Array, sigma: Array, r: Array) -> DynamicalModel:
    return dsx.LTI_continuous(
        A=-theta * jnp.eye(1),
        L=sigma * jnp.eye(1),
        H=jnp.eye(1),
        R=jnp.eye(1) * r**2,
        initial_mean=jnp.zeros(1),
        initial_cov=jnp.eye(1) * 4.0,
    )


conditioner = (
    Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert")),
    Discretizer(ExactAffineConfig(covariance_jitter=1e-6)),
)


def ou_level(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    theta = jnp.asarray(numpyro.sample("theta", dist.LogNormal(-1.5, 0.7)))
    sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0)))
    r = jnp.asarray(numpyro.sample("r", dist.HalfNormal(1.0)))
    state_space_series(h, "f", y, ou_dynamics(theta, sigma, r), conditioner=conditioner, times=times)
```

with `times` the irregular observation grid closed over by the model. The same model under `Smoother(smoother_config=ContinuousTimeKFSmootherConfig())` alone, without the `Discretizer`, is the continuous-discrete Kalman smoother: the same posterior at 50 times the cost per gradient (A.11). The robustness caveats of the discretized form are in Section 8.

## 7. Inference and drivers matrix

| Component | `Filter` | `Smoother` | `LatentPathBuilder` | Evidence |
| --- | --- | --- | --- | --- |
| `MCMC(NUTS(model))`, 1 or more chains | yes | yes, identical density and draws | yes; the path site is sampled | A.3, A.4, A.7, A.8 |
| `SVI(model, AutoNormal(model), ...)` + `draw_posterior` | yes; the guide covers only the parameters | yes | expected (the path site is a plain unconstrained site); not exercised | A.4 |
| `numpyro_forecast.contrib.blackjax` kernels | expected (same `initialize_model` density); not exercised | idem | idem | none |
| `forecast(key, model, posterior, data, covariates)` | yes, including `batch_size`, `parallel=False`, `device` | yes | yes (shared builder instance) | A.3, A.7, A.8 |
| `predict_in_sample`, `to_datatree` | no: raises with the alternatives | yes | yes | A.7, A.8 |
| `backtest(...)` with a NUTS or SVI `forecast_fn`; `eval_train=True` with an `in_sample_fn` | forecasts yes, `eval_train` no | yes | yes (per-window shapes warm the builder's cache) | A.3, A.10 |
| `eval_crps`, `eval_coverage`, `eval_mae`, `eval_rmse`, `evaluate_forecast`, `metrics.*` | yes (they only see draws) | yes | yes | A.3 |
| `predictions_to_datatree(draws, ...)`, ArviZ plots and `az.summary` on the posterior | yes | yes | yes | design |
| `dsx.simulate` prior predictive checks | yes (independent of the conditioner) | yes | yes | A.6 |
| `backtest_vectorized` (vmapped SVI over rolling windows) | plausible, not exercised | idem | idem | none |
| `dsx.plate` / batched panels, nf batch dims left of time | not in scope (Section 12) | idem | idem | none |

A posterior from `mcmc.get_samples()` or `draw_posterior` also carries dsx's deterministic sites (`f_filtered_states_mean`, `f_smoothed_states_*`, `f_state_path`, `f_marginal_loglik`, ...). Passing that dict straight to `forecast()` or `predict_in_sample()` is fine: `Predictive` substitutes the recorded values into the trace, but the rollout and the in-sample draw read the conditioner's in-memory result, so the draws are unaffected (verified throughout Appendix A, which always forecasts from the full dict). They do cost memory proportional to `(draws, t_obs, state)`; set the `record_*` fields of the config to `False` when they are not wanted.

## 8. Covariates, time and windows

- nf's `covariates` array spans the full horizon and is the only channel through which anything reaches the model at prediction time. Two things travel in it: the observed series (read back as `y` over the first `t_obs` rows; only those rows are ever read) and the exogenous inputs (forwarded as `controls` over the full horizon). The block maps the controls to `ctrl_values` on the full time grid, which satisfies the simulator's rule that `ctrl_times` contain every prediction time exactly.
- Regression enters through `D` (observation) or `B` (state) of the linear-Gaussian classes, or through the `u` argument of a callable transition or observation model. Seasonality is either a regression on `fourier_features` covariates (`D`), or a seasonal state block in `A` with `H` selecting it (no covariates needed).
- `times` carries the observation times and, in its last `future` entries, the forecast times, regular or not: the `Horizon` counts the steps, the grid places them (Section 6.1). Time-varying linear-Gaussian parameters (callables of `t`) are supported by `KFConfig(filter_source="cuthbert")` only; continuous-time dynamics read the actual intervals from the grid, on both sides of the split.
- **Continuous-time models on irregular grids** are the case dsx supports and nf's index-based blocks cannot express at all, and the block carries them unchanged with `times` set to the irregular grid (A.11). Two conditioners are exact for a linear SDE. `Smoother(ContinuousTimeKFSmootherConfig())` is the continuous-discrete Kalman smoother, which solves the moment ODEs on every interval: correct, and about 80 ms per gradient at $T = 120$, so a NUTS run of 600 iterations took 521 s. `(Smoother(KFSmootherConfig(filter_source="cuthbert")), Discretizer(ExactAffineConfig(...)))` discretizes the SDE exactly on the observation grid (interval-dependent $A_k = e^{A \Delta t_k}$ and $Q_k$, which is why the `cuthbert` backend is required) and runs the discrete smoother: the same log likelihood to three decimals at 1.4 ms per gradient, and the same NUTS run in 28 s. The discretized form has three robustness requirements that the continuous-discrete form does not: dsx's runtime checks in the discretizer raise an exception on a non-finite transition covariance, which aborts HMC on the extreme proposals of its step-size search instead of counting a divergence, so `EQX_ON_ERROR=nan` must be set in the environment before dynestyx (equinox) is imported; `ExactAffineConfig(covariance_jitter=1e-6)` guards the singular covariances of tiny intervals; and the prior must keep $\theta$ away from 0, where $\sigma^2 (1 - e^{-2\theta \Delta t}) / (2 \theta)$ is a $0/0$ limit (a `LogNormal` prior on a mean-reversion rate, as in Section 6.3). Without all three the sampler either aborts or collapses its step size and runs at the maximum tree depth (A.11). These are the reasons the example notebook does not include a continuous-time example yet (Section 14) and the first items for the dsx maintainers in Section 17.
- In a `backtest` the block sees `times[:t_obs]` of the grid the model closes over, so the window is implicitly assumed to start at the first entry. This is exact for expanding windows and for time-invariant dynamics on any window; a rolling window (`t0 > 0`) with absolute-time-dependent parameters or irregular spacing cannot be expressed today, because `backtest`'s `model_fn()` receives no window offset. Listed as an nf follow-up (Section 17).

## 9. Mapping nf models to dsx dynamics

Two families of nf models map to dsx, in different ways.

Models whose latent path nf **samples** (`innovations` plus arithmetic, `markov_series`) map to a `DynamicalModel` with a proper stochastic state, and the conditioner chooses the inference: `Filter`/`Smoother` marginalize the path (exactly for linear-Gaussian dynamics, approximately otherwise), `LatentPathBuilder` samples it explicitly with dsx's machinery, which is the closest analog of nf's direct form and the road to discretized SDE/ODE latents (`Discretizer`) and to NaN-aware observations (its `missing_observation_strategy`).

Models whose latent path nf **filters deterministically** (`ssoe`: ARMA, exponential smoothing in innovations form, Croston/TSB) split in two. Members that are Markov in the observations themselves (autoregressions, and the `ssoe` recursions whose state is a window of past observations) are written in dsx with `DiracIdentityObservation` under a `LatentPathBuilder`: the state is the observation, no latent is sampled, and the factor is the exact likelihood (Section 6.3, A.9). Members with a latent level that is neither observed nor a proper stochastic state (the innovations-form ETS family, whose level is a deterministic function of the past observations and the single error) have no dsx dynamics: the exact-observation idiom requires the whole state to be observed, and a linear-Gaussian transition needs a full-rank noise covariance, which the single source of error violates. Those models stay on nf's `ssoe`, which is already the marginalized form (a deterministic recursion, no latent sampling, nothing for a filter to add).

| nf idiom | dsx dynamics | Conditioner | Notes |
| --- | --- | --- | --- |
| Random walk level: `innovations` + `cumsum` | `LTI_discrete(A=I, Q=q^2, H=I, R=r^2)` | `Filter`/`Smoother` with `KFConfig`/`KFSmootherConfig` (exact); `LatentPathBuilder` | Example 1, three strategies |
| Local linear trend (level + slope) | `A=[[1, 1], [0, 1]]`, `H=[[1, 0]]`, diagonal `Q` | idem | |
| Level + regression on covariates | `LTI_discrete(..., D=beta[None, :])` | idem | Example 2 |
| Seasonal dummies / Fourier state | seasonal block in `A`, `H` selecting the current season | idem | alternative to `D` |
| VAR(p) as an explicit latent (`markov_series` + `var_step`) | `A = numpyro_forecast.var.companion_matrix(coefs)`, `H` selecting the first lag block | idem | nf already ships the companion matrix |
| AR(p) on the observations (`ssoe`) | lag-window state, `DiracIdentityObservation` | `LatentPathBuilder` (exact, no latent site) | A.9 for `p = 1` |
| `markov_series` with a nonlinear Gaussian transition | callable `state_evolution(x, u, t_now, t_next)` returning a Gaussian | `EnKFConfig`, `UKFConfig`, `EKFConfig` (approximate, pseudo-marginal); `LatentPathBuilder` (explicit) | |
| `markov_series` + count/heavy-tailed `predict` link | custom `ObservationModel` returning `Poisson`, `NegativeBinomial`, `StudentT` | `PFConfig` (approximate, pseudo-marginal); `LatentPathBuilder` (explicit) | |
| Continuous-time latent (irregular sampling) | `LTI_continuous`, `AffineDrift` + `Diffusion` | `Smoother(ContinuousTimeKFSmootherConfig())` (exact, slow); `(Smoother(KFSmootherConfig(filter_source="cuthbert")), Discretizer(ExactAffineConfig()))` (exact, 50x cheaper, robustness caveats in Section 8); `Discretizer` + `LatentPathBuilder` for nonlinear SDEs | `times` carries the intervals; A.11 |
| Missing observations | `NaN` in `y` | `KFConfig(filter_source="cuthbert")`; `LatentPathBuilder(missing_observation_strategy=...)` | nf models have no missing-data story of their own; roadmap |

## 10. Pinned dynestyx facts

Everything the block depends on, with where it was checked:

1. Installation: `uv pip install dynestyx==0.5.0` into the `--extra all` environment resolves without changing `jax`, `numpyro` or `arviz`; it adds about 60 packages, among them `cd-dynamax`, `cuthbert`, `effectful`, `equinox`, `diffrax`, `flax`, `orbax-checkpoint`, `tfp-nightly`, `mkdocs`. The `ml-bugfix-controldim-LTI` branch installs the same way from git. It shipped as `dynestyx==0.5.1` on 2026-09-15 (dynestyx#361, a vectorized `summarize_gaussian_mixture`, `cd-dynamax` bumped to 0.4.3); whether the release resolves against the same pins was not checked.
2. `dsx.sample(name, dynamics, obs_times=, obs_values=)` under `Filter(KFConfig(...))` registers the sites `f_marginal_log_likelihood` (sample site of a `Unit` distribution, the factor), `f_marginal_loglik` (deterministic scalar) and, by default, `f_filtered_states_mean` `(T, state)`, `f_filtered_states_cov` `(T, state, state)`, `f_filtered_states_cov_diag` and `f_filtered_states_chol_cov`. `mcmc.get_samples()` returns the deterministic ones next to the parameters (A.2, A.3).
3. `Simulator(n_simulations=1)` (or `DiscreteTimeSimulator`) outside the conditioner, with `predict_times`, registers `f_predicted_observations` `(n_sim, T_pred, obs)`, `f_predicted_states` `(n_sim, T_pred, state)`, `f_predicted_times` `(n_sim, T_pred)` and `f_{j}_x_0` per segment; `Predictive` prepends the draw axis; `dynestyx.flatten_draws` merges `(draws, n_sim)` (A.2).
4. `dsx.sample` returns the conditioner's result object: a `ConditionedResult` (fields `marginal_loglik`, `times`, `states`, `dists`, `predicted_observations`) for `Filter` and `Smoother`, a `LatentStateResult` (fields `state_path`, `state_path_times`, `state_dists`, `joint_log_prob`, ...) for `LatentPathBuilder`. Its `predicted_observations` field holds the filter's one-step-ahead outputs, not the simulator's rollout (A.2).
5. Anchor semantics: the discrete simulator's first state is the initial-condition draw at `predict_times[0]` with no transition; in a posterior rollout that draw comes from the conditioned distribution at the last observation time $\leq$ `predict_times[0]` (`Filter`, `Smoother`), or from the final state of the path (`LatentPathBuilder`, which only supports `predict_times >= max(state_path_times)`). With `predict_times = [T, ...]` the first forecast row therefore has the variance of the filtered state at $T - 1$ plus observation noise, identical to the anchored row at $T - 1$ (A.1, A.8).
6. One-step-ahead predicted-observation outputs (`f_predicted_observations_mean`/`_cov` sites, and the `Evaluation` handler that scores them) are produced only by the continuous-time filters and by `EnKFConfig(filter_source="cuthbert")`; the discrete `KFConfig` exposes the filtered states instead.
7. `LTI_discrete(A, Q, H, R, B=None, b=None, D=None, d=None, initial_mean=None, initial_cov=None)` infers `control_dim` from `B`, and, since dynestyx#361 (released in v0.5.1), from `D` when `B` is `None`. On dynestyx 0.5.0 a `D`-only model raised `Controls are provided (shape: (120, 4)), but dynamics.control_dim is 0` (A.4) and needed the explicit `DynamicalModel(..., control_dim=k)` constructor.
8. Filter backends for the Kalman filter: `KFConfig()` defaults to `filter_source="cd_dynamax"`; `"cuthbert"` adds missing-data (`NaN`) support, time-varying callable parameters and an associative (parallel-in-time) scan enabled by default. All three give the same log likelihood; their costs differ by an order of magnitude on CPU at $T = 120$ (A.5).
9. dsx reads randomness from `numpyro.prng_key()` inside the active `seed` handler (`BaseFilterConfig.crn_seed` pins it for stochastic filters); deterministic filters need none.
10. `Smoother(smoother_config=KFSmootherConfig(filter_source="cd_dynamax"))` registers the same factor as the filter (identical log density and, under the same key, identical NUTS draws), returns the per-time smoothing distributions as `ConditionedResult.dists` (one `MultivariateNormal` per observation time), and records `f_smoothed_states_mean`/`_cov`/`_cov_diag` when its `record_smoothed_*` flags are set. A `Simulator` outside it rolls out from the smoothed distribution at the anchor, for `predict_times >= max(obs_times)` only (A.7, A.10).
11. `LatentPathBuilder` registers one sample site `f_state_path_params` of shape `(t_obs, state)` with an improper-uniform prior whose forward simulation seeds the sampler, the factor `f_joint_log_prob_factor` (the joint state-observation density) and the deterministics `f_state_path`, `f_state_path_times`, `f_state_path_param_times`, `f_joint_log_prob`. Under `Predictive` the substituted site reconstructs the posterior path; a `Simulator` outside the builder rolls out from the final state only. With `DiracIdentityObservation` and no missing data the site has shape `(0,)` and the path is the data (A.8, A.9).
12. `LatentPathBuilder` needs a concrete observation layout: it computes one eagerly the first time it sees concrete `obs_values` and caches it per `(name, obs_values.shape)`; under a `jit` with traced observations (nf's `forecast()` and `predict_in_sample()` drivers) it can only reuse a cached layout, so the model must reuse the instance that ran the fit. A fresh `LatentPathBuilder()` inside the model raises `needs a fixed observation missingness pattern, but cannot infer one from traced obs_values` at forecast time (A.8).
13. Time grids: the `Filter`/`Smoother` rollout does its segment bookkeeping on the host and needs concrete (NumPy) `obs_times`/`predict_times` under `jit`; the `LatentPathBuilder`'s forward sampler and `dsx.simulate` index the grid inside a `lax.scan` and need jax arrays (A.6, A.8).
14. `Discretizer` is an innermost handler: it rewrites a continuous-time `DynamicalModel` into a discrete transition (exact Gaussian discretization for an affine SDE with `ExactAffineConfig`, the default for that case) and forwards outward, so the composition is `with Smoother(...), Discretizer(...)`. The discretized transition has interval-dependent parameters, which the `cd_dynamax` backend rejects (`requires constant (array-valued) parameters`) and the `cuthbert` backend accepts. Its log likelihood equals the continuous-discrete Kalman smoother's to three decimals at $T = 120$ on an irregular grid (A.11).
15. dsx guards the discretizer with `equinox.error_if` checks (`Transition covariance is singular or indefinite`, `Discretization produced a non-finite transition covariance`). Under NUTS these raise on the extreme proposals of the step-size search even when the density and its gradient are finite on the whole parameter range of interest; `EQX_ON_ERROR=nan`, read by equinox at import, turns them into NaN densities that NumPyro counts as divergences (A.11).
16. In-window prediction: `Smoother` (and `LatentPathBuilder`) validate `predict_times >= max(obs_times)` and reject anything earlier with `in-window smoothing predictions are not implemented yet. Please use Filter for in-window predictions for now`; `Filter` accepts in-window `predict_times` and starts each such segment from the filtering distribution at the preceding observation time (A.1 used one such point as the anchor).

## 11. Performance

Local level model, $T = 120$, `MCMC(NUTS)` with 2 chains of 500 warmup plus 500 samples, Apple M4 Pro, CPU, `float32`, one common data draw (A.7, A.8):

| Model and conditioner | Wall time | ESS of $q$ | ESS of $r$ | Leapfrog steps |
| --- | --- | --- | --- | --- |
| dsx, `Filter(KFConfig())` (cd_dynamax) | 5.0 s | 479 | 459 | 4 822 |
| dsx, `Smoother(KFSmootherConfig(filter_source="cd_dynamax"))` | 4.1 s | 479 | 459 | 4 822 |
| dsx, `LatentPathBuilder()` (explicit path) | 5.1 s | 140 | 368 | 21 992 |
| nf direct (`innovations`, no reparameterization; A.5, different data draw) | 3.8 s | 76 | 327 | 236 568 |

Per gradient evaluation of the log density (`jax.value_and_grad`, jitted, same model and data): nf direct 0.01 ms, cd_dynamax KF 0.17 ms, cd_dynamax RTS smoother 0.17 ms (the backward pass is dead code under `grad` and only runs for the in-sample predictive), cuthbert associative KF 1.26 ms, cuthbert sequential KF 2.52 ms. Recording fewer filter outputs does not change the gradient cost.

Reading: the marginalized model needs 50 times fewer gradient evaluations than nf's direct model for 6 times the effective sample size of the state noise scale, because NUTS moves in a 2-dimensional posterior instead of a 122-dimensional one with a funnel between $q$ and the path. The explicit path through the builder sits in between (5 times fewer gradients than the direct model, twice its ESS; the improper-uniform parameterization has no `LocScaleReparam` equivalent yet). On CPU at this length the per-gradient cost of the filter cancels most of the wall-clock advantage; the advantage grows with $T$ (the direct posterior grows with $T$, the filter's cost is linear in $T$ and does not affect the sampler's geometry) and on accelerators (where the associative scan pays off). `LocScaleReparam` on the direct model lifts its ESS to about 350 (A.4) and remains the right baseline for a fair comparison in the notebook.

Forecasting 300 draws over 24 steps with `forecast()` (jit plus `vmap`, including compilation): 0.7 s, the re-conditioning of every draw on the 120-step window included. That re-conditioning is a `vmap`ped pass of the conditioner over the window, so its cost grows with `t_obs` times the number of draws and with the conditioner (a particle filter costs more per step than the Kalman filter); at this size it is nowhere near dominant, and the cached-anchor rollout of Approach C is the fallback when it becomes so. `backtest` with a NUTS closure (200 warmup, 200 samples per window): about 27 s per window, dominated by recompilation of the fit per window shape; `eval_train=True` doubles the fits.

The committed notebook reproduces the picture with the fair baseline (`LocScaleReparam` on the direct model), 4 chains of 1 000 plus 1 000 iterations and the three strategies side by side; its numbers are in Section 14 of the notebook itself.

Recommendations that follow: a `Smoother` with `KFSmootherConfig(filter_source="cd_dynamax")` by default for linear-Gaussian models (same cost as the filter, and the in-sample predictive comes with it); `filter_source="cuthbert"` when `NaN` observations or callable time-varying parameters are needed, or on a GPU with long series; `LatentPathBuilder` when the path itself is the object of interest, the observation model is non-Gaussian, or the dynamics are discretized continuous-time; `LocScaleReparam` on the direct model in any comparison.

Continuous-time (A.11): Ornstein-Uhlenbeck level on 144 irregularly spaced observations, $T = 120$. Per gradient, `Smoother(ContinuousTimeKFSmootherConfig())` 78 to 96 ms versus `(Smoother(KFSmootherConfig(filter_source="cuthbert")), Discretizer(ExactAffineConfig()))` 1.4 to 1.9 ms at identical log likelihoods. NUTS, one chain of 300 plus 300: 521 s versus 28 s (the latter with `EQX_ON_ERROR=nan`, a covariance jitter and a `LogNormal` prior on $\theta$). Forecasting from either: `(draws, 24, 1)` with the marginal variance growing with the actual gap lengths.

## 12. Boundaries and non-goals

- **In-window prediction.** dsx's `Smoother` and `LatentPathBuilder` support `predict_times >= max(obs_times)` only ([dynestyx#272](https://github.com/BasisResearch/dynestyx/issues/272)); the block never asks for less, and the in-sample predictive comes from the conditioned distributions, not from a rollout. The target form of that mode is the same `Simulator` ∘ stack composition as forecasting, with `predict_times = times[:t_obs]`: dsx supports it for `Filter` today (yielding the filtering predictive, a different quantity) and will for the other two once #272 lands, at which point `_in_sample_states` and the block's own `{name}_smoothed_states` site disappear.
- **Marginal, not joint, in-sample predictive.** The in-sample draws are per-time marginals of the smoothing predictive; a functional of the whole path (a running maximum, a turning-point count) needs backward simulation, which the block does not do. nf's direct models produce joint path draws; for per-time scores and bands the two coincide.
- **A `Filter` conditioner has no in-sample predictive.** The block raises; use a `Smoother` (same fit) or read `f_filtered_states_*` for the filtering view.
- **Continuous-time models are supported but not showcased.** The block carries them (Section 8, A.11); the cheap exact form needs three environment-level workarounds today, so the example notebook stays with discrete-time models until dsx's discretizer tolerates HMC exploration.
- **Absolute time in rolling windows** (Section 8): expanding windows only, or time-invariant dynamics.
- **Prior predictive checks** use `dsx.simulate` (pure JAX, no NumPyro) rather than `Predictive(model)(key, covariates)`, which would hit the in-sample branch.
- **Panels and hierarchy** (`dsx.plate`, nf batch dims to the left of time) are not addressed. dsx's plate is an effectful handler with its own batching contract for arrays and dynamics; mapping it onto nf's leftward batch dims is a separate design.
- **`backtest_vectorized`** is not exercised.
- **Nothing is built in the other direction** (nf building blocks inside a dsx model). nf's blocks register sites against a `Horizon`; dsx models are Equinox modules of dynamics. There is no natural composition and no demand for one.
- **No new inference machinery** (no fit helpers, no forecaster objects). `svi.run` and `mcmc.run` on the model are the whole story.

## 13. Packaging plan

### v1 (example notebook)

- `docs/examples/dynestyx_integration.ipynb`, authored with jupytext (`py:percent`), executed, committed with outputs, `.py` deleted. The block (`StateSpaceResult`, `state_space_series` and its two helpers) is defined in the notebook verbatim as in Section 6.2.
- `pyproject.toml`: `dynestyx>=0.5.1` joins the `docs` extra, which `all` and `all_cuda` include, so `uv sync --extra all` (the `make setup` every CI job runs) installs it. The first revision kept dsx in a separate extra outside `all` to spare CI its transitive closure (about 60 packages, `tfp-nightly` among them), but `ty` type-checks the notebooks, so the `prek` CI job would fail on the notebook's unresolved `dynestyx` import as soon as it lands on `main`; the docs tooling is where a dependency that exists only for an example belongs. The pin is `>=0.5.1`, the first release with dynestyx#361, so the `D`-only `LTI_discrete` shortcut of Section 6.3 works out of the box and the notebook needs no 0.5.0 fallback and no git reference (which would block publishing to PyPI). `uv.lock` is gitignored in this repository, so nothing else changes in the commit.
- `README.md`: the sentence listing the optional extras says that `docs` also installs `dynestyx` for the example.
- No change under `numpyro_forecast/`, no `reference:` change, `tests/test_docs_reference.py` unaffected.

### v2 (proposed, after review by both maintainers)

- `numpyro_forecast/contrib/dynestyx.py` with `state_space_series`, `StateSpaceResult` and `StateSpaceHandler`, lazily importing dsx through `numpyro_forecast.optional.require("dynestyx", extra="dynestyx")` on first call (a user-facing `dynestyx` extra reappears with v2, because a `contrib` module cannot ask users for the docs tooling) (the package import must stay free of dsx, matching the `base-import` CI leg's invariant and `contrib/blackjax.py`), an `_api_canary("dynestyx", ["sample", "Filter", "Smoother", "LatentPathBuilder", "Discretizer", "Simulator", "DynamicalModel", "simulate"])` tripwire, and signatures that never name dsx types (`dynamics: object`, `conditioner: object`, as `contrib/blackjax.py` does for blackjax, because the jaxtyping/beartype import hook resolves annotations at call time and a `TYPE_CHECKING`-only import would not resolve). The `isinstance` dispatch in `_in_sample_states` then goes through the lazily imported module.
- `great-docs.yml`: `contrib.dynestyx.state_space_series`, `contrib.dynestyx.StateSpaceResult` and `contrib.dynestyx.StateSpaceHandler` under "Extensions (contrib)".
- `tests/test_contrib_dynestyx.py` guarded by `pytest.importorskip("dynestyx")`: the anchor invariant (first horizon row has the conditioned variance plus one transition) for all three conditioners, the `Filter` in-sample error, `forecast()` and `predict_in_sample()` shapes under `batch_size`/`parallel=False` for all three, agreement of the KF marginal likelihood with a direct `MultivariateNormal` log density on a tiny series, the irregular `times` pass-through, the rejection of malformed handler stacks, the `(Smoother, Discretizer)` stack on a continuous-time model, and one `backtest` window with `eval_train=True`. Whether CI installs the extra for these (a separate job) or they stay skip-by-default is an open question (Section 17).
- The notebook then imports the block from the package and drops its local definition.

## 14. Example scaffolding

One notebook, two examples, both synthetic so the truth is known and the exactness claims can be checked.

**Example 1: local level, one process, three inference strategies.** Simulate $y_t$ from the local level model with known $q$ and $r$ ($T = 120$ training steps, 24 held out) with `dsx.simulate`. Fit nf's direct model (`innovations` + `cumsum` + `predict`, `LocScaleReparam`), the dsx model under a `LatentPathBuilder` (explicit path, built by dsx) and the dsx model under a `Smoother` (marginalized), same priors, same NUTS budget. Compare: posterior of $q$, $r$ against truth, ESS, leapfrog steps and wall time, the three forecast fans ($50\%$ and $94\%$ HDI), CRPS and coverage on the held-out window. In-sample: `to_datatree` on all three models, their in-sample predictive bands side by side (the smoothing predictive for the smoother, the posterior path plus observation noise for the builder, the replayed path plus noise for the direct model), then the latent level itself: filtered versus smoothed from the recorded sites as the pedagogy for what a `Filter` cannot serve, and the smoothed level against the builder's posterior path (`f_state_path`) and the direct model's `x0 + cumsum(drift)`, the three reconstructions of the same posterior, which the dsx maintainers asked to see on numpyro_forecast#126. Prior predictive check with `dsx.simulate`.

**Example 2: local level with seasonal regression, covariates through the block.** Add a Fourier seasonal component to the simulated series; the covariate array is the series followed by `fourier_features`, which the block forwards as `ctrl_values` and the dynamics consume through `D` of `LTI_discrete`. Fit with SVI (`AutoNormal`, `draw_posterior`) to show the variational path, forecast, then run `backtest` on expanding windows with a NUTS `forecast_fn` and an `in_sample_fn`, `eval_train=True`, plotting in-sample and out-of-sample CRPS and the coverage per window. This is Direction (1) of dynestyx#264 made concrete: nf's evaluation workflow applied to a dsx model.

Notebook outline (jupytext `py:percent`; conventions per `AGENTS.md`: `description` metadata, `thumbnail` cell tag on the forecast overlay figure, `$94\%$ HDI` in LaTeX, `\text{Normal}`, no em-dashes, no `plt.show()`, no RST markup in prose):

1. Title and motivation: the two issues, what each library is, the seam, the three conditioners, what the reader will see.
2. Setup: imports, `az.style`, `rng_key`, versions printed.
3. The block: `StateSpaceResult`, `state_space_series`, a short explanation of the conditioner stack, the series-in-covariates contract, the sites the block registers, the anchor and the time grids.
4. Example 1 data (`dsx.simulate`), plot.
5. Prior predictive check with `dsx.simulate`.
6. The three models; fit all with NUTS; comparison table; posterior overlays.
7. Forecasts with `forecast`; three-panel fan plot (thumbnail); metrics table.
8. In-sample: `to_datatree` for the three models, three-panel predictive bands, the filtered-versus-smoothed level, and the smoothed level against the builder's and the direct model's posterior paths.
9. Example 2 data with seasonality; the regression model; SVI fit; forecast plot.
10. `backtest` with NUTS closures and `eval_train=True`; per-window CRPS (in and out of sample) and coverage plots; `results_to_dataframe`.
11. Takeaways, boundaries (Section 12 in two paragraphs, including why there is no continuous-time example yet), link to this document.

File map:

```
DYNESTYX_INTEGRATION_DESIGN.md              # this document
docs/examples/dynestyx_integration.py       # jupytext source, deleted after execution
docs/examples/dynestyx_integration.ipynb    # committed with outputs
pyproject.toml                              # `dynestyx` in the `docs` extra
README.md                                   # extras sentence
```

## 15. Verification plan

1. `uv sync --extra all` installs `dynestyx==0.5.1` through the `docs` extra (v0.5.1 ships dynestyx#361, so the git install of the `ml-bugfix-controldim-LTI` branch that the probes used is no longer needed); `python -c "import numpyro_forecast"` does not import `dynestyx` (the `base-import` invariant).
2. The notebook executes end to end with `uv run jupytext --to notebook --execute docs/examples/dynestyx_integration.py`, every figure embedded.
3. Numerical checks inside the notebook: recovered $q$, $r$ within two posterior standard deviations of the truth for all three fits; the three forecast fans agree visually and their CRPS agree within Monte Carlo error; the anchor invariant holds (the marginal variance of the first forecast row exceeds that of the anchor row by about $q^2$, one transition); the in-sample predictive bands of the smoothed and the direct model coincide.
4. `uv run ruff check docs/examples/dynestyx_integration.ipynb && uv run ruff format --check docs/examples/dynestyx_integration.ipynb`; `uv run ty check` (the notebooks are type-checked).
5. `uv run pytest tests/test_docstring_markup.py tests/test_docs_reference.py` green (markup rules apply to notebook cells; no public API changes).
6. `prek run --all-files` green.
7. `make docs` renders the new example page with card description and thumbnail (optional, needs Quarto).

## 16. Risks

1. **dsx API drift.** The block relies on site names (`{name}_predicted_observations`, `{name}_predicted_states`), the anchor semantics, the result objects' `dists`/`state_path` fields and the `Simulator` ∘ conditioner composition. All are documented in dsx's docstrings and tutorials, but the library is at 0.x. Mitigation: pin `>=0.5.1` now, add the `_api_canary` and the anchor tests in v2, and re-run the notebook on each dsx minor.
2. **Silent off-by-one.** A user who forgets the anchor gets a forecast that lags one step with plausible-looking uncertainty. Mitigation: the block owns the anchor; the notebook and the v2 tests assert the variance increment.
3. **Traced time grids.** Anyone rewriting the block with `jnp.arange`, or deriving `times` from a covariate column, breaks the `Filter`/`Smoother` rollout under `jit`; anyone passing NumPy grids to the builder breaks its scan. Mitigation: `_time_grid` plus the per-conditioner conversion; v2 tests run `forecast()` (which jits) for all three.
4. **Builder instance created inside the model.** Works for the fit, fails at the first jitted `forecast()` with dsx's layout error. Mitigation: the docstring and the notebook create the conditioner outside the model; a v2 test covers it.
5. **Approximate default configs.** `Filter()`/`Smoother()` with no config silently use an ensemble/extended Kalman method on a linear-Gaussian model. Mitigation: every example passes the KF configs explicitly and the docstring says so; consider a `KFConfig` default when `dynamics` is linear-Gaussian in v2.
6. **Dependency weight.** dsx's transitive closure includes `tfp-nightly`, and through the `docs` extra every CI job installs it. Mitigation: accepted for now, since `ty` needs the import resolvable to check the notebook; the package's own dependencies and the `base-import` leg are untouched, and a `ty` exclusion for the notebook plus a separate extra is the way back if the closure ever breaks a CI leg.
7. **Release lag.** dynestyx#361 shipped in v0.5.1 on 2026-09-15, but the probes from A.7 onward ran on the branch that became it, and the release also bumps `cd-dynamax` to 0.4.3, which the probes did not see. Mitigation: execute the notebook on `dynestyx==0.5.1` before committing it (Section 15) and keep the pin at `>=0.5.1`.
8. **Discretized continuous-time models under HMC.** The exact discretization is 50 times cheaper than the continuous-discrete smoother but aborts or stalls NUTS without `EQX_ON_ERROR=nan`, a covariance jitter and a prior bounded away from $\theta = 0$ (Section 8). Mitigation: documented, not showcased; raised with the dsx maintainers (Section 17).

## 17. Follow-ups and questions

For `numpyro_forecast`:

- Promote the block to `contrib/dynestyx.py` (Section 13, v2) once the dsx maintainers have reviewed the composition.
- A window offset for `backtest`'s model factory (or a documented closure pattern) so rolling windows can carry absolute `times`.
- Panel support through `dsx.plate`.
- A continuous-time example (Section 6.3's Ornstein-Uhlenbeck level on irregular times, or an SDE latent under `Discretizer` + `LatentPathBuilder`) once the discretizer's runtime checks cooperate with HMC; the case the dsx maintainers single out as their most distinctive.
- A `Horizon` predictive mode set by the drivers (`predict_in_sample`/`to_datatree` passing `data` plus a mode instead of `data=None`) would remove the series-in-covariates redundancy for `ssoe` and for this block alike.
- `to_datatree(time_coord=...)` rejects a NumPy array at its beartype check (`Sequence[Any]`); accepting array-likes is needed for irregular time coordinates (A.11).

For `dynestyx` (to raise upstream, wording to be agreed):

- Inference-time runtime checks (`equinox.error_if` in the discretizer) should propagate NaN rather than raise, so HMC counts a divergence instead of aborting; today this needs `EQX_ON_ERROR=nan` (Section 8, A.11). Relatedly, the exact-affine covariance is a $0/0$ limit at $\theta \to 0$; a stable formula there would remove the prior constraint.
- In-window prediction under `Smoother`/`LatentPathBuilder` (dynestyx#272) would let the block's in-sample mode be the same `Simulator` composition as its forecast mode (Section 12) and delete its hand-rolled draw.
- Trace-safe segment bookkeeping in the `Filter`/`Smoother` rollout (or a blessed pure-JAX rollout entry point, Approach C) would remove the NumPy-grid requirement and allow `times` to come from a covariate column.
- A replay mode for the conditioners (`frozen_data=True`, the maintainers' name for it on #125) that reuses the conditioned state of the fit instead of re-conditioning on the window under `Predictive`, for the cases where the per-draw pass over the window dominates the forecast (Section 3); until then the cached-anchor rollout of Approach C is the block-side fallback.
- A documented way to obtain the simulator's rollout from `dsx.sample`'s return value when a `Simulator` wraps a conditioner (today only the sites carry it).
- The anchor semantics of posterior rollouts deserve a line in the `DiscreteTimeSimulator` docstring.
- One-step-ahead predicted observations for the discrete `KFConfig` backends, so `Evaluation` covers the exact linear-Gaussian case.
- Whether the innovations-form ETS family can be expressed with `DiracIdentityObservation` in a way this document missed (Section 9).

Open decisions:

- Whether v2 tests run in CI with the extra installed (a fifth job) or stay `importorskip`.
- The block's name. `state_space_series` names the model class and leaves the strategy to the conditioner; `filtered_series` (the first revision) named one strategy and no longer fits.

## Appendix A: probe results

All probes ran on Apple M4 Pro, CPU, `float32`, in the `uv` environment of this branch. A.1 to A.6 ran on `dynestyx==0.5.0`; A.7 onward on the `ml-bugfix-controldim-LTI` branch, since released as v0.5.1 (the probes were not re-run on the release). Scripts were throwaway and are not committed.

### A.1 Anchor semantics

Local level, $T = 120$, KF (cuthbert), NUTS 300 plus 300 draws, rollout under `Predictive` with `DiscreteTimeSimulator(n_simulations=1)` outside `Filter`. Marginal variance of `f_predicted_observations` across draws:

| `predict_times` | row 0 | row 1 | row 2 |
| --- | --- | --- | --- |
| `[119, 120, 121, ...]` (anchored) | 0.339 | 0.375 | 0.534 |
| `[120, 121, 122, ...]` (no anchor) | 0.339 | 0.375 | 0.534 |

The rows are identical, so the unanchored rollout labels the filtered state at 119 as the forecast for 120. In the anchored run the successive variance increments were 0.035, 0.159, 0.047 and 0.140 against a posterior mean of $q^2$ of 0.104: one transition per row, up to Monte Carlo noise at 300 draws.

### A.2 Sites

Filter-only trace: `q`, `r`, `f_marginal_log_likelihood` (sample, shape `(0,)`), `f_marginal_loglik` (deterministic, scalar), `f_filtered_states_mean` `(120, 1)`, `f_filtered_states_chol_cov` `(120, 1, 1)`, `f_filtered_states_cov` `(120, 1, 1)`, `f_filtered_states_cov_diag` `(120, 1)`.

Rollout under `Predictive` with 300 posterior draws and `predict_times` of length 25: the filter sites with a leading `300`, plus `f_120_x_0` `(300, 1, 1)`, `f_predicted_observations` `(300, 1, 25, 1)`, `f_predicted_states` `(300, 1, 25, 1)`, `f_predicted_times` `(300, 1, 25)`; `flatten_draws` gives `(300, 25, 1)`. Accessing `.predicted_observations` on the object returned by `dsx.sample` inside the model returned `None` (it is the `Filter`'s result); the inner-trace read returned the arrays above.

### A.3 nf drivers on the block

First-revision block (`Filter` only), NUTS 300 plus 300 draws: 26 s including compilation; posterior keys `q`, `r`, `f_marginal_loglik`, `f_filtered_states_*`. `forecast(key, model, posterior, data, covariates)`: shape `(300, 24, 1)` in 0.73 s; CRPS 0.476 on the held-out window. `forecast(..., batch_size=100, parallel=False)`: shape `(300, 24, 1)`. `backtest(..., min_train_window=100, test_window=12, stride=16, num_samples=200)` with a NUTS closure: three windows `(0, 100, 112)`, `(0, 116, 128)`, `(0, 132, 144)` with CRPS 0.580, 0.497, 0.498 in 83 s total.

### A.4 Covariates, SVI, and agreement with the direct model

Local level plus a Fourier regression (period 12, 2 harmonics, 4 covariates), $T = 120$, NUTS 2 chains of 500 plus 500. dsx model (`DynamicalModel` with `control_dim=4`, `D=beta[None, :]`, KF cuthbert): 40.8 s, ESS $q$ 1 033, $r$ 1 153, posterior means $q = 0.310$, $r = 0.483$, $\beta = (1.35, -0.10, -0.01, -0.18)$, CRPS 0.518. Direct nf model with `LocScaleReparam`: 4.7 s, ESS $q$ 351, $r$ 548, means $q = 0.308$, $r = 0.484$, $\beta = (1.35, -0.09, 0.00, -0.18)$, CRPS 0.521. Truth $q = 0.3$, $r = 0.5$. On dynestyx 0.5.0, `LTI_discrete(..., D=beta[None, :])` without `B` raised `Controls are provided (shape: (120, 4)), but dynamics.control_dim is 0`; on the dynestyx#361 branch the same call builds a model with `control_dim = 4`.

SVI on the dsx model (`AutoNormal`, `Adam(0.01)`, 2 000 steps): 3.0 s; `draw_posterior` returned `beta`, `q`, `r` and the filter's deterministic sites; `forecast` from that posterior: shape `(500, 24, 1)`, CRPS 0.549.

### A.5 Backend cost

Local level, $T = 120$, NUTS with 2 chains of 500 plus 500: `KFConfig()` (cd_dynamax) 4.8 s, ESS $q$ 591, $r$ 574, 4 440 leapfrog steps; `KFConfig(filter_source="cuthbert")` 38.7 s, ESS 555 and 582, 4 506 steps; nf direct without reparameterization 3.8 s, ESS 76 and 327, 236 568 steps. Per gradient: nf direct 0.01 ms, cd_dynamax KF 0.17 ms, cuthbert associative KF 1.26 ms, cuthbert sequential KF 2.52 ms. The log likelihood at $q = 0.3$, $r = 0.5$ was $-124.36$ for all three KF backends (the direct model's value, $-1015.84$, is a joint density over the path and is not comparable).

### A.6 Time grids under the jitted driver

The `Filter` block with the grids built three ways, forecast with `forecast()` (20 draws, 10 steps): `np.arange` succeeded with shape `(20, 10, 1)`; `jnp.asarray(np.arange(...))` and `jnp.arange` both failed inside the simulator's host-side segment bookkeeping with `TracerArrayConversionError`. The converse holds for `dsx.simulate`, the pure-JAX generator used for data simulation and prior predictive checks: it indexes the time grid inside a `lax.scan` and therefore needs a `jax.Array` (`jnp.asarray` of the NumPy grid), while a NumPy grid raises the same error there.

### A.7 The `Smoother` conditioner through every driver

Section 6.2 block, local level with the series in `covariates`, `Smoother(KFSmootherConfig(filter_source="cd_dynamax"))`, NUTS 2 chains of 500 plus 500: 4.1 s, ESS $q$ 479, $r$ 459, 4 822 leapfrog steps, posterior means 0.319 and 0.491, identical to the `Filter` run under the same key (5.0 s, same ESS and steps). `forecast`: `(1000, 24, 1)`, variance of the first three rows 0.467, 0.585, 0.660. `predict_in_sample`: `(1000, 120, 1)`, predictive standard deviation 0.593 at the first step and 0.581 mid-window. `to_datatree`: groups `posterior`, `posterior_predictive`, `observed_data`, `constant_data`, `predictions`, `predictions_constant_data`. The `Filter` conditioner raised the intended `ValueError` from `predict_in_sample`.

### A.8 The `LatentPathBuilder` conditioner

Same data and block, one `LatentPathBuilder()` instance created outside the model. Training trace: `q`, `r`, `f_state_path_params` (sample, `(120, 1)`), `f_joint_log_prob_factor` (sample, `(0,)`), `f_state_path_param_times` `(120,)`, `f_state_path` `(120, 1)`, `f_state_path_times` `(120,)`, `f_joint_log_prob` (scalar). NUTS 2 chains of 500 plus 500: 5.1 s, ESS $q$ 140, $r$ 368, 21 992 leapfrog steps, means 0.314 and 0.496. `forecast`: `(1000, 24, 1)`, variance of the first three rows 0.478, 0.557, 0.697. `predict_in_sample`: `(1000, 120, 1)`; `to_datatree`: all six groups. With NumPy grids the builder's forward sampler failed inside its `lax.scan` (`TracerArrayConversionError` on `times[t_idx]`); with jax grids and a builder instance created inside the model the fit succeeded but `forecast()` failed with dsx's `needs a fixed observation missingness pattern, but cannot infer one from traced obs_values`; with jax grids and the shared instance everything passed.

### A.9 Exact observations: AR(1) under `DiracIdentityObservation`

AR(1) with $\phi = 0.7$, $\sigma = 0.5$, $T = 120$, the Section 6.3 `ar1_dynamics` under a shared `LatentPathBuilder()`. Training trace: `f_state_path_params` of shape `(0,)` (no latent), `f_joint_log_prob_factor` `(0,)`, `f_state_path` `(120, 1)`, `f_completed_obs_values` `(120, 1)`. NUTS 500 plus 500: 1.6 s, posterior means $\phi = 0.772$, $\sigma = 0.494$. `forecast`: `(500, 24, 1)` with first-row mean $-0.872$ against $\hat{\phi}\, y_T = -0.848$. `predict_in_sample`: `(500, 120, 1)`, equal to the data. A scalar state (`dist.Normal` without `.expand([1]).to_event(1)`) fitted but produced a rollout of shape `(500, 24, 120)`: the state must be a length-1 vector when `y` carries an observation axis.

### A.10 Irregular times, `eval_train`, smoother gradient cost

`Filter` block with `times` drawn as a cumulative sum of uniform $[0.5, 1.5]$ increments over 144 steps: NUTS 200 plus 200 and `forecast` of shape `(200, 24, 1)`, all finite. `backtest` on the `Smoother` model with NUTS closures for both `forecast_fn` and `in_sample_fn`, `eval_train=True`, `min_train_window=100`, `test_window=12`, `stride=22`: two folds at split points 100 and 122 with out-of-sample CRPS 0.618 and 0.353 and in-sample CRPS 0.233 and 0.238. Per gradient of the log density: `Filter(KFConfig())` 0.17 ms, `Smoother(KFSmootherConfig(filter_source="cd_dynamax"))` 0.17 ms, both at log likelihood $-124.36$.

### A.11 Continuous time on irregular observations

Ornstein-Uhlenbeck level $dx = -\theta x \, dt + \sigma \, dW$, $y_t = x_t + v_t$, $\theta = 0.15$, $\sigma = 0.6$, $r = 0.3$, simulated with `dsx.simulate` on 144 times whose gaps are uniform in $[0.3, 2.5]$; $T = 120$ observed, 24 held out; `times` passed to the block. Under `Smoother(ContinuousTimeKFSmootherConfig())`: NUTS 300 plus 300 in 521 s, posterior means $\theta = 0.229$, $\sigma = 0.612$, $r = 0.164$; `forecast` `(300, 24, 1)` in 5.2 s with first-row variances 0.366, 0.590, 0.786, 0.853 for gaps 1.06, 1.25, 2.43, 1.54; `predict_in_sample` `(300, 120, 1)`; `to_datatree(..., time_coord=times)` rejected the NumPy array (`Sequence[Any]`) and accepted `times.tolist()`. Under `(Smoother(KFSmootherConfig(filter_source="cuthbert")), Discretizer(ExactAffineConfig(covariance_jitter=1e-6)))`: log likelihood at the true parameters $-117.782$ against $-117.782$ for the continuous-discrete smoother; per gradient 1.4 to 1.9 ms against 78 to 96 ms; with `KFSmootherConfig(filter_source="cd_dynamax")` the discretized model raised `requires constant (array-valued) parameters. Received callable field(s): A, cov`. NUTS on the discretized model: with dsx's default error handling it aborted with `Transition covariance is singular or indefinite` (no jitter) and then `Discretization produced a non-finite transition covariance` (with jitter, with a `HalfNormal` or a `LogNormal` prior on $\theta$, in `float32` or `float64`), although the log density and its gradient were finite at every point of a grid with $\theta \in [10^{-4}, 100]$ and $\sigma \in [10^{-3}, 10]$; with `EQX_ON_ERROR=nan`, the jitter and the `LogNormal` prior it ran 300 plus 300 iterations in 28 s to $\theta = 0.173$, $\sigma = 0.666$, $r = 0.173$, forecast `(300, 24, 1)` with variances 0.532, 0.748, 1.115, 1.088 for the same gaps, `predict_in_sample` `(300, 120, 1)`. Without the jitter, or in `float32`, a 2-chain run of the same model made no progress in 10 minutes: the NaN densities are counted as divergences, the step size collapses and every iteration runs to the maximum tree depth.

### A.12 Handler stacks

The Section 6.2 block with `conditioner` given as a sequence: `(Discretizer(), Smoother())` raised `the Filter, Smoother or LatentPathBuilder must come first (outermost) in conditioner`; `(Filter(), Smoother())` and `()` raised `conditioner needs exactly one Filter, Smoother or LatentPathBuilder`. Single handlers behaved as in A.7 and A.8 with the sites registered by the block: `Filter` `forecast` `(200, 24, 1)` and the in-sample `ValueError`; `Smoother` and `LatentPathBuilder` `forecast` `(200, 24, 1)` and `predict_in_sample` `(200, 120, 1)`.
