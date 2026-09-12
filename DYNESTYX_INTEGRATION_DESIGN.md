# Design: integrating `numpyro_forecast` with `dynestyx`

Status: design with a working example. Every claim about `dynestyx` below was verified against `dynestyx==0.5.0` installed next to the current `numpyro_forecast` pins (`jax==0.11.0`, `numpyro==0.21.0`, `arviz==1.3.0`) with the throwaway probes summarized in Appendix A. The example notebook that implements this design is `docs/examples/dynestyx_integration.ipynb` (Section 14).

Tracking: [juanitorduz/numpyro_forecast#34](https://github.com/juanitorduz/numpyro_forecast/issues/34) and [BasisResearch/dynestyx#264](https://github.com/BasisResearch/dynestyx/issues/264).

## 1. Summary

A `dynestyx` state space model is written **as a `numpyro_forecast` model function** `(covariates, data=None) -> None`: the model derives its `Horizon` from the shapes, samples its parameters with `numpyro.sample`, builds a `dynestyx.DynamicalModel`, and hands both to one new model building block, `filtered_series(h, name, dynamics, ...)`. While training the block runs `dsx.sample` under a `Filter`, which adds the marginal log likelihood $\log p(y_{1:T} \mid \theta)$ as a NumPyro factor; while forecasting it nests the same `Filter` inside a `Simulator`, which rolls the filtered state forward over the horizon, and it returns the horizon draws so the model registers them as the `"forecast"` deterministic site. Nothing else changes: `SVI`, `MCMC`, `forecast`, `backtest`, the metrics and `predictions_to_datatree` all work on the model unchanged, because the model honors the same site contract as `innovations`, `markov_series` and `ssoe`.

The block is the entire integration. It has no class hierarchy, no adapter objects and no second driver stack; it is the same shape as the existing building blocks (a plain function that calls NumPyro primitives against a `Horizon`), and it reuses `dynestyx`'s documented handler composition rather than its internals. It lives in the example notebook first and is proposed for `numpyro_forecast/contrib/dynestyx.py` once the maintainers of both libraries have reviewed it (Section 13).

What the user gains: the latent path is marginalized by a filter (exactly, with the Kalman filter, for linear-Gaussian models; approximately, with the EnKF/EKF/UKF/particle filter, beyond that), so NUTS and SVI see only the parameters. On the local level model with $T = 120$ the marginalized NUTS run drew 591 effective samples of the state noise scale in 4.8 s against 76 in 3.8 s for the direct model, using 53 times fewer gradient evaluations (Section 11).

## 2. Context

`numpyro_forecast` (nf below) is a functional port of the ideas in `pyro.contrib.forecast`: a model is a plain NumPyro function built from model building blocks (`Horizon.from_data`, `innovations`, `markov_series`, `ssoe`, `predict`) that register the `"obs"` site over the training window and the `"forecast"` site over the horizon; inference is whatever NumPyro or BlackJAX code the user writes; the drivers (`forecast`, `predict_in_sample`, `to_datatree`, `backtest`, `backtest_vectorized`) and the probabilistic metrics (`eval_crps`, `eval_coverage`, ...) read those sites by name. Its latent time series are sampled directly: a random walk is the cumulative sum of `T` sampled innovations, so a fit infers `T` per-step latents plus the parameters.

`dynestyx` (dsx below) is a probabilistic programming layer for dynamical systems on top of NumPyro. A `DynamicalModel` bundles an initial condition, a state evolution (discrete-time transition, or a continuous-time SDE/ODE) and an observation model; `dsx.sample(name, dynamics, obs_times=, obs_values=, predict_times=)` is the single primitive, and effect handlers give it its meaning: `Filter` and `Smoother` marginalize the latent path and register the marginal log likelihood as a NumPyro factor, `LatentPathBuilder` samples the path explicitly, and the simulators (`Simulator`, `DiscreteTimeSimulator`, `ODESimulator`, `SDESimulator`) generate trajectories, including posterior rollouts when they wrap a `Filter`. An `Evaluation` handler scores the one-step-ahead predictive observations of a filter with proper scoring rules.

In dynestyx#264 the dsx maintainers suggested two directions: (1) use nf to evaluate a dsx model, and (2) use dsx to fit nf-style models, the benefit being that "your model would be running a non-linear filtering algorithm to connect the noisy data to the underlying model". This document turns both into one mechanism. Direction (2) is the block itself. Direction (1) follows for free: once a dsx model is an nf model, `backtest` and the metrics score its multi-step forecasts on rolling windows, complementing dsx's own in-sample one-step-ahead `Evaluation`.

An earlier draft of this design (branch `dynestyx`) targeted nf's pre-refactor class API (`ForecastingModel`, `Forecaster`, `HMCForecaster`) and was blocked by an ArviZ version conflict. Both are gone: nf is now functional (PR #81) and dynestyx v0.5.0 migrated to ArviZ $\geq 1.0$ (dynestyx#346). That draft's adapter objects (`DsxFilterFit`, `DynestyxForecaster`) are superseded by the block; nothing from it is kept.

## 3. Where the two libraries meet

Both libraries are NumPyro effect-handler code, so they compose at the trace level. The seam is small and can be stated exactly:

- nf owns the **horizon bookkeeping** (`Horizon`: `t_obs`, `future`, `duration`), the **site contract** (`"obs"` and `"forecast"`, plus the `_future` suffix convention for sampled latents), the **drivers** (jitted, chunked, device-aware `Predictive` wrappers) and the **evaluation workflow** (rolling windows, CRPS, coverage, MASE, ArviZ export).
- dsx owns the **dynamics** (`DynamicalModel`), the **conditioning** (`Filter`: marginal likelihood plus filtered distributions) and the **rollout** (a `Simulator` outside the `Filter` starts every prediction segment from the filtered distribution at the last observation time that precedes it).
- The block translates between them: integer time steps become float `obs_times`/`predict_times`, `h.data` becomes `obs_values`, `covariates` become `ctrl_values`, and the simulator's `{name}_predicted_observations` site becomes the array the model registers as `"forecast"`.

Three properties make the composition sound rather than merely possible:

1. The `Filter` registers the marginal likelihood as `numpyro.factor`, so the model's joint density is $p(\theta) \, p(y_{1:T} \mid \theta)$ with the path integrated out. `SVI` with any autoguide, `MCMC` with any kernel, and the BlackJAX kernels in `numpyro_forecast.contrib.blackjax` all consume that density through `initialize_model`, unchanged.
2. Fitting always happens with `future == 0` (nf's invariant), so the simulator and its `_predicted_*` sites never appear in the guide or in the MCMC trace: the forecast branch of the block is only ever executed under `Predictive`. This is exactly the role the `_future` suffix plays for `innovations` and `markov_series`.
3. Under `Predictive(posterior_samples=...)` the model re-runs the filter for every posterior draw and then rolls the filtered state forward, so the forecast is conditioned on the data through the state, with the draw's own parameters, which is the correct posterior predictive $\int p(y_{T+1:T+H} \mid x_T, \theta) \, p(x_T \mid y_{1:T}, \theta) \, p(\theta \mid y_{1:T}) \, dx_T \, d\theta$.

## 4. Design principles

- **One model, two interpretations.** A model function is written once; the `Horizon` (and only the `Horizon`) decides whether the block conditions or forecasts. This is nf's existing rule and it matches dsx's own "separation of concerns" (a `DynamicalModel` has no notion of `predict`; handlers interpret `dsx.sample`).
- **Pure functions, explicit randomness.** The block is a plain function of `(h, name, dynamics, ...)` with no state; randomness comes from the enclosing NumPyro `seed` handler exactly as for `numpyro.sample`, so `Predictive`, `MCMC` and `SVI` control every key. No global state, no cached fits.
- **Static shapes, host-side time grids.** `t_obs`, `future` and `duration` are Python integers derived from array shapes, so the time grids are built with NumPy (`np.arange`) and are constants inside `jax.jit`. This is what keeps dsx's rollout, which does its segment bookkeeping on the host with `np.searchsorted`, safe inside nf's jitted `_predict` driver (verified, Appendix A.3). Building the grids with `jnp.arange` inside the model would trace them and break the rollout under `jit`.
- **Compile once, vectorize over draws.** nf's `forecast()` jits one `Predictive` per `(model, shape)` and vectorizes the sample axis with `vmap`. The Kalman filter (a `lax.scan`, or cuthbert's associative scan) and the discrete simulator (a `lax.scan`) are both `vmap`-friendly, so the whole forecast for 300 draws compiles and runs in 0.7 s on CPU (Appendix A.3). `batch_size` chunking, `parallel=False` and `device="host"` keep working because the driver does not know that dsx is inside.
- **Reuse documented dsx composition, not internals.** The block uses `Simulator` outside `Filter` with `predict_times`, which is the posterior-rollout composition dsx documents, and reads the simulator's site through `numpyro.handlers.trace`, a plain NumPyro idiom (`ssoe` already uses an inner trace). It never touches `ConditionedResult.dists`, `eqx.tree_at` on a `DynamicalModel`, or private helpers; the pure-JAX alternative that does is discussed as Approach C.
- **No new inference code.** Fitting is `svi.run` or `mcmc.run` on the model. The block never wraps either.

## 5. Approaches considered

### A. A model building block inside nf's model contract (recommended)

The dsx model is an nf `ForecastModel`; `filtered_series` is the only new function. Every nf driver, closure and metric works with zero adapter code, `backtest`'s `model_fn`/`forecast_fn` contract is untouched, and a user who already has an nf model swaps one block for another. The cost is that two nf drivers do not apply to a marginalized model (Section 12): `predict_in_sample` and `to_datatree`, which call the model with `data=None`.

### B. Adapter functions around a native dsx model

Keep the dsx model in its native signature `(obs_times, obs_values, predict_times)` and write `fit_dsx_filter`, `dsx_forecast` and a `DynestyxForecaster` object that satisfy `backtest`'s closures (the earlier draft). This reimplements what `forecast()` already does (chunking, device placement, compile caching) with a second `Predictive` call site, introduces a second model contract next to `ForecastModel`, and cannot be swapped into an existing nf model. It is only preferable when the same dsx model must also run in pure dsx workflows, and even then the `DynamicalModel` construction can be shared as a helper function while the two thin model functions stay separate. Rejected.

### C. Pure-JAX rollout without handler nesting

Condition with `dsx.condition` under a `Filter` to obtain a `ConditionedResult` (marginal likelihood, filtered `dists` and `times`), register the factor by hand, then build the anchored dynamics with `eqx.tree_at` (initial condition set to `dists[-1]`, `t0` to `times[-1]`) and call `DiscreteTimeSimulator(n_simulations=1).simulate(anchored, rng_key=numpyro.prng_key(), predict_times=...)`. This is the most explicit functional form: the key is passed by hand, no trace is read, no host-side segment search runs, and only the final segment is simulated. It depends on `ConditionedResult.dists`, on `DynamicalModel` being an Equinox module with mutable-by-`tree_at` fields, and on `numpyro.prng_key()`; none of these is documented as the way to roll out a posterior. Not exercised in the probes. It is the natural internal implementation of Approach A's forecast branch if the dsx maintainers endorse it, with the block's public contract unchanged.

Decision: A, with the rollout implemented through the documented `Simulator` ∘ `Filter` composition, and C recorded as the candidate optimization.

## 6. The building block: `filtered_series`

### 6.1 Contract

```python
filtered_series(
    h: Horizon,
    name: str,
    dynamics: DynamicalModel,
    *,
    filter_config: BaseFilterConfig | None = None,
    controls: Float[Array, " duration control"] | None = None,
    dt: float = 1.0,
    simulator_config: SimulatorConfig | None = None,
) -> FilteredSeriesResult
```

- `h` is the current call's `Horizon`; `h.data` must be present (the block raises otherwise, see below); `h.data` has shape `(t_obs, obs)` with time at axis `-2` and is passed to dsx unchanged (dsx accepts `(T, D)` observations).
- `name` is the dsx site prefix. The filter registers `{name}_marginal_log_likelihood` (the factor), `{name}_marginal_loglik` (a deterministic scalar) and the `{name}_filtered_states_*` deterministics selected by `filter_config`; the simulator registers `{name}_predicted_times`, `{name}_predicted_states`, `{name}_predicted_observations` and one `{name}_{j}_x_0` per rollout segment. All of these are dsx's names; the block adds none of its own.
- `dynamics` is any `DynamicalModel` the chosen filter supports. The observation dimension must equal `h.data.shape[-1]`.
- `filter_config` selects the filter (`KFConfig()` for linear-Gaussian models, `EnKFConfig`/`EKFConfig`/`UKFConfig` for nonlinear Gaussian ones, `PFConfig` for non-Gaussian observations, the `ContinuousTime*` variants for SDE/ODE dynamics). `None` takes dsx's default (`EnKFConfig()` for discrete-time models), which is approximate: pass `KFConfig()` explicitly for linear-Gaussian models.
- `controls` are the exogenous inputs over the full horizon, shape `(duration, control)`, normally the model's `covariates` themselves. They become `ctrl_values` on the grid `ctrl_times = arange(duration) * dt`, which covers both the observation times and the prediction times, as the simulator requires. The dynamics consume them through `B`/`D` (linear-Gaussian) or the `u` argument of a callable transition/observation.
- `dt` maps integer steps to dsx float times (`obs_times = arange(t_obs) * dt`). It is irrelevant for discrete-time dynamics and is the sampling interval for continuous-time ones.
- `simulator_config` is forwarded to `Simulator`, which auto-selects the discrete, ODE or SDE backend from the dynamics; the discrete backend has no configuration.

Behavior:

- Training (`h.future == 0`): run `dsx.sample` under `Filter(filter_config)`; the factor is added to the trace; return a result whose time axes have size 0.
- Forecasting (`h.future > 0`): run `dsx.sample` under `Simulator(n_simulations=1)` outside `Filter(filter_config)` with `predict_times = arange(t_obs - 1, duration) * dt`, read `{name}_predicted_observations` and `{name}_predicted_states` from an inner trace, drop the simulation axis and the anchor row, and return `y_future` of shape `(future, obs)` and `x_future` of shape `(future, state)`.
- `h.data is None` (prior sampling, or the `predict_in_sample`/`to_datatree` drivers): raise `ValueError`. A marginalized model has no path to sample from without observations. Prior predictive checks use `dsx.simulate(dynamics, rng_key=..., predict_times=..., n_simulations=...)`, the pure-JAX generator, directly.
- The block registers nothing but dsx's sites. As with `ssoe`, the caller writes `numpyro.deterministic("forecast", result.y_future)` when `h.future > 0`; an unconditional registration is harmless to `forecast()` but lands a size-0 variable in every posterior.

### 6.2 Reference implementation

This is the code the notebook defines and the code proposed for `contrib/dynestyx.py`; the two are identical, docstring included, so the block graduates without edits.

```python
from dataclasses import dataclass
from typing import Any

import dynestyx as dsx
import jax.numpy as jnp
import numpy as np
import numpyro
from dynestyx import DynamicalModel, Filter, Simulator
from dynestyx.inference.configs.filter import BaseFilterConfig
from dynestyx.inference.configs.simulator import SimulatorConfig
from jaxtyping import Float

from numpyro_forecast import Horizon
from numpyro_forecast.typing import Array


@dataclass(frozen=True)
class FilteredSeriesResult:
    """Horizon draws produced by `filtered_series` (size-0 time axes while training).

    Attributes
    ----------
    y_future
        Observation draws over the horizon, shape ``(future, obs)``.
    x_future
        Latent state draws over the horizon, shape ``(future, state)``.
    """

    y_future: Float[Array, " future obs"]
    x_future: Float[Array, " future state"]


def _time_grid(size: int, dt: float) -> np.ndarray:
    """Host-side float time grid ``[0, dt, ..., (size - 1) dt]`` (a constant under jit)."""
    return (np.arange(size) * dt).astype(np.float32)


def filtered_series(
    h: Horizon,
    name: str,
    dynamics: DynamicalModel,
    *,
    filter_config: BaseFilterConfig | None = None,
    controls: Float[Array, " duration control"] | None = None,
    dt: float = 1.0,
    simulator_config: SimulatorConfig | None = None,
) -> FilteredSeriesResult:
    """Condition a dynestyx model on the observed window and roll it over the horizon.

    In-sample the block runs ``dsx.sample`` under ``Filter``, which adds the
    marginal log likelihood of the observed window as a NumPyro factor; when
    forecasting it nests the same ``Filter`` inside a ``Simulator`` whose
    posterior rollout starts from the filtered state at the last observed step
    and returns the horizon draws. The guide never sees the rollout because
    fitting always happens with ``h.future == 0``.

    Parameters
    ----------
    h
        The horizon for the current model call; ``h.data`` must be present.
    name
        Prefix of the ``dynestyx`` sites (``{name}_marginal_loglik``,
        ``{name}_filtered_states_mean``, ``{name}_predicted_observations``, ...).
    dynamics
        The ``dynestyx`` model; its observation dimension must match ``h.data``.
    filter_config
        The filter to condition with (``KFConfig()`` for linear-Gaussian
        models). ``None`` takes the ``dynestyx`` default, an ensemble Kalman
        filter, which is approximate.
    controls
        Exogenous inputs over the full horizon, shape ``(duration, control)``,
        normally the model's ``covariates``; forwarded as ``ctrl_values`` on a
        grid covering the observation and prediction times.
    dt
        Spacing that maps integer steps to ``dynestyx`` float times.
    simulator_config
        Forwarded to ``Simulator`` (solver options for continuous-time models).

    Returns
    -------
    FilteredSeriesResult
        The horizon draws ``y_future`` and ``x_future``.

    Raises
    ------
    ValueError
        If ``h.data`` is ``None``.
    """
    if h.data is None:
        msg = (
            "filtered_series needs observed data: a marginalized model has no latent "
            "path to sample without observations. Prior predictive checks go through "
            "dynestyx.simulate; predict_in_sample and to_datatree are not applicable."
        )
        raise ValueError(msg)
    # NumPy grids on purpose: host-side constants under jit (dynestyx annotates them
    # as jax Arrays but accepts array-likes, hence the untyped keyword dict).
    grids: dict[str, Any] = {"obs_times": _time_grid(h.t_obs, dt)}
    if controls is not None:
        grids |= {"ctrl_times": _time_grid(h.duration, dt), "ctrl_values": controls}
    if h.future == 0:
        with Filter(filter_config=filter_config):
            dsx.sample(name, dynamics, obs_values=h.data, **grids)
        return FilteredSeriesResult(
            y_future=jnp.zeros((0, dynamics.observation_dim)),
            x_future=jnp.zeros((0, dynamics.state_dim)),
        )
    # Anchor at the last observed step: the simulator's first predicted state is the
    # filtered draw at predict_times[0] with no transition, so that row is dropped.
    grids["predict_times"] = _time_grid(h.duration, dt)[h.t_obs - 1 :]
    simulator = Simulator(simulator_config, n_simulations=1)
    with numpyro.handlers.trace() as tr, simulator, Filter(filter_config=filter_config):
        dsx.sample(name, dynamics, obs_values=h.data, **grids)
    y = tr[f"{name}_predicted_observations"]["value"]  # (n_sim, future + 1, obs)
    x = tr[f"{name}_predicted_states"]["value"]  # (n_sim, future + 1, state)
    return FilteredSeriesResult(y_future=y[0, 1:, :], x_future=x[0, 1:, :])
```

Why each non-obvious line is there:

- `np.arange(...)` rather than `jnp.arange(...)`: see the static-shapes principle in Section 4 and Appendix A.3. Wrapping the NumPy grid in `jnp.asarray` does not help: under the jitted driver that call is staged too and the rollout fails with the same `TracerArrayConversionError` (Appendix A.6). The grids are passed through an untyped keyword dict because `dynestyx` annotates them as `jax.Array` while accepting array-likes; `ty` type-checks the notebooks.
- The inner `numpyro.handlers.trace()`: `dsx.sample` returns the innermost handler's result, which is the `Filter`'s `ConditionedResult`; the simulator's rollout reaches the model only through the sites it registers. An inner trace records those sites and lets them continue up the handler stack, so the outer `Predictive` trace still sees them (Appendix A.2).
- `predict_times` starting at `t_obs - 1`: the anchor semantics were measured, not assumed. Without the anchor, the first forecast row reproduces the filtered state at `t_obs - 1` instead of stepping to `t_obs`, which is an off-by-one that the marginal variances expose (Appendix A.1).
- A fresh `Simulator` per call: `Simulator` caches the concrete backend it resolves on its instance; a fresh instance per model execution keeps the block free of shared mutable state under `vmap`.
- `y[0, 1:, :]`: drop the `n_simulations` axis (always 1; the posterior sample axis is nf's) and the anchor row.

### 6.3 A model built on the block

```python
def local_level(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    q = numpyro.sample("q", dist.HalfNormal(1.0))  # state noise scale
    r = numpyro.sample("r", dist.HalfNormal(1.0))  # observation noise scale
    dynamics = dsx.LTI_discrete(
        A=jnp.eye(1),
        Q=jnp.eye(1) * q**2,
        H=jnp.eye(1),
        R=jnp.eye(1) * r**2,
        initial_mean=jnp.zeros(1),
        initial_cov=jnp.eye(1) * 100.0,
    )
    result = filtered_series(h, "f", dynamics, filter_config=KFConfig())
    if h.future > 0:
        numpyro.deterministic("forecast", result.y_future)
```

This is the local level model $x_t = x_{t-1} + w_t$, $w_t \sim \text{Normal}(0, q)$, $y_t = x_t + v_t$, $v_t \sim \text{Normal}(0, r)$, with $q$ and $r$ standard deviations squared into dsx's covariance arrays. The same generative process in nf's direct form is `innovations` plus `jnp.cumsum` plus `predict`, and the two fits agree (Appendix A.4).

With covariates in the observation equation, $y_t = H x_t + D u_t + v_t$:

```python
def local_level_regression(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    num_features = covariates.shape[-1]
    q = numpyro.sample("q", dist.HalfNormal(1.0))
    r = numpyro.sample("r", dist.HalfNormal(1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([num_features]).to_event(1))
    dynamics = dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1) * 100.0),
        state_evolution=dsx.LinearGaussianStateEvolution(A=jnp.eye(1), cov=jnp.eye(1) * q**2),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.eye(1), R=jnp.eye(1) * r**2, D=beta[None, :]
        ),
        control_dim=num_features,
    )
    result = filtered_series(
        h, "f", dynamics, filter_config=KFConfig(), controls=covariates
    )
    if h.future > 0:
        numpyro.deterministic("forecast", result.y_future)
```

`LTI_discrete` infers `control_dim` from `B` alone, so a model whose controls enter only through `D` must use the explicit `DynamicalModel` constructor with `control_dim` (Appendix A.4; also Section 17).

## 7. Inference and drivers matrix

| Component | Works on a `filtered_series` model | Evidence |
| --- | --- | --- |
| `MCMC(NUTS(model))`, 1 or more chains | yes; the posterior holds the parameters plus dsx's deterministic sites | A.3, A.4, A.5 |
| `SVI(model, AutoNormal(model), ...)` + `draw_posterior` | yes; `Trace_ELBO` includes the factor, the guide covers only the parameters | A.4 |
| `numpyro_forecast.contrib.blackjax` kernels | expected (they consume the same `initialize_model` density); not exercised | none |
| `forecast(key, model, posterior, data, covariates)` | yes, including `batch_size`, `parallel=False`, `device` | A.3 |
| `backtest(..., model_fn=lambda: model, forecast_fn=...)` with a NUTS or SVI closure | yes | A.3 |
| `eval_crps`, `eval_coverage`, `eval_mae`, `eval_rmse`, `evaluate_forecast`, `metrics.*` | yes (they only see draws) | A.3 |
| `predictions_to_datatree(forecast_draws, ...)`, ArviZ plots and `az.summary` on the posterior | yes (posterior dict plus draws) | design |
| `predict_in_sample`, `to_datatree` | no: both call the model with `data=None` (Section 12) | design |
| `backtest_vectorized` (vmapped SVI over rolling windows) | plausible (pure JAX under `vmap`), not exercised | none |
| `dsx.plate` / batched panels, nf batch dims left of time | not in scope (Section 12) | none |

A posterior from `mcmc.get_samples()` or `draw_posterior` also carries dsx's deterministic sites (`f_filtered_states_mean`, `f_filtered_states_cov`, ..., `f_marginal_loglik`). Passing that dict straight to `forecast()` is fine: `Predictive` substitutes the recorded values into the trace, but the rollout reads the filter's in-memory result, so the draws are unaffected (verified by A.3 and A.4, which forecast from the full dict). They do cost memory proportional to `(draws, t_obs, state)`; set the `record_filtered_states_*` fields of the filter config to `False` when they are not wanted.

## 8. Covariates, time and windows

- nf's `covariates` array spans the full horizon and is the only channel through which anything reaches the model at prediction time. The block maps it to `ctrl_values` on a grid that covers observation and prediction times, which satisfies the simulator's rule that `ctrl_times` contain every prediction time exactly.
- Regression enters through `D` (observation) or `B` (state) of the linear-Gaussian classes, or through the `u` argument of a callable transition or observation model. Seasonality is either a regression on `fourier_features` covariates (`D`), or a seasonal state block in `A` with `H` selecting it (no covariates needed).
- Time-varying linear-Gaussian parameters (callables of `t`) are supported by `KFConfig(filter_source="cuthbert")` only.
- Time restarts at 0 in every `backtest` window because the window's `Horizon` restarts at 0. This is correct for time-invariant dynamics and for covariate-driven effects (the covariates are sliced with the window). Models with callable parameters that depend on absolute time would need a `t0` offset argument on the block; deferred until a use case appears.
- `dt` only matters for continuous-time dynamics; discrete transitions ignore the interval.

## 9. Mapping nf models to dsx dynamics

The gain from dsx is confined to models whose latent path nf **samples**. `ssoe` models (ARMA, exponential smoothing in innovations form, Croston/TSB) are already marginalized: their recursion is deterministic given the observed series and the parameters, they sample nothing per step, and a Kalman filter would recompute the same likelihood at a higher cost. The single-source-of-error form also has perfectly correlated state and observation noise, which `LTI_discrete` (independent `Q` and `R`) cannot express. Those models stay in nf as they are.

| nf idiom (sampled path) | dsx dynamics | Filter | Notes |
| --- | --- | --- | --- |
| Random walk level: `innovations` + `cumsum` | `LTI_discrete(A=I, Q=q^2, H=I, R=r^2)` | `KFConfig` (exact) | Example 1 |
| Local linear trend (level + slope) | `A=[[1, 1], [0, 1]]`, `H=[[1, 0]]`, diagonal `Q` | `KFConfig` (exact) | |
| Level + regression on covariates | `LinearGaussianObservation(D=beta[None, :])`, `control_dim=k` | `KFConfig` (exact) | Example 2 |
| Seasonal dummies / Fourier state | seasonal block in `A`, `H` selecting the current season | `KFConfig` (exact) | alternative to `D` |
| VAR(p) as an explicit latent (`markov_series` + `var_step`) | `A = numpyro_forecast.var.companion_matrix(coefs)`, `H` selecting the first lag block | `KFConfig` (exact) | nf already ships the companion matrix |
| `markov_series` with a nonlinear Gaussian transition | callable `state_evolution(x, u, t_now, t_next)` returning a Gaussian | `EnKFConfig`, `UKFConfig`, `EKFConfig` (approximate, pseudo-marginal) | |
| `markov_series` + count/heavy-tailed `predict` link | custom `ObservationModel` returning `Poisson`, `NegativeBinomial`, `StudentT` | `PFConfig` (approximate, pseudo-marginal) | the boundary of exactness |
| Continuous-time latent (irregular sampling) | `LTI_continuous`, `AffineDrift` + `Diffusion` | `ContinuousTimeKFConfig`, `ContinuousTimeEnKFConfig` | `dt` is the sampling interval |
| Missing observations | `NaN` in `obs_values` | `KFConfig(filter_source="cuthbert")` | nf models have no missing-data story of their own |

## 10. Pinned dynestyx 0.5.0 facts

Everything the block depends on, with where it was checked:

1. Installation: `uv pip install dynestyx==0.5.0` into the `--extra all` environment resolves without changing `jax`, `numpyro` or `arviz`; it adds about 60 packages, among them `cd-dynamax`, `cuthbert`, `effectful`, `equinox`, `diffrax`, `flax`, `orbax-checkpoint`, `tfp-nightly`, `mkdocs`.
2. `dsx.sample(name, dynamics, obs_times=, obs_values=)` under `Filter(KFConfig(...))` registers the sites `f_marginal_log_likelihood` (sample site of a `Unit` distribution, the factor), `f_marginal_loglik` (deterministic scalar) and, by default, `f_filtered_states_mean` `(T, state)`, `f_filtered_states_cov` `(T, state, state)`, `f_filtered_states_cov_diag` and `f_filtered_states_chol_cov`. `mcmc.get_samples()` returns the deterministic ones next to the parameters (A.2, A.3).
3. `Simulator(n_simulations=1)` (or `DiscreteTimeSimulator`) outside the `Filter`, with `predict_times`, registers `f_predicted_observations` `(n_sim, T_pred, obs)`, `f_predicted_states` `(n_sim, T_pred, state)`, `f_predicted_times` `(n_sim, T_pred)` and `f_{j}_x_0` per segment; `Predictive` prepends the draw axis; `dynestyx.flatten_draws` merges `(draws, n_sim)` (A.2).
4. `dsx.sample` returns the `Filter`'s `ConditionedResult` (fields `marginal_loglik`, `times`, `states`, `dists`, `predicted_observations`); its `predicted_observations` field holds the filter's one-step-ahead outputs, not the simulator's rollout (A.2).
5. Anchor semantics: the discrete simulator's first state is the initial-condition draw at `predict_times[0]` with no transition; in a posterior rollout that draw comes from the filtered distribution at the last observation time $\leq$ `predict_times[0]`. With `predict_times = [T, ...]` the first forecast row therefore has the variance of the filtered state at $T - 1$ plus observation noise, identical to the anchored row at $T - 1$ (A.1).
6. One-step-ahead predicted-observation outputs (`f_predicted_observations_mean`/`_cov` sites, and the `Evaluation` handler that scores them) are produced only by the continuous-time filters and by `EnKFConfig(filter_source="cuthbert")`; the discrete `KFConfig` exposes the filtered states instead.
7. `LTI_discrete(A, Q, H, R, B=None, b=None, D=None, d=None, initial_mean=None, initial_cov=None)` sets `control_dim = B.shape[-1]` when `B` is given and 0 otherwise; passing `ctrl_values` to a model with `control_dim == 0` raises. The explicit `DynamicalModel(..., control_dim=k)` constructor accepts `D`-only controls (A.4).
8. Filter backends for the Kalman filter: `KFConfig()` defaults to `filter_source="cd_dynamax"`; `"cuthbert"` adds missing-data (`NaN`) support, time-varying callable parameters and an associative (parallel-in-time) scan enabled by default. All three give the same log likelihood; their costs differ by an order of magnitude on CPU at $T = 120$ (A.5).
9. dsx reads randomness from `numpyro.prng_key()` inside the active `seed` handler (`BaseFilterConfig.crn_seed` pins it for stochastic filters); deterministic filters need none.

## 11. Performance

Local level model, $T = 120$, `MCMC(NUTS)` with 2 chains of 500 warmup plus 500 samples, Apple M4 Pro, CPU, `float32` (A.5):

| Model and backend | Wall time | ESS of $q$ | ESS of $r$ | Leapfrog steps |
| --- | --- | --- | --- | --- |
| dsx, `KFConfig()` (cd_dynamax) | 4.8 s | 591 | 574 | 4 440 |
| dsx, `KFConfig(filter_source="cuthbert")` (associative) | 38.7 s | 555 | 582 | 4 506 |
| nf direct (`innovations`, no reparameterization) | 3.8 s | 76 | 327 | 236 568 |

Per gradient evaluation of the log density (`jax.value_and_grad`, jitted, same model and data): nf direct 0.01 ms, cd_dynamax KF 0.17 ms, cuthbert associative KF 1.26 ms, cuthbert sequential KF 2.52 ms. Recording fewer filter outputs does not change the gradient cost.

Reading: the marginalized model needs 53 times fewer gradient evaluations for 8 times the effective sample size of the state noise scale, because NUTS moves in a 2-dimensional posterior instead of a 122-dimensional one with a funnel between $q$ and the path. On CPU at this length the per-gradient cost of the filter cancels most of the wall-clock advantage; the advantage grows with $T$ (the direct posterior grows with $T$, the filter's cost is linear in $T$ and does not affect the sampler's geometry) and on accelerators (where the associative scan pays off). `LocScaleReparam` on the direct model lifts its ESS to about 350 (A.4) and remains the right baseline for a fair comparison in the notebook.

Forecasting 300 draws over 24 steps with `forecast()` (jit plus `vmap`, including compilation): 0.7 s. `backtest` with a NUTS closure (200 warmup, 200 samples per window): about 27 s per window, dominated by recompilation of the fit per window shape.

Recommendations that follow: `KFConfig()` (cd_dynamax) by default on CPU; `filter_source="cuthbert"` when `NaN` observations or callable time-varying parameters are needed, or on a GPU with long series; use `LocScaleReparam` on the direct model in any comparison.

The committed notebook reproduces the picture with the fair baseline (`LocScaleReparam` on the direct model) and 4 chains of 1 000 plus 1 000 iterations: direct model about 13 s, ESS 876 for $q$ and 799 for $r$ on 1 019 459 leapfrog steps; marginalized model about 10.5 s, ESS 1 657 and 1 717 on 17 592 leapfrog steps; posterior means within 0.01 of each other; test CRPS 0.446 against 0.438.

## 12. Boundaries and non-goals

- **`predict_in_sample` and `to_datatree` do not apply.** Both call the model with `data=None`, which for nf's direct models means "replay the posterior latents and sample the likelihood". A marginalized model has no posterior latents to replay; its in-sample analog is the filtered (or smoothed) predictive, which needs the data. The block raises with a message that says so. What the user does instead: read `f_filtered_states_mean`/`f_filtered_states_cov` from the posterior dict (already recorded by the filter under `MCMC` and by `draw_posterior`), build the filtered observation predictive $H m_{t \mid t}$ with variance $H P_{t \mid t} H^\top + R$ by hand, and plot it; or use dsx's `Smoother` for the smoothed path. A follow-up (Section 17) considers an nf driver variant that passes `data` to the model for exactly this class of model.
- **Prior predictive checks** use `dsx.simulate` (pure JAX, no NumPyro) rather than `Predictive(model)(key, covariates)`.
- **Panels and hierarchy** (`dsx.plate`, nf batch dims to the left of time) are not addressed. dsx's plate is an effectful handler with its own batching contract for arrays and dynamics; mapping it onto nf's leftward batch dims is a separate design.
- **`backtest_vectorized`** is not exercised.
- **Nothing is built in the other direction** (nf building blocks inside a dsx model). nf's blocks register sites against a `Horizon`; dsx models are Equinox modules of dynamics. There is no natural composition and no demand for one.
- **No new inference machinery** (no fit helpers, no forecaster objects). `svi.run` and `mcmc.run` on the model are the whole story.

## 13. Packaging plan

### v1 (example notebook)

- `docs/examples/dynestyx_integration.ipynb`, authored with jupytext (`py:percent`), executed, committed with outputs, `.py` deleted. The block (`FilteredSeriesResult`, `filtered_series`) is defined in the notebook verbatim as in Section 6.2.
- `pyproject.toml`: a new optional extra `dynestyx = ["dynestyx>=0.5.0"]`, deliberately **not** aggregated into `all` or `all_cuda`. Reasons: the notebooks are not executed in CI (the docs build renders stored outputs and `pytest` collects only `tests/` and `README.md`), so no CI job needs dsx; the extra pulls about 60 packages including `tfp-nightly`, which would slow and destabilize every CI leg for no coverage. The extra documents the pin the notebook was executed against and makes `uv sync --extra all --extra dynestyx` the reproducible authoring command.
- `README.md`: the sentence listing the optional extras gains `dynestyx`.
- No change under `numpyro_forecast/`, no `reference:` change, `tests/test_docs_reference.py` unaffected.

### v2 (proposed, after review by both maintainers)

- `numpyro_forecast/contrib/dynestyx.py` with `filtered_series` and `FilteredSeriesResult`, lazily importing dsx through `numpyro_forecast.optional.require("dynestyx", extra="dynestyx")` on first call (the package import must stay free of dsx, matching the `base-import` CI leg's invariant and `contrib/blackjax.py`), an `_api_canary("dynestyx", ["sample", "Filter", "Simulator", "DynamicalModel", "simulate"])` tripwire, and signatures that never name dsx types (`dynamics: object`, `filter_config: object | None`, as `contrib/blackjax.py` does for blackjax, because the jaxtyping/beartype import hook resolves annotations at call time and a `TYPE_CHECKING`-only import would not resolve).
- `great-docs.yml`: `contrib.dynestyx.filtered_series` and `contrib.dynestyx.FilteredSeriesResult` under "Extensions (contrib)".
- `tests/test_contrib_dynestyx.py` guarded by `pytest.importorskip("dynestyx")`: the anchor invariant (first horizon row has the filtered variance plus one transition), the `data is None` error, `forecast()` shapes under `batch_size`/`parallel=False`, agreement of the KF marginal likelihood with a direct `MultivariateNormal` log density on a tiny series, and one `backtest` window. Whether CI installs the extra for these (a separate job) or they stay skip-by-default is an open question (Section 17).
- The notebook then imports the block from the package and drops its local definition.

## 14. Example scaffolding

One notebook, two examples, both synthetic so the truth is known and the exactness claims can be checked.

**Example 1: local level, one process, two inference strategies.** Simulate $y_t$ from the local level model with known $q$ and $r$ ($T = 120$ training steps, 24 held out). Fit the direct nf model (`innovations` + `cumsum` + `predict`, `LocScaleReparam`) and the dsx model of Section 6.3 with the same priors and the same NUTS budget. Compare: posterior of $q$, $r$ against truth (forest plot with reference lines), ESS and wall time, the forecast fan charts ($50\%$ and $94\%$ HDI) overlaid, CRPS and coverage on the held-out window. Show the filtered level from `f_filtered_states_mean` with its HDI band as the in-sample view. Prior predictive check with `dsx.simulate`.

**Example 2: local level with seasonal regression, covariates through the block.** Add a Fourier seasonal component to the simulated series and pass `fourier_features` as `covariates`, which the block forwards as `ctrl_values` and the dynamics consume through `D`. Fit with SVI (`AutoNormal`, `draw_posterior`) to show the variational path, forecast, and run `backtest` on rolling windows with a NUTS `forecast_fn`, plotting CRPS and coverage per window. This is Direction (1) of dynestyx#264 made concrete: nf's evaluation workflow applied to a dsx model.

Notebook outline (jupytext `py:percent`; conventions per `AGENTS.md`: `description` metadata, `thumbnail` cell tag on the forecast overlay figure, `$94\%$ HDI` in LaTeX, `\text{Normal}`, no em-dashes, no `plt.show()`, no RST markup in prose):

1. Title and motivation: the two issues, what each library is, the seam, what the reader will see.
2. Setup: imports, `az.style`, `rng_key`, versions printed.
3. The block: `FilteredSeriesResult`, `filtered_series`, a short explanation of the anchor and of the NumPy time grids.
4. Example 1 data, plot.
5. Prior predictive check with `dsx.simulate`.
6. The two models; fit both with NUTS; `az.summary` side by side; wall times.
7. Forecasts with `forecast`; overlay plot (thumbnail); metrics table.
8. Filtered level plot.
9. Example 2 data with seasonality; the regression model; SVI fit; forecast plot.
10. `backtest` with a NUTS closure; per-window CRPS/coverage plot; `results_to_dataframe`.
11. Takeaways, boundaries (Section 12 in two paragraphs), link to this document.

File map:

```
DYNESTYX_INTEGRATION_DESIGN.md              # this document; status line updated
docs/examples/dynestyx_integration.py       # jupytext source, deleted after execution
docs/examples/dynestyx_integration.ipynb    # committed with outputs
pyproject.toml                              # `dynestyx` extra (not in `all`)
README.md                                   # extras sentence
```

## 15. Verification plan

1. `uv sync --extra all --extra dynestyx` resolves; `python -c "import numpyro_forecast"` does not import `dynestyx` (the `base-import` invariant).
2. The notebook executes end to end with `uv run jupytext --to notebook --execute docs/examples/dynestyx_integration.py`, every figure embedded.
3. Numerical checks inside the notebook: recovered $q$, $r$ within two posterior standard deviations of the truth for both fits; the two forecast fans agree visually and their CRPS agree within Monte Carlo error; the anchor invariant holds (the marginal variance of the first forecast row exceeds that of the anchor row by about $q^2$, one transition).
4. `uv run ruff check docs/examples/dynestyx_integration.ipynb && uv run ruff format --check docs/examples/dynestyx_integration.ipynb`.
5. `uv run pytest tests/test_docstring_markup.py tests/test_docs_reference.py` green (markup rules apply to notebook cells; no public API changes).
6. `prek run --all-files` green.
7. `make docs` renders the new example page with card description and thumbnail (optional, needs Quarto).

## 16. Risks

1. **dsx API drift.** The block relies on site names (`{name}_predicted_observations`, `{name}_predicted_states`), the anchor semantics and the `Simulator` ∘ `Filter` composition. All three are documented in dsx's `DiscreteTimeSimulator` docstring and tests, but the library is at 0.x. Mitigation: pin `>=0.5.0` now, add the `_api_canary` and the anchor test in v2, and re-run the notebook on each dsx minor.
2. **Silent off-by-one.** A user who forgets the anchor gets a forecast that lags one step with plausible-looking uncertainty. Mitigation: the block owns the anchor; the notebook and the v2 tests assert the variance increment.
3. **Traced time grids.** Anyone rewriting the block with `jnp.arange` breaks the rollout under `jit`. Mitigation: the comment and the `_time_grid` helper; a v2 test runs `forecast()` (which jits).
4. **Approximate default filter.** `filter_config=None` silently uses an EnKF on a linear-Gaussian model. Mitigation: every example passes `KFConfig()` explicitly and the docstring says so; consider defaulting to `KFConfig()` when `dynamics` is linear-Gaussian in v2.
5. **Dependency weight.** dsx's transitive closure includes `tfp-nightly`. Mitigation: a separate extra outside `all`; no CI job depends on it in v1.
6. **In-sample expectations.** Users may reach for `to_datatree` and hit the error. Mitigation: the error message names the alternatives; the notebook shows the filtered-level plot.

## 17. Follow-ups and questions

For `numpyro_forecast`:

- Promote the block to `contrib/dynestyx.py` (Section 13, v2) once the dsx maintainers have reviewed the composition.
- An in-sample driver for marginalized models: `predict_in_sample` variant that passes `data` to the model and reads a filtered or smoothed predictive site; or a documented recipe from `f_filtered_states_*` plus `predictions_to_datatree`.
- A `t0` argument on the block for absolute-time callable parameters in backtests.
- Panel support through `dsx.plate`.

For `dynestyx` (to raise upstream, wording to be agreed):

- `LTI_discrete` could infer `control_dim` from `D` when `B` is `None` (Section 10, item 7).
- A documented way to obtain the simulator's rollout from `dsx.sample`'s return value when a `Simulator` wraps a `Filter` (today only the sites carry it), or a blessed pure-JAX rollout entry point (Approach C).
- The anchor semantics of posterior rollouts deserve a line in the `DiscreteTimeSimulator` docstring.
- One-step-ahead predicted observations for the discrete `KFConfig` backends, so `Evaluation` covers the exact linear-Gaussian case.

Open decisions:

- Whether v2 tests run in CI with the extra installed (a fifth job) or stay `importorskip`.
- The block's name. `filtered_series` says what it does (the series is conditioned by a filter, in contrast with `markov_series`, whose path is sampled); `state_space` and `dynamical_series` were considered and rejected as less specific.

## Appendix A: probe results

All probes ran on Apple M4 Pro, CPU, `float32`, in the `uv` environment of this branch with `dynestyx==0.5.0`. Scripts were throwaway and are not committed.

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

Model of Section 6.3, NUTS 300 plus 300 draws: 26 s including compilation; posterior keys `q`, `r`, `f_marginal_loglik`, `f_filtered_states_*`. `forecast(key, model, posterior, data, covariates)`: shape `(300, 24, 1)` in 0.73 s; CRPS 0.476 on the held-out window. `forecast(..., batch_size=100, parallel=False)`: shape `(300, 24, 1)`. `backtest(..., min_train_window=100, test_window=12, stride=16, num_samples=200)` with a NUTS closure: three windows `(0, 100, 112)`, `(0, 116, 128)`, `(0, 132, 144)` with CRPS 0.580, 0.497, 0.498 in 83 s total.

### A.4 Covariates, SVI, and agreement with the direct model

Local level plus a Fourier regression (period 12, 2 harmonics, 4 covariates), $T = 120$, NUTS 2 chains of 500 plus 500. dsx model (`DynamicalModel` with `control_dim=4`, `D=beta[None, :]`, KF cuthbert): 40.8 s, ESS $q$ 1 033, $r$ 1 153, posterior means $q = 0.310$, $r = 0.483$, $\beta = (1.35, -0.10, -0.01, -0.18)$, CRPS 0.518. Direct nf model with `LocScaleReparam`: 4.7 s, ESS $q$ 351, $r$ 548, means $q = 0.308$, $r = 0.484$, $\beta = (1.35, -0.09, 0.00, -0.18)$, CRPS 0.521. Truth $q = 0.3$, $r = 0.5$. `LTI_discrete(..., D=beta[None, :])` without `B` raised `Controls are provided (shape: (120, 4)), but dynamics.control_dim is 0`.

SVI on the dsx model (`AutoNormal`, `Adam(0.01)`, 2 000 steps): 3.0 s; `draw_posterior` returned `beta`, `q`, `r` and the filter's deterministic sites; `forecast` from that posterior: shape `(500, 24, 1)`, CRPS 0.549.

### A.5 Backend cost

See the two tables in Section 11. The log likelihood at $q = 0.3$, $r = 0.5$ was $-124.36$ for all three KF backends (the direct model's value, $-1015.84$, is a joint density over the path and is not comparable).

### A.6 Time grids under the jitted driver

The block of Section 6.2 with the grids built three ways, forecast with `forecast()` (20 draws, 10 steps): `np.arange` succeeded with shape `(20, 10, 1)`; `jnp.asarray(np.arange(...))` and `jnp.arange` both failed inside the simulator's host-side segment bookkeeping with `TracerArrayConversionError`. The converse holds for `dsx.simulate`, the pure-JAX generator used for data simulation and prior predictive checks: it indexes the time grid inside a `lax.scan` and therefore needs a `jax.Array` (`jnp.asarray` of the NumPy grid), while a NumPy grid raises the same error there.
