# `contrib.dynestyx.state_space`: design and implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make `predict_in_sample`, `to_datatree` and `backtest(eval_train=True)` work for dynestyx state space models, with the dynestyx handler around the driver call as in PR #148, by adding one optional building block that owns the `"obs"` and `"forecast"` sites.

**Architecture:** `numpyro_forecast.contrib.dynestyx.state_space(h, name, y, dynamics, *, controls=None)` is a plain model function (no closure, no handler of its own) that builds the dynestyx time grids from the `Horizon`, calls `dsx.sample` once under a local `numpyro.handlers.trace`, and registers the package sites from the handler's return value and that trace: nothing beyond dynestyx's own marginal-likelihood factor while training, `"forecast"` from the Simulator's `<name>_predicted_observations` when `h.future > 0`, and `"obs"` when `h.data is None` by pushing in-sample state draws (the smoothing marginals under a `Smoother`, the posterior path under a `LatentPathBuilder`) through `dynamics.observation_model`. The drivers stay a site-name contract and never learn about dynestyx. dynestyx is imported lazily through `optional.require`, behind a new `dynestyx` extra.

**Tech Stack:** jax, numpyro, dynestyx 0.7.0 (`Filter`, `Smoother`, `LatentPathBuilder`, `DiscreteTimeSimulator`, `DynamicalModel`), pytest, jupytext.

**Spec:** this document (sections "Facts", "Contract", "Alternatives"). The review of PR #148 that motivated it is in the PR thread.

## Global Constraints

- Every function has complete type hints checked with `ty`; every public function has a NumPy-style docstring; docstring markup is the great-docs subset (`tests/test_docstring_markup.py`).
- Array layout: time at axis `-2`, observation dim at `-1`. The block supports no batch dims (no `dsx.plate`).
- Building blocks are plain functions that call `numpyro.sample`/`numpyro.deterministic`; they take no `rng_key` (randomness comes from the active `seed` handler, read with `numpyro.prng_key()`).
- Never spell the literal `"time"` plate name in package code (not needed here: the block opens no plate).
- Package import must not import dynestyx (`tests/test_package.py::test_base_import_no_extras`, CI `base-import` leg).
- Integer literals with four or more digits use underscores (`2_000`).
- No em-dashes, no hard-wrapped prose, American spelling, `$94\%$ HDI` in LaTeX.
- The notebook is authored as a `py:percent` script, executed with `uv run jupytext --to notebook --execute`, and only the `.ipynb` is committed; the `description` metadata and the `thumbnail` cell tag must survive.
- Every public symbol is listed under `reference:` in `great-docs.yml` (`tests/test_docs_reference.py`).
- `dynestyx>=0.7.0` (the lock at the PR head already resolves 0.7.0; every fact below was verified on 0.7.0, whose handler-stack validation and simulation-axis normalization the block relies on).

---

## Facts (dynestyx 0.7.0 source, verified by probes and an adversarial review on 2026-10-10)

1. `dsx.sample(name, dynamics, obs_times=, obs_values=, ctrl_times=, ctrl_values=, predict_times=)` (`handlers.py:202-258`) is interpreted by the innermost dynestyx handler, which forwards to the outer ones. The **return value** is the innermost handler's result: `dynestyx.types.ConditionedResult` under `Filter` and `Smoother` (`dists`: a Python list with one per-step marginal `Distribution` per observation time; filtering marginals under `Filter`, smoothing marginals under `Smoother`), `dynestyx.types.LatentStateResult` under `LatentPathBuilder` (`state_path`: the `(t_obs, state)` path; `state_path_params` is a **sample site** `<name>_state_path_params`, centered, so it is in the posterior and `Predictive` substitutes it). With `obs_values` given and no inference handler active, dynestyx's `_validate_handler_stack` (`handlers.py:94-96`) raises `ValueError("Observations require Filter, Smoother, or LatentPathBuilder. ...")` before any result exists, and with `predict_times` given and no Simulator it raises `ValueError("predict_times requires a Simulator. ...")` (`handlers.py:97-98`). The block therefore owns neither error and has no branch for them.
2. With a `DiscreteTimeSimulator(n_simulations=S)` outside a `Filter`/`Smoother`/`LatentPathBuilder` and `predict_times` given, the simulator rolls the final conditioned state forward and registers `<name>_predicted_times` `(S, P)`, `<name>_predicted_states` `(S, P, state)`, `<name>_predicted_observations` `(S, P, obs)` and `<name>_1_x_0` as deterministic sites, unconditionally (`simulation/utils.py:93-100`, no record gating). These are **not** in the return value (the inner handler returns its own result), only in the trace. With `predict_times=None` the simulator passes through and registers nothing, but `_validate_handler_stack` (`handlers.py:103-108`) emits `UserWarning("Simulator has no predict_times to simulate at.")`; the block silences exactly that message when `h.future == 0`, so one `with DiscreteTimeSimulator(n_simulations=1), Smoother(): ...` stack serves `predict_in_sample`, `forecast` and `to_datatree` without noise.
3. `Smoother` and `LatentPathBuilder` reject `predict_times < max(obs_times)` ("in-window smoothing predictions are not implemented yet. Please use `Filter` for in-window predictions for now.", `inference/smoothers.py:140-147`). dynestyx therefore cannot produce in-sample observation draws under a smoother; the block does it from `dists`.
4. `Smoother` registers `<name>_smoothed_states_mean` `(t_obs, state)` (and `_cov`, `_cov_diag`, cuthbert also `_chol_cov`), `Filter` registers `<name>_filtered_states_mean` etc. Recording is gated per field by `record_smoothed_states_mean: bool | None` (and siblings) and `record_max_elems: int = 100_000` on `BaseSmootherConfig` (`inference/configs/smoother.py:57-63`; `None` records when the field has at most `record_max_elems` elements); under any `dsx.plate` no state site is registered at all (`smoothers.py:367-379`). Both handlers return the same `ConditionedResult` type with no discriminator field, so the block tells them apart by the presence of `<name>_smoothed_states_mean` in its local trace and refuses to proceed when it is absent (loud failure, never a silent filtered predictive). `dists` is not always `MultivariateNormal` (EnRTS gives `LowRankMultivariateNormal`, the PF smoother `WeightedParticles`); the per-step unflatten below handles any uniform list whose elements implement `.sample`, and the block raises if `dists` is empty.
5. numpyro `Distribution` objects are pytrees whose aux data carries `batch_shape`; stacking the leaves of the `dists` list with `jax.tree.map` yields a distribution with a stale `batch_shape=()` (`.mean` fails). The safe vectorization is: stack the leaves, then `jax.vmap` a function that unflattens one step's leaves with the shared treedef and samples it (verified for `MultivariateNormal` from both backends).
6. `dynamics.observation_model(x_t, u_t, t)` returns a numpyro `Distribution` for one time step (`LinearGaussianObservation.__call__` returns a `MultivariateNormal` and accepts `u=None`; a user callable such as `lambda x, u, t: dist.StudentT(nu, x + bias + u @ w, sigma).to_event(1)` does the same). A `jax.vmap` over `(x_t, u_t, t, key)` gives the `(t_obs, obs)` observation draw.
7. Missing data: cuthbert `KFConfig`/`EnKFConfig` filters and `KFSmootherConfig`/`EnRTSSmootherConfig` smoothers accept NaN in `obs_values`; cd_dynamax raises; `PFConfig` only warns; `LatentPathBuilder` accepts (`inference/checkers.py::_validate_missing_observation_support`).
8. `LatentPathBuilder` samples the path **centered** with no reparameterization option (`inference/latent/builder.py:469-473`); the block does not change that.
9. `numpyro.infer.Predictive` masks the model with `mask=False`, drops deterministic sites from `posterior_samples` (`exclude_deterministic=True`) so `<name>_state_path` is recomputed from the substituted `<name>_state_path_params`, and returns deterministic sites named in `return_sites`.
10. `DiscreteTimeSimulator` places the rollout's initial state **at the first predict time** and transitions only between consecutive predict times (`simulation/discrete.py:46-83`). With `predict_times = [t_obs, ...]` the first forecast step is therefore a draw of the anchor (the smoothed state at `t_obs - 1`, or the final path state) with no transition, and every later step is lagged by one; verified with `A = 0.8`: predicted state mean at the first step `1.619` against the smoothed `m_T = 1.621`, variance `0.0295` against `P_T = 0.0292`. Asking for `predict_times = [t_obs - 1, ..., duration - 1]` (dynestyx allows `predict_times >= max(obs_times)`) and dropping the first row gives the correct rollout: means `A^k m_T`, variances `A^{2k} P_T + q^2 (1 + ... + A^{2(k-1)})` to Monte Carlo precision for both `Smoother` and `LatentPathBuilder`. The block does this; the direct `forecast(site="f_predicted_observations")` of PR #148 does not, so its dynestyx forecasts are lagged by one step (harmless for the random walk mean, too narrow by `q^2` at the first step).
11. dynestyx 0.7.0's `LatentPathBuilder` infers the observation missingness layout from **concrete** `obs_values` on its first call and caches it per `(name, obs_values.shape)` (`inference/latent/builder.py:86-110`); under a `jit`/`vmap`/`pmap` trace it raises `ValueError("... needs a fixed observation missingness pattern ... Reuse a builder that has first seen concrete observations ...")` unless the cache is warm. `MCMC(chain_method="parallel")` traces `initialize_model` inside `pmap`, so a fresh builder fails there; `chain_method="sequential"` runs the first init eagerly and works; the jitted drivers work with the builder instance that fitted the model. On the executing machine (macOS arm64, CPU, jax from the lock) the `pmap`ped NUTS program of a path-builder model segfaults in XLA compilation even with a warm builder (adapter form too, so unrelated to the block); the notebook runs the path-builder fits and their `innovations` counterpart with `chain_method="sequential"` and says so.

## Contract

```python
def state_space(
    h: Horizon,
    name: str,
    y: Float[Array, " time obs"],
    dynamics: Any,
    *,
    controls: Float[Array, " duration control"] | None = None,
) -> None
```

- `h`: the current call's `Horizon`. `y`: the observed window, `(h.t_obs, obs)`, read from the covariates by the caller (as with `ssoe`); NaN entries are missing observations, and they must be in `y` (the likelihood reads `y`, never `h.data`; `to_datatree`'s `observed_data` reads `data`). `dynamics`: a `dynestyx.DynamicalModel` (annotated `Any`, dynestyx is optional). `controls`: `(h.duration, control)` covariates over the whole horizon, or `None`.
- Time grids: `times = jnp.arange(h.duration, dtype=jnp.float32)`; `obs_times = times[: h.t_obs]`; `predict_times = times[h.t_obs - 1 :]` when `h.future > 0` else `None` (anchor-inclusive, fact 10); `ctrl_times = times` when `controls` is given.
- Modes, decided by `h` exactly as the other building blocks do:
  - **training** (`h.data is not None`, `h.future == 0`): `dsx.sample(...)`; dynestyx's `<name>_marginal_log_likelihood` factor (or the `LatentPathBuilder` joint factor) is the likelihood; the block registers no site.
  - **forecast** (`h.data is not None`, `h.future > 0`): requires `<name>_predicted_observations` in the local trace (a Simulator around the call), shape `(n_simulations, future + 1, obs)` with leading axis `1` (`n_simulations=1`); registers `numpyro.deterministic("forecast", predicted[0, 1:])`, shape `(future, obs)` (the dropped row is the anchor, fact 10).
  - **in-sample predictive** (`h.data is None`, which implies `h.future == 0`): state draw `x` of shape `(t_obs, state)`: `result.state_path` under `LatentPathBuilder` (a joint posterior path); one draw per step from `result.dists` under `Smoother` (per-step smoothing marginals, independent across steps); then `obs = vmap(observation_model(x_t, u_t, t).sample)` and `numpyro.deterministic("obs", obs)`, shape `(t_obs, obs)`.
- Errors (`ValueError` unless stated): `y.shape[-2] != h.t_obs`; `controls.shape[-2] != h.duration` (rank is enforced by the jaxtyping annotations, so a batch dim raises `jaxtyping.TypeCheckError`, a `TypeError`, before the body runs: the module is imported inside the package's `install_import_hook` block); forecast with `n_simulations != 1`; in-sample under a `ConditionedResult` without `<name>_smoothed_states_mean` (a `Filter`, or a `Smoother` with `record_smoothed_states_mean=False` or a mean larger than `record_max_elems`), message pointing at the `Smoother` and noting that a `Filter` fit's draws are valid under it; in-sample with an empty `dists`; in-sample with no `seed` handler (`numpyro.prng_key()` is `None`): `RuntimeError`. A missing handler and a missing Simulator are dynestyx's own `ValueError`s (fact 1).
- Documented caveats: the `Smoother` in-sample draws are per-step marginals (exact for per-step bands, CRPS and coverage; not a joint path), the `LatentPathBuilder` draws are joint; the block models one series per model function (no batch dims, and it owns the bare-named `"obs"` and `"forecast"` sites, so it cannot run under `handlers.scope` or be called twice in one model); the `Filter` is for fitting only; missing data per fact 7.
- Coupling surface, all in one module: `dynestyx.sample`, `dynestyx.types.ConditionedResult` (`.dists`), `dynestyx.types.LatentStateResult` (`.state_path`), the site suffixes `_predicted_observations` and `_smoothed_states_mean`, and `DynamicalModel.observation_model(x, u, t)`. A canary test pins the attribute surface.
- Time grids (settled by the code review's probes): `obs_times` is a `jnp` array, because the `LatentPathBuilder`'s forward sampler indexes it with a traced step inside a `lax.scan` (a NumPy grid raises `TracerArrayConversionError` at the first fit init); `predict_times` and `ctrl_times` are NumPy, because dynestyx's Simulator segments the rollout on the host (`simulation/base.py:225`, `np.asarray(jax.device_get(...))`). The one combination left out is forecasting under a bare `Filter` plus Simulator: its segmentation also converts `obs_times` on the host, which a traced grid inside the jitted drivers cannot survive, so the block translates that `TracerArrayConversionError` into a `ValueError` pointing at the `Smoother` (same posterior, same forecast). Upstream could normalize both sides (`jnp.asarray` in the sampler, host conversion only when concrete).
- Scalar-event observation models (`dist.Normal(x[0], r)`) get their event axis appended in `_observe`, as dynestyx pads its own rollout, so the `"obs"` site keeps the `(time, obs)` layout.

Notebook usage after this change (the adapters disappear):

```python
def gaussian_ssm(covariates, data=None):
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]  # the observed series travels in column 0
    controls = covariates[..., 1:]  # the Fourier features span the full horizon
    ...  # priors
    state_space(h, "f", y, gaussian_dynamics(bias, weight, drift_scale, sigma), controls=controls)


with kalman:  # Filter or Smoother
    mcmc.run(key, covariates[:t_obs], train)

with DiscreteTimeSimulator(n_simulations=1), smoother:
    tree = to_datatree(key, gaussian_ssm, mcmc.get_samples(), train, covariates, num_chains=4)
```

## Alternatives considered

- **Notebook-only recipe** (closed-form Gaussian predictive from the `_smoothed_states_*` sites, then `predictions_to_datatree`): zero coupling, but Gaussian-only unless the user vmaps the observation model, re-derived by every user, and `to_datatree`/`backtest(eval_train=True)` stay unsupported.
- **Dispatch inside the drivers**: rejected. The drivers are a site-name contract; dynestyx is optional; dynestyx has no public active-handler API; it would put dynestyx site names into `predictive.py`.
- **A `conditioner` argument on the block** (the previous notebook design): rejected now that the drivers jit per call, because the handler around the driver call is the dynestyx idiom and composes with `Predictive`.
- **Upstream**: if `Smoother` forwards all smoothed marginals so that `Simulator(predict_times=obs_times)` yields the smoothing predictive, or if `simulation/utils._sample_observation_path` becomes public, the in-sample branch shrinks to reading `<name>_predicted_observations` with no change to the block's API.

The `site` parameter that PR #148 added to `forecast()` has no consumer once the block registers `"forecast"`, so this change removes it (separate commit, easy to drop if wanted).

---

## File structure

- Create `numpyro_forecast/contrib/dynestyx.py`: the block, its validation helpers, the in-sample draw helpers. One responsibility: translate a `Horizon` and the package sites to and from one `dsx.sample` call.
- Create `tests/contrib/test_dynestyx.py`: behavior tests on a Gaussian local level with a closed form, `pytest.importorskip("dynestyx")`.
- Modify `numpyro_forecast/__init__.py`: import the module inside the jaxtyping hook block (runtime type checks) so great-docs can resolve `contrib.dynestyx.*`.
- Modify `pyproject.toml`: new `dynestyx` extra, `docs` depends on it, `all`/`all_cuda` include it.
- Modify `tests/test_package.py`, `.github/workflows/ci.yml`: add `dynestyx` to the leak lists.
- Modify `great-docs.yml`: `reference:` entry, skill lines.
- Modify `README.md`, `AGENTS.md`: extras and conventions.
- Modify `numpyro_forecast/predictive.py`, `tests/test_predictive.py`: remove `forecast(site=...)`.
- Modify `docs/examples/dynestyx_integration.ipynb`: models use the block; new in-sample predictive section with `to_datatree`; takeaways for the new cells.

## Task 1: Packaging and import isolation

**Files:**
- Modify: `pyproject.toml:74-82`
- Modify: `tests/test_package.py:123`
- Modify: `.github/workflows/ci.yml:64`
- Modify: `README.md:28-31`
- Modify: `numpyro_forecast/contrib/__init__.py`

- [ ] **Step 1: Add the extra**

```toml
docs = [
    "great-docs==0.17.0",
    "numpyro_forecast[dynestyx]",
]
dynestyx = [
    "dynestyx>=0.7.0",
]
cuda = ["jax[cuda12]; sys_platform == 'linux'"]
all = ["numpyro_forecast[dataframes,optax,blackjax,dynestyx,dev,docs]"]
all_cuda = ["numpyro_forecast[dataframes,optax,blackjax,dynestyx,dev,docs,cuda]"]
```

- [ ] **Step 2: Extend the leak lists**

`tests/test_package.py`: `leaked = [m for m in ('optax', 'blackjax', 'dynestyx') if m in sys.modules]`. `ci.yml`: `('optax','blackjax','polars','dynestyx')`.

- [ ] **Step 3: README extras bullets**

Add `- \`dynestyx\`: the \`state_space\` building block in \`numpyro_forecast.contrib.dynestyx\` for [dynestyx](https://github.com/BasisResearch/dynestyx) state space models.` and rewrite the `all` bullet: "the four above plus the `dev` and `docs` tooling (`all_cuda` adds `cuda`)".

- [ ] **Step 4: `uv sync --extra all` and run `tests/test_package.py`**

Run: `uv sync --extra all && uv run pytest tests/test_package.py -q`
Expected: pass.

- [ ] **Step 5: Commit** `build: add the dynestyx extra`

## Task 2: The building block

**Files:**
- Create: `numpyro_forecast/contrib/dynestyx.py`
- Create: `tests/contrib/test_dynestyx.py`
- Modify: `numpyro_forecast/__init__.py:8-28`
- Modify: `great-docs.yml:149-160`

**Interfaces:**
- Produces: `state_space(h, name, y, dynamics, *, controls=None) -> None` as in the Contract.

- [ ] **Step 1: Test fixture and the closed-form training test**

```python
"""Tests for the dynestyx state space building block (``numpyro_forecast.contrib.dynestyx``)."""

from contextlib import AbstractContextManager

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer.util import log_density

from numpyro_forecast import Horizon, backtest, forecast, predict_in_sample, to_datatree
from numpyro_forecast.contrib.dynestyx import state_space
from numpyro_forecast.optional import _api_canary
from numpyro_forecast.typing import Array

pytest.importorskip("dynestyx")
from dynestyx import (  # the statement-form importorskip above is an import boundary for ruff
    DiscreteTimeSimulator,
    DynamicalModel,
    Filter,
    LatentPathBuilder,
    Smoother,
)
from dynestyx.inference.configs.smoother import KFSmootherConfig
from dynestyx.inference.filters import KFConfig
from dynestyx.models import LinearGaussianObservation, LinearGaussianStateEvolution

T_OBS, FUTURE = 24, 6
Q_TRUE, R_TRUE, P0 = 0.3, 0.2, 4.0


def local_level_dynamics(q: Array, r: Array) -> DynamicalModel:
    """Random walk level observed with noise: x_t = x_{t-1} + q e_t, y_t = x_t + r v_t."""
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), P0 * jnp.eye(1)),
        state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=q**2 * jnp.eye(1)),
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=r**2 * jnp.eye(1)),
    )


def local_level(covariates: Array, data: Array | None = None) -> None:
    """The series travels in the covariates (column 0), as with ``ssoe``."""
    h = Horizon.from_data(covariates, data)
    q = numpyro.sample("q", dist.HalfNormal(1.0))
    r = numpyro.sample("r", dist.HalfNormal(1.0))
    state_space(h, "f", covariates[..., : h.t_obs, :], local_level_dynamics(q, r))


def simulate(rng_key: Array, n: int) -> Array:
    """A random walk observed with noise; dynestyx observes the initial state at the first time."""
    key_x0, key_q, key_r = random.split(rng_key, 3)
    x0 = jnp.sqrt(P0) * random.normal(key_x0)
    steps = jnp.concatenate([jnp.zeros(1), jnp.cumsum(Q_TRUE * random.normal(key_q, (n - 1,)))])
    return (x0 + steps + R_TRUE * random.normal(key_r, (n,)))[:, None]


SERIES = simulate(random.PRNGKey(0), T_OBS + FUTURE)
TRAIN, COVARIATES = SERIES[:T_OBS], SERIES
PARAMS = {"q": jnp.asarray(Q_TRUE), "r": jnp.asarray(R_TRUE)}
smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert"))


def closed_form_loglik(y: Array, q: float, r: float) -> Array:
    """Marginal Gaussian log likelihood of a random walk plus noise observed at 0..T-1."""
    t = jnp.arange(y.shape[0])
    cov = P0 + q**2 * jnp.minimum(t[:, None], t[None, :]) + r**2 * jnp.eye(y.shape[0])
    return dist.MultivariateNormal(jnp.zeros(y.shape[0]), cov).log_prob(y[:, 0])


def test_training_registers_the_marginal_likelihood() -> None:
    """Under a Smoother the model's log density is the priors plus the exact Kalman marginal."""
    with smoother:
        joint, tr = log_density(local_level, (TRAIN, TRAIN), {}, PARAMS)
    priors = dist.HalfNormal(1.0).log_prob(PARAMS["q"]) + dist.HalfNormal(1.0).log_prob(
        PARAMS["r"]
    )
    assert jnp.allclose(joint, priors + closed_form_loglik(TRAIN, Q_TRUE, R_TRUE), rtol=1e-4)
    assert "obs" not in tr and "forecast" not in tr
```

Note on the `level.at[0].set(x0)`: dynestyx places the initial state at the first observation time (no transition before it), so $\operatorname{Cov}(y_s, y_t) = P_0 + q^2\min(s,t) + r^2\,\delta_{st}$ with $t$ counted from $0$. If the first assertion fails by a constant offset, check this convention first.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/contrib/test_dynestyx.py -q`
Expected: `ModuleNotFoundError: numpyro_forecast.contrib.dynestyx`.

- [ ] **Step 3: Write the module**

```python
"""dynestyx state space models as a `numpyro_forecast` building block.

[Module docstring: the design context from this document's Contract, Facts 1 to 9
condensed, the three modes, the caveats, the coupling surface, the upstream note.]
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
    key = numpyro.prng_key()
    if key is None:
        msg = (
            "the in-sample predictive of state_space draws the states and observations, so the "
            "model must run under a seed handler (Predictive, predict_in_sample, to_datatree)"
        )
        raise RuntimeError(msg)
    return key


def _in_sample_states(result: Any, trace: dict[str, Any], name: str, dsx: Any) -> Array:
    """One in-sample state draw ``(t_obs, state)``: the posterior path, or the smoothing marginals."""
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
        return jax.tree.unflatten(treedef, step_leaves).sample(key)

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
    """[NumPy docstring per the Contract: Parameters, Raises, Notes with the modes and caveats, a short Examples block.]"""
    dsx = require("dynestyx", extra="dynestyx")
    _validate_window(h, y, controls)
    times = jnp.arange(h.duration, dtype=jnp.float32)
    obs_times = times[: h.t_obs]
    predict_times = times[h.t_obs - 1 :] if h.future > 0 else None  # anchor-inclusive, see Notes
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
```

The `strict=True` zip over the per-distribution leaf lists requires that every step's distribution has the same treedef; a mismatch raises `ValueError` from `zip`, which is the right failure (mixed distribution types across steps are not a supported `dists`). dynestyx itself raises when no inference handler is active or when `predict_times` is given without a Simulator (fact 1), so the block has no branch for either. `warnings.catch_warnings` is entered around the trace so the filter is scoped to the one `dsx.sample` call.

- [ ] **Step 4: Register the module**

`numpyro_forecast/__init__.py`, inside the `with install_import_hook(...)` block, after the `from numpyro_forecast import (...)` import: `import numpyro_forecast.contrib.dynestyx  # noqa: F401` with the comment that it imports no optional extra and exists for the great-docs attribute walk and runtime type checks. `great-docs.yml` `Extensions (contrib)` contents: add `- contrib.dynestyx.state_space` and extend `desc` to "Optional backends behind pyproject extras (never imported by default): BlackJAX kernels and Pathfinder, the dynestyx state space building block."

- [ ] **Step 5: Run the test, verify it passes**

Run: `uv run pytest tests/contrib/test_dynestyx.py tests/test_package.py tests/test_docs_reference.py tests/test_docstring_markup.py -q`

- [ ] **Step 6: Commit** `feat(contrib): dynestyx state_space building block`

## Task 3: Behavior tests for the three modes and the errors

**Files:**
- Modify: `tests/contrib/test_dynestyx.py`

- [ ] **Step 1: Add the tests**

```python
def _repeat(params: dict[str, Array], n: int) -> dict[str, Array]:
    return {k: jnp.broadcast_to(v, (n, *v.shape)) for k, v in params.items()}


def _smoothed_moments() -> tuple[Array, Array]:
    """Smoothed state mean and covariance diagonal at PARAMS, ``(t_obs, 1)`` each, from dynestyx's sites."""
    with smoother:
        tr = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(local_level, random.PRNGKey(0)), data=PARAMS
            )
        ).get_trace(TRAIN, TRAIN)
    return tr["f_smoothed_states_mean"]["value"], tr["f_smoothed_states_cov_diag"]["value"]


def test_in_sample_predictive_matches_the_smoothing_moments() -> None:
    """Under a Smoother the ``obs`` draws have mean ``m_t`` and variance ``P_t + r^2`` per step."""
    n = 4_000
    with smoother:
        draws = predict_in_sample(random.PRNGKey(1), local_level, _repeat(PARAMS, n), TRAIN)
    mean, var = _smoothed_moments()
    assert draws.shape == (n, T_OBS, 1)
    assert jnp.allclose(draws.mean(axis=0), mean, atol=4 * jnp.sqrt((var + R_TRUE**2) / n).max())
    assert jnp.allclose(draws.var(axis=0), var + R_TRUE**2, rtol=0.15)


def test_in_sample_predictive_replays_the_latent_path() -> None:
    """Under a LatentPathBuilder the ``obs`` draws are the posterior path plus observation noise."""
    n = 2_000
    path = jnp.linspace(-1.0, 1.0, T_OBS)[:, None]
    posterior = _repeat({**PARAMS, "f_state_path_params": path}, n)
    with LatentPathBuilder():
        draws = predict_in_sample(random.PRNGKey(1), local_level, posterior, TRAIN)
    residual = draws - path
    assert jnp.allclose(residual.mean(axis=0), 0.0, atol=4 * R_TRUE / jnp.sqrt(n))
    assert jnp.allclose(residual.std(axis=0), R_TRUE, rtol=0.1)


def test_forecast_rolls_the_smoothed_state_forward() -> None:
    """``forecast`` reads the Simulator rollout: ``(sample, future, obs)`` with the random walk mean."""
    n = 2_000
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        draws = forecast(random.PRNGKey(2), local_level, _repeat(PARAMS, n), TRAIN, COVARIATES)
    mean, var = _smoothed_moments()
    assert draws.shape == (n, FUTURE, 1)
    # a random walk's forecast mean is the last smoothed state, the variance grows by q^2 per step
    expected_var = var[-1] + Q_TRUE**2 * jnp.arange(1, FUTURE + 1)[:, None] + R_TRUE**2
    assert jnp.allclose(draws.mean(axis=0), mean[-1], atol=4 * jnp.sqrt(expected_var / n).max())
    assert jnp.allclose(draws.var(axis=0), expected_var, rtol=0.15)


def test_to_datatree_exports_both_predictive_groups() -> None:
    """One handler stack serves the in-sample predictive and the forecast of ``to_datatree``."""
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        tree = to_datatree(
            random.PRNGKey(3), local_level, _repeat(PARAMS, 8), TRAIN, COVARIATES, num_chains=2
        )
    assert tree["posterior_predictive"]["obs"].shape == (2, 4, T_OBS, 1)
    assert tree["predictions"]["obs"].shape == (2, 4, FUTURE, 1)


def test_missing_observations_widen_the_in_sample_band() -> None:
    """NaN rows are skipped by the cuthbert smoother and the predictive is widest inside the gap."""
    gap = slice(8, 14)
    train = TRAIN.at[gap].set(jnp.nan)
    with smoother:
        draws = predict_in_sample(random.PRNGKey(4), local_level, _repeat(PARAMS, 2_000), train)
    sd = draws.std(axis=0)[:, 0]
    assert jnp.isfinite(draws).all()
    assert sd[gap].min() > sd[: gap.start].max()


def test_in_sample_predictive_rejects_a_filter() -> None:
    """A Filter has only the filtering marginals: the block refuses and points at the Smoother."""
    with (
        Filter(filter_config=KFConfig(filter_source="cuthbert")),
        pytest.raises(ValueError, match="Smoother"),
    ):
        predict_in_sample(random.PRNGKey(5), local_level, _repeat(PARAMS, 2), TRAIN)


def test_forecast_requires_a_single_simulation() -> None:
    """``n_simulations != 1`` is rejected: the package's forecast site has no simulation axis."""
    with (
        DiscreteTimeSimulator(n_simulations=2),
        smoother,
        pytest.raises(ValueError, match="n_simulations=1"),
    ):
        forecast(random.PRNGKey(6), local_level, _repeat(PARAMS, 2), TRAIN, COVARIATES)


def test_window_shapes_are_validated() -> None:
    """The time axes must match the horizon; a batch dim fails the jaxtyping rank check."""
    h = Horizon.from_data(COVARIATES, TRAIN)
    dynamics = local_level_dynamics(jnp.asarray(Q_TRUE), jnp.asarray(R_TRUE))
    with pytest.raises(ValueError, match="t_obs"):
        state_space(h, "f", TRAIN[:-1], dynamics)
    with pytest.raises(ValueError, match="duration"):
        state_space(h, "f", TRAIN, dynamics, controls=jnp.zeros((T_OBS, 2)))
    with pytest.raises(TypeError):  # jaxtyping.TypeCheckError: " time obs" is exactly 2-D
        state_space(h, "f", TRAIN[None], dynamics)


def test_backtest_scores_in_sample_under_one_handler_stack() -> None:
    """``backtest(eval_train=True)`` runs ``predict_in_sample`` and ``forecast`` through the block."""

    def forecast_fn(rng_key, model, train, train_cov, full_cov, num_samples, *, batch_size=None):
        return forecast(rng_key, model, _repeat(PARAMS, num_samples), train, full_cov)

    def in_sample_fn(rng_key, model, train, train_cov, num_samples, *, batch_size=None):
        return predict_in_sample(rng_key, model, _repeat(PARAMS, num_samples), train_cov)

    with DiscreteTimeSimulator(n_simulations=1), smoother:
        results = backtest(
            random.PRNGKey(7),
            lambda: local_level,
            SERIES,
            COVARIATES,
            forecast_fn=forecast_fn,
            in_sample_fn=in_sample_fn,
            train_window=12,
            test_window=6,
            stride=6,
            num_samples=20,
            eval_train=True,
        )
    assert len(results) == 3
    assert all(
        jnp.isfinite(r.metrics["crps"]) and jnp.isfinite(r.train_metrics["crps"]) for r in results
    )


def test_dynestyx_surface_canary() -> None:
    """The attribute and site-name surface the block depends on, pinned against drift."""
    _api_canary(
        "dynestyx",
        ["sample", "types.ConditionedResult", "types.LatentStateResult", "DiscreteTimeSimulator"],
    )
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        tr = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(local_level, random.PRNGKey(0)), data=PARAMS
            )
        ).get_trace(COVARIATES, TRAIN)
    assert tr["f_smoothed_states_mean"]["value"].shape == (T_OBS, 1)
    assert tr["f_predicted_observations"]["value"].shape == (1, FUTURE + 1, 1)
    assert tr["forecast"]["value"].shape == (FUTURE, 1)
```

The `backtest` closures follow the `ForecastFn` / `InSampleFn` signatures in `numpyro_forecast/typing.py` (check the exact parameter order there before writing them; `BacktestResult.train_metrics` is the `eval_train=True` output, confirm the attribute name in `evaluate.py`). The canary test pins the two site suffixes the block reads, which `_api_canary` cannot express, and the forecast site's `(1, future + 1, obs)` layout (fact 10).

- [ ] **Step 2: Run, fix, run**

Run: `uv run pytest tests/contrib/test_dynestyx.py -q` then the full suite `make tests`.

- [ ] **Step 3: Commit** `test(contrib): dynestyx block behavior`

## Task 4: Remove `forecast(site=...)`

**Files:**
- Modify: `numpyro_forecast/predictive.py` (the `site` parameter, its docstring entry and the Notes sentence about dynestyx's `f_predicted_observations` / `flatten_draws`; `_predictive_kernel(model, site, parallel)` keeps `site` as the internal argument because `predict_in_sample` passes `"obs"`).
- Modify: `tests/test_predictive.py` (delete `test_forecast_reads_any_named_site`).

- [ ] **Step 1: Edit, run `uv run pytest tests/test_predictive.py -q`, commit** `refactor: drop forecast(site=), the dynestyx block registers "forecast"`

## Task 5: Notebook

**Files:**
- Modify: `docs/examples/dynestyx_integration.ipynb` via `uv run jupytext --to py:percent` round trip.

- [ ] **Step 1: Convert to a script**

Run: `uv run jupytext --to py:percent --output docs/examples/dynestyx_integration.py docs/examples/dynestyx_integration.ipynb`

- [ ] **Step 2: Edit the script**

- Intro: "A `dynestyx` model plugs into the drivers through the `state_space` building block of `numpyro_forecast.contrib.dynestyx`: it turns the `Horizon` into the dynestyx time grids and registers the `\"obs\"` and `\"forecast\"` sites. The `dynestyx` handler goes around each call, as in the `dynestyx` examples." Import `from numpyro_forecast.contrib.dynestyx import state_space` and `from dynestyx import Smoother`, `from dynestyx.inference.configs.smoother import KFSmootherConfig`; `to_datatree` from `numpyro_forecast`.
- `arma_ssm(covariates, data=None)`: `h = Horizon.from_data(covariates, data)`, `y = covariates[..., : h.t_obs, :]`, priors, `state_space(h, "f", y, arma_dynamics(mu, phi, theta, sigma))`. Delete `arma_adapter`; fit `NUTS(arma_ssm)`.
- BART: `bart_covariates = jnp.concatenate([bart, fourier_features(...)], axis=-1)` with the comment "column 0 is the series, columns 1: the 52 Fourier terms, shape (duration, 53)" and a `print` of the shape. `univariate_model` and `gaussian_model` read `fourier = covariates[..., 1:]`, size `weight` with `fourier.shape[-1]` and compute `regression = (weight * fourier).sum(axis=-1, keepdims=True)`. `level_regression(covariates, data=None)` and `gaussian_ssm(covariates, data=None)` read `y = covariates[..., : h.t_obs, :1]` and `controls = covariates[..., 1:]`, build their `DynamicalModel` with `control_dim=controls.shape[-1]` (and `D=weight[None, :]` for the Gaussian one), and call `state_space(h, "f", y, dynamics, controls=controls)`. Delete both adapters.
- Forecast cell: `with DiscreteTimeSimulator(n_simulations=1), latent_path: forecasts["dynestyx latent path"] = forecast(key_latent, level_regression, ..., batch_size=500)`; no `site`, no `flatten_draws`. Prose: "The `dynestyx` forecast is the `Simulator` rollout, registered as `\"forecast\"` by the block, which asks the simulator for one extra step at the last observation so the first forecast step is a transition and not a copy of the last state."
- New section after "Integrating out the level": `## In-sample posterior predictive`. Prose before: what `predict_in_sample`/`to_datatree` need (the smoothing distribution), that the Filter fit's draws are valid under a Smoother because both share the marginal likelihood, and that the Smoother draws are per-step marginals while the `innovations` and `LatentPathBuilder` draws are joint paths. Code: `smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert"))`; `trees = {"innovations": to_datatree(key, gaussian_model, mcmc_gaussian.get_samples(), bart_train, bart_covariates, num_chains=4)}` and under `with DiscreteTimeSimulator(n_simulations=1), smoother:` the same for `gaussian_ssm` with `mcmc_gaussian_kalman.get_samples()`. One figure (two panels, `layout="constrained"`, bold suptitle, shared legend outside) with the in-sample $94\%$ HDI band from `posterior_predictive` and the forecast band from `predictions` over the last 104 training weeks plus the 52 test weeks. Print the in-sample and test CRPS per model. Markdown after with the numbers read from the output.
- Missing data section: the block reads the window from the covariates, so the gap must be in the covariates: `bart_covariates_gap = bart_covariates.at[gap, 0].set(jnp.nan)` (and `bart_gap = bart_train.at[gap].set(jnp.nan)` stays as `data`); fit `level_regression` with `gap_mcmc.run(key_gap, bart_covariates_gap[:t_obs], bart_gap)`. Say in the prose that the NaNs live in the covariate column the block reads. Unchanged otherwise.
- Any `for` loop or `print` the edit touches gets its blank lines.

- [ ] **Step 3: Execute and inspect**

Run: `uv run jupytext --to notebook --execute docs/examples/dynestyx_integration.py && rm docs/examples/dynestyx_integration.py`
Then dump the text outputs and patch every number in the prose of the touched cells; check the `description` metadata and the `thumbnail` tag survived; run `uv run pytest tests/test_build_docs.py tests/test_docstring_markup.py tests/test_examples.py -q`.

- [ ] **Step 4: Commit** `docs: dynestyx notebook uses the state_space block; in-sample predictive section`

## Task 6: Conventions and agent-facing docs

**Files:**
- Modify: `AGENTS.md` (building-blocks paragraph: add a sentence on the optional block; dependencies sentence: the `dynestyx` extra; the dev-docs example file name can point at this document).
- Modify: `great-docs.yml` skill lines: replace "A dynestyx model plugs in through an adapter ... predict_in_sample and to_datatree are not supported for dynestyx models." with the block usage: "A dynestyx model is a model function that calls `state_space(h, name, y, dynamics, controls=...)` from `numpyro_forecast.contrib.dynestyx` (extra `dynestyx`), with the observed window (and its NaNs) read from the covariates. Fit under `with Filter()/Smoother()/LatentPathBuilder(): mcmc.run(...)`; call `forecast`, `predict_in_sample` and `to_datatree` under `with DiscreteTimeSimulator(n_simulations=1), Smoother(): ...` (or a `LatentPathBuilder`); a bare `Filter` cannot give the in-sample predictive. The Smoother's in-sample draws are per-step marginals, not joint paths." Fix the missing-data line: "the cd_dynamax backend rejects them and the particle filter only warns". Also fix `numpyro_forecast/contrib/__init__.py`'s docstring (it names only blackjax).

- [ ] **Step 1: Edit, run `uv run pytest tests/test_docs_reference.py tests/test_docstring_markup.py -q` and `prek run --all-files`, commit** `docs: conventions for the dynestyx block`

## Task 7: Full verification

- [x] `uv run ruff check . && uv run ruff format --check . && uv run ty check && make tests` (793 passed before the review fixes; re-run after them).
- [x] Smoke: the notebook itself (`gaussian_ssm` fitted under `Filter`, exported with `to_datatree` under `DiscreteTimeSimulator(1), Smoother()`), executed end to end.

## Review outcome

An adversarial review of the plan (before execution) and of the code (after) each ran against the dynestyx 0.7.0 source. The plan review removed two dead error branches that dynestyx raises first, replaced a `ValueError` expectation with the jaxtyping `TypeCheckError` for batch dims, caught that the notebook's gap had to live in the covariate column the block reads, pinned `dynestyx>=0.7.0`, and added the idle-Simulator warning filter. The code review found the Filter-forecast tracer failure and the `handlers.scope` recipe (both fixed above), the scalar-event padding, the missing controls and Filter tests, a decision-table row for the skill, and two notebook claims not backed by outputs (removed). Everything else it verified as correct: the anchor-inclusive rollout under all three handlers, the nested trace under `Predictive` (vmap and jit), `lax.map`, chunking, `device="host"`, x64 and MCMC, `Predictive`'s exclusion of deterministic sites, the control alignment, and the smoothing-moment closed form.
