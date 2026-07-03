# Design and scaffolding: integrating `numpyro_forecast` with `dynestyx`

Status: design and scaffolding only (no implementation yet). This document is the single source of truth for the work and is written to be paste-ready into [BasisResearch/dynestyx#264](https://github.com/BasisResearch/dynestyx/issues/264) and the offline chat.

## 1. Context

Issue #264 asks whether `dynestyx` (dsx) samplers can be used for `numpyro_forecast` (nf)-style forecasting, and whether the two libraries can integrate. The dsx collaborator suggested two directions: (1) use nf to evaluate a dsx model, and (2) use dsx to fit an nf-style model. This document turns both into concrete code.

The two libraries are complementary rather than overlapping. nf is a forecasting-workflow layer: a `ForecastingModel` skeleton (`time_series` / `predict`), SVI/NUTS forecasters, and a probabilistic-evaluation stack (`backtest`, `eval_crps`, `eval_coverage`). Its models are direct: the latent path is sampled per step (`level = jnp.cumsum(drift)`), so inference infers every per-step latent. dsx is a state-space inference engine built on NumPyro effect handlers: a `DynamicalModel` (initial condition, state evolution, observation model) decoupled from inference (`Simulator` / `Filter` / `Smoother` interpret `dsx.sample`). Its `Filter` marginalizes the latent state (KF/EnKF/EKF/UKF/PF) and adds the marginal log-likelihood as a NumPyro factor, so NUTS/SVI sample only parameters.

The two meet at the point where a fitted model emits forecast-sample arrays. The honest overlap is univariate structural state-space forecasting.

## 2. Locked decisions

| Decision | Choice |
| --- | --- |
| Demo model | Discrete local-level state-space model (exact linear-Gaussian), fit with `KFConfig()` |
| Notebook scope | Direction A and Direction B, both in one notebook |
| Dataset | Synthetic only, with known ground-truth `q` / `r` so both methods can be shown to recover truth |
| Adapter home | In-notebook helper (plus a throwaway prototype); no new public API, no `reference:` or test changes |
| Adapter architecture | Functional core plus a thin OOP shim, mirroring `functional.py` and `forecaster.py` |
| dsx dependency | A scoped `integration` optional-dependency extra, aggregated into `all` |

## 3. The exact model mapping (one generative process, two inference strategies)

The local-level model is the minimal case where the mapping is exact:

$$x_t = x_{t-1} + w_t, \quad w_t \sim \text{Normal}(0, q); \qquad y_t = x_t + v_t, \quad v_t \sim \text{Normal}(0, r).$$

- nf-direct: sample `drift ~ Normal(0, q)` per step, form `level = cumsum(drift)`, observe `Normal(level, r)`. Inference infers `T` drifts plus `{q, r}`.
- dsx-filter: `LTI_discrete(A=[[1]], Q=[[q**2]], H=[[1]], R=[[r**2]])` with `KFConfig()`. The Kalman filter marginalizes the level analytically; NUTS samples only `{q, r}`.

Both use identical priors (`HalfNormal` on the standard deviations `q`, `r`), so the comparison is fair. The narrative point: identical posterior predictive (up to Monte Carlo error), but dsx infers 2 latent dimensions instead of `T + 2`, giving cleaner mixing and an exact marginal likelihood.

## 4. Pinned dynestyx API patterns (verified against dsx source and tests)

These are the load-bearing facts the adapter depends on. They are verified against dsx `dynestyx/inference/filters.py`, `dynestyx/models/lti_dynamics.py`, `dynestyx/models/observations.py`, and `tests/test_filter_simulator.py` at commit `b095bda`.

1. Fit / conditioning (marginal likelihood only), no simulator, no `predict_times`:

   ```python
   def filtered():
       with Filter(filter_config=KFConfig(filter_source="cuthbert")):
           return model(obs_times=obs_times, obs_values=obs_values)
   MCMC(NUTS(filtered), ...).run(rng_key)
   ```

2. Forecast / rollout from the filtered posterior uses a nested `Simulator` inside `Filter`, with `predict_times`:

   ```python
   with DiscreteTimeSimulator(n_simulations=1):
       with Filter(filter_config=KFConfig(filter_source="cuthbert")):
           pred = Predictive(model, posterior_samples=post)(
               rng_key, obs_times=obs_times, obs_values=obs_values,
               predict_times=predict_times,
           )
   ```

3. The forecast observation site is `f_predicted_observations` (not `f_observations`). Under `Predictive` its shape is `(num_samples, n_sim, T_pred, obs_dim)`; sibling sites are `f_predicted_states` and `f_predicted_times`. `dynestyx.flatten_draws` collapses the leading `(num_samples, n_sim)` into one draws axis.

4. `LTI_discrete(A, Q, H, R, B=None, b=None, D=None, d=None, initial_mean=None, initial_cov=None)` builds the `DynamicalModel`; `q`, `r` are standard deviations in both models, squared into the covariance arrays `Q`, `R`.

5. For the linear-Gaussian model, `KFConfig(filter_source="cuthbert")` is exact. `EnKFConfig()` is the default but only approximate; the demo uses `KFConfig` for an exact comparison. Non-Gaussian observations would require `PFConfig` (documented as the boundary, out of scope here).

## 5. Adapter architecture (functional core plus OOP shim)

This mirrors nf's own split: `functional.py` holds pure primitives (`fit_svi`, `fit_mcmc`, `draw_posterior`, `forecast`, `predict_in_sample`) and `forecaster.py` wraps them in `Forecaster` / `HMCForecaster`. The dsx adapter follows the same shape and, crucially, reuses nf primitives instead of reimplementing them:

- Posterior thinning/resampling: wrap the dsx posterior dict in nf's `MCMCFit` and call nf's `draw_posterior`. That reuses the exact even-grid-thin / resample-with-replacement logic in `functional._draw_posterior_impl` and keeps behavior consistent with the built-in forecasters.
- Draws flattening: dsx's `flatten_draws`.
- `rng_key` is the first positional parameter of every randomness-consuming function (AGENTS.md), and keys are split explicitly for the thinning step and the predictive step, exactly as `functional.forecast` splits `key_post` / `key_pred`.

### 5.1 Functional core

```python
from dataclasses import dataclass

import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Filter, flatten_draws
from dynestyx.inference.filters import KFConfig

from numpyro_forecast.functional import MCMCFit, draw_posterior
from numpyro_forecast.typing import Array


@dataclass(frozen=True)
class DsxFilterFit:
    """The result of fitting a dynestyx model with a Filter plus NUTS.

    Carries everything the rollout needs: the parameter posterior and the
    train/time metadata (dsx conditions on the training observations at every
    rollout, unlike nf whose posterior already encodes the in-sample latents).
    """

    samples: dict[str, Array]     # posterior over parameters only
    t_obs: int                    # number of in-sample steps
    dt: float                     # spacing that maps integer indices -> dsx float times
    train_obs: Array              # (t_obs,) conditioning observations
    filter_config: object         # dynestyx BaseFilterConfig | None


def fit_dsx_filter(
    rng_key: Array,
    model,                        # dsx model: (obs_times, obs_values, predict_times) -> ...
    data: Array,                  # (t_obs, obs) in-sample, obs == 1 for this demo
    *,
    filter_config=None,
    dt: float = 1.0,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 1,
    progress_bar: bool = False,
) -> DsxFilterFit:
    """Fit a dynestyx DynamicalModel's parameters with a Filter (marginal likelihood) and NUTS."""
    t_obs = data.shape[-2]
    obs_times = jnp.arange(t_obs) * dt
    train_obs = data[..., 0]

    def filtered() -> None:
        with Filter(filter_config=filter_config):
            model(obs_times=obs_times, obs_values=train_obs)

    mcmc = MCMC(
        NUTS(filtered),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key)
    return DsxFilterFit(mcmc.get_samples(), t_obs, dt, train_obs, filter_config)


def _dsx_rollout(
    rng_key: Array,
    model,
    fit: DsxFilterFit,
    predict_times: Array,
    obs_values: Array,
    num_samples: int,
) -> Array:
    """Roll out `f_predicted_observations` at `predict_times`, conditioned on `obs_values`.

    Returns draws of shape ``(num_samples, len(predict_times), obs)``.
    """
    key_thin, key_pred = random.split(rng_key)
    # Reuse nf's thinning/resampling by presenting the dsx posterior as an MCMCFit.
    post = draw_posterior(key_thin, MCMCFit(samples=fit.samples), num_samples)
    obs_times = jnp.arange(fit.t_obs) * fit.dt
    with DiscreteTimeSimulator(n_simulations=1):
        with Filter(filter_config=fit.filter_config):
            pred = Predictive(model, posterior_samples=post)(
                key_pred,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )
    draws = pred["f_predicted_observations"]
    # Normalize to (num_samples, T_pred, obs): flatten (num_samples, n_sim) when present.
    if draws.ndim == 4:
        draws = flatten_draws(draws)
    return draws


def dsx_forecast(
    rng_key: Array,
    model,
    fit: DsxFilterFit,
    data: Array,                  # in-sample (t_obs, obs), for conditioning
    covariates: Array,            # (duration, cov); only its time length is used
    num_samples: int,
    *,
    batch_size: int | None = None,
) -> Array:
    """Forecast the future horizon; returns ``(num_samples, future, obs)``."""
    duration = covariates.shape[-2]
    predict_times = jnp.arange(duration) * fit.dt
    draws = _dsx_rollout(rng_key, model, fit, predict_times, data[..., 0], num_samples)
    return draws[:, fit.t_obs:, :]


def dsx_predict_in_sample(
    rng_key: Array,
    model,
    fit: DsxFilterFit,
    covariates: Array,            # (t_obs, cov)
    num_samples: int,
    *,
    batch_size: int | None = None,
) -> Array:
    """In-sample posterior predictive over the observed window; ``(num_samples, t_obs, obs)``."""
    predict_times = jnp.arange(fit.t_obs) * fit.dt
    draws = _dsx_rollout(rng_key, model, fit, predict_times, fit.train_obs, num_samples)
    return draws[:, : fit.t_obs, :]
```

### 5.2 OOP shim (satisfies nf's forecaster protocol)

```python
class DynestyxForecaster:
    """Fit a dynestyx DynamicalModel with a Filter, exposing nf's forecaster protocol.

    Same external contract as ``numpyro_forecast.Forecaster`` / ``HMCForecaster``:
    ``__call__(rng_key, data, covariates, num_samples) -> (num_samples, future, obs)``,
    plus ``predict_in_sample`` and a scalar ``params`` dict for backtest.
    """

    def __init__(
        self,
        rng_key: Array,
        model,
        data: Array,
        covariates: Array,
        *,
        filter_config=None,
        dt: float = 1.0,
        num_warmup: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 1,
        progress_bar: bool = False,
    ) -> None:
        self.model = model
        self._fit = fit_dsx_filter(
            rng_key, model, data,
            filter_config=filter_config, dt=dt,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=progress_bar,
        )
        self.posterior_samples = self._fit.samples
        # Scalar summaries so backtest's _scalar_params records something useful.
        self.params = {k: v.mean() for k, v in self._fit.samples.items() if v.ndim == 1}

    def __call__(self, rng_key, data, covariates, num_samples, *, batch_size=None):
        return dsx_forecast(rng_key, self.model, self._fit, data, covariates,
                            num_samples, batch_size=batch_size)

    def predict_in_sample(self, rng_key, covariates, num_samples, *, batch_size=None):
        return dsx_predict_in_sample(rng_key, self.model, self._fit, covariates,
                                     num_samples, batch_size=batch_size)
```

The adapter is defined in-notebook, but structured so it could later be lifted verbatim into a `numpyro_forecast/contrib/dynestyx.py` module (which would then need `reference:` and `test_docs_reference.py` updates). That promotion is out of scope for now.

## 6. Model builders

nf-direct (functional model; `LocScaleReparam` on `drift` avoids the random-walk funnel so NUTS over all latents is a fair baseline):

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast import forecasting_model
from numpyro_forecast.functional import time_series, predict


def local_level_nf(h, covariates):
    q = numpyro.sample("q", dist.HalfNormal(1.0))
    r = numpyro.sample("r", dist.HalfNormal(1.0))
    drift = time_series(h, "drift", lambda: dist.Normal(0.0, q),
                        reparam=LocScaleReparam())
    level = jnp.cumsum(drift, axis=-2)
    predict(h, dist.Normal(0.0, r), level)


nf_model = forecasting_model(local_level_nf)
```

dsx model builder (a factory so `backtest` can call `model_fn()` per window):

```python
def make_local_level_model():
    def model(obs_times=None, obs_values=None, predict_times=None):
        q = numpyro.sample("q", dist.HalfNormal(1.0))
        r = numpyro.sample("r", dist.HalfNormal(1.0))
        dynamics = dsx.LTI_discrete(
            A=jnp.array([[1.0]]), Q=jnp.array([[q**2]]),
            H=jnp.array([[1.0]]), R=jnp.array([[r**2]]),
            initial_mean=jnp.zeros(1), initial_cov=jnp.array([[10.0**2]]),
        )
        return dsx.sample("f", dynamics, obs_times=obs_times,
                          obs_values=obs_values, predict_times=predict_times)
    return model
```

## 7. Consistency checklist (contract alignment, verified against nf source)

`backtest._run_window` in `numpyro_forecast/evaluate.py` drives forecasters generically:

- It constructs the forecaster as `forecaster_fn(key_fit, model_fn(), train_data, train_covariates, **options)`. The adapter's `__init__(rng_key, model, data, covariates, *, ...)` matches, and `model_fn = make_local_level_model` returns the dsx model. Options carry `filter_config`, `dt`, and the NUTS budget.
- It forecasts as `forecaster(key_forecast, train_data, test_covariates, num_samples, batch_size=batch_size)`. The adapter's `__call__` matches; it slices `[t_obs:]` so the returned length is `test_covariates.shape[-2] - train_data.shape[-2] = t2 - t1`, exactly the length of `truth = data[t1:t2]`.
- For `eval_train=True` it calls `forecaster.predict_in_sample(key, train_covariates, num_samples, batch_size=...)`. The adapter matches. `eval_train` is optional; if the in-sample rollout proves degenerate it is simply left `False` in the notebook.
- `_scalar_params` reads `forecaster.params` and keeps size-1 entries; `self.params` holds posterior means of `q`, `r` (scalars), so window parameter estimates are recorded.

Time-convention note: `obs_times` restart at 0 each window (`jnp.arange(t_obs) * dt`). This is correct here because the local-level SSM is time-invariant; absolute-calendar-time models would instead need a per-window `t0` offset.

## 8. Deliverables and file map

```
DYNESTYX_INTEGRATION_DESIGN.md                       # this document (repo root)
<scratchpad>/dynestyx_adapter_prototype.py           # throwaway prototype (NOT committed)
docs/examples/dynestyx_local_level.py                # jupytext source (authoring only, deleted after)
docs/examples/dynestyx_local_level.ipynb             # committed notebook (with outputs)
pyproject.toml                                       # add `integration` optional-dependency extra
```

No `great-docs.yml` `reference:` change (no new public symbols); its `sections: docs/examples` entry auto-registers the notebook page. `tests/test_docs_reference.py` stays green.

## 9. Notebook outline (`docs/examples/dynestyx_local_level.ipynb`)

Authored as jupytext `py:percent`, executed with a scratchpad throwaway kernelspec (the default `python3` kernel points at the wrong venv), committed with outputs, then the `.py` is deleted. Repo writing rules apply: ArviZ >= 1.0 (`az.plot_lm`, both 50% and 94% HDI), title / legend / labels on every plot, LaTeX `r"$94\%$ HDI"`, explicit `\text{Normal}(\mu, \sigma)`, American English, no em-dashes, integer underscores.

1. Imports, rc params, `rng_key`.
2. Markdown: motivation (issue #264), what dsx and nf are, the two directions, the local-level testbed.
3. Synthetic data with known `q_true`, `r_true`; `T = 200`, `future = 40`; train/test split; `covariates = jnp.zeros((T + future, 1))` (horizon only); plot.
4. Markdown: Direction B, one model and two inference strategies.
5. nf-direct: `local_level_nf` fit with `HMCForecaster`; record wall-time and the latent count `T + 2`; forecast the horizon.
6. dsx model builder plus the functional-core and OOP-shim adapter cells.
7. Fit dsx via `DynestyxForecaster` (NUTS over `{q, r}` only); record wall-time; forecast.
8. Compare: posteriors of `q`, `r` against truth (with reference lines); forecast overlay via `az.plot_lm` (both methods, 50% and 94% HDI); a CRPS / coverage / runtime / latent-count table.
9. Markdown: Direction A. Show `eval_crps` / `eval_coverage` on the dsx forecasts, then `nf.backtest(rng, data, covariates, model_fn=make_local_level_model, forecaster_fn=DynestyxForecaster, forecaster_options={"filter_config": KFConfig(filter_source="cuthbert"), "dt": 1.0, ...})`; plot rolling CRPS and coverage.
10. Markdown: takeaways and caveats (overlap is univariate structural SSM; nf owns covariates and seasonality, dsx owns nonlinear dynamics and partial observations); link this document.

## 10. Dependency change (`pyproject.toml`)

Add a scoped extra in `[project.optional-dependencies]` and aggregate it into `all` (keeps the heavy dsx stack, which pulls `cd_dynamax`, `cuthbert`, `effectful`, `equinox`, isolated; committed notebooks are not re-executed at docs build, so dsx is authoring-only):

```toml
integration = ["dynestyx>=<pinned-at-install>"]
all = ["numpyro_forecast[dataframes,dev,docs,integration]"]
```

The concrete version is resolved at install time.

## 11. Risks and how they are handled

1. Dependency conflict (highest risk): dsx may clash with nf's `jax>=0.10` / `numpyro>=0.21` pins. If `uv sync` cannot resolve, keep the `integration` extra as documentation and author/prototype in an isolated scratch venv; the notebook still ships because it is not re-executed at build.
2. Rollout context (now pinned in section 4, but backend-version-sensitive): the prototype asserts that `f_predicted_observations` exists and has the expected shape before any notebook cell is written. If a backend variation changes the site name or nesting, the prototype catches it first.
3. Scalar-observation shapes (`obs_values` is `(T,)`, `f_predicted_observations` carries a trailing obs axis): normalized in `_dsx_rollout`; the prototype asserts final shape `(num_samples, future, 1)`.
4. In-sample rollout semantics differ from nf's (dsx re-filters on the training obs rather than reusing sampled latents): `predict_in_sample` is best-effort and `eval_train` defaults to `False` if it misbehaves.

## 12. Verification (run during the build phase, in order)

1. `uv sync --extra integration` (or `--all-extras`) resolves and installs dsx; on failure, fall back per risk 1 and record it.
2. Run `<scratchpad>/dynestyx_adapter_prototype.py`: recovered `q`, `r` within a couple of posterior standard deviations of truth; `eval_crps` finite; one `nf.backtest` fold returns a `BacktestResult`; forecast shape is `(num_samples, future, 1)`.
3. Author and execute the notebook with jupytext using the scratchpad kernelspec; confirm every cell runs and figures embed; delete the `.py`.
4. `uv run ruff check docs/examples/dynestyx_local_level.ipynb` and `uv run ruff format --check docs/examples/dynestyx_local_level.ipynb`.
5. `uv run pytest tests/test_docs_reference.py` stays green (no public-API change; notebooks are not collected because `testpaths = ["tests"]`).
6. Optional: `make docs` renders the new example page.
```
