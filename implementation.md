# Implementation Plan — `numpyro_forecast`

## Context

`numpyro_forecast` is a greenfield repo (only `README.md`, `LICENSE`, `.gitignore`, GitHub remote at `juanitorduz/numpyro_forecast`). The goal is a **JAX/NumPyro port of Pyro's `pyro.contrib.forecast` module** (`ForecastingModel`, `Forecaster`, `backtest`, evaluation metrics), keeping the public API close to Pyro while implementing the train-vs-forecast mechanism the idiomatic NumPyro way (a random-walk cumulative sum + a `future`-horizon argument with separate in-sample/forecast latent sites + `Predictive`) instead of porting Pyro's poutine messengers.

Success = reproducing two examples:
1. Pyro's univariate forecasting tutorial (already ported by the user at juanitorduz.github.io/numpyro_forecasting-univariate) — BART weekly ridership, Fourier seasonality, StudentT likelihood, random-walk level, SVI, CRPS.
2. NumPyro's hierarchical forecasting tutorial (num.pyro.ai) — BART hourly OD counts, per-station drift/seasonality/affinity, SVI, CRPS.

Both load data via `numpyro.examples.datasets.load_bart_od()` — no data needs to be vendored.

Infra mirrors the sibling repo `/Users/juanitorduz/Documents/probcast` (hatchling + uv, ruff lint/format, type checking, pytest+cov, pre-commit hooks run via **prek**, CI matrix on 3.12/3.13, Claude GitHub Actions). Python `>=3.12`.

**Coding standards (hard requirements):**
- **Every function** (public and private) has complete **input and return type hints**, checked with **`ty`** (astral-sh/ty) via the `astral-sh/ty-pre-commit` hook — not mypy.
- **Every public function/class** has a **numpy-style docstring** (Parameters / Returns / etc.), enforced by ruff `D`/`DOC` with `pydocstyle` convention = numpy.

**Confirmed decisions (via AskUserQuestion):** idiomatic NumPyro mechanism; both SVI + MCMC backends; full probcast toolchain parity; examples as notebooks + reduced-step smoke tests in CI.

---

## Package layout (flat, mirrors Pyro module names)

```
numpyro_forecast/
├── numpyro_forecast/
│   ├── __init__.py          # public API exports
│   ├── forecaster.py        # ForecastingModel (ABC), predict, Forecaster (SVI), HMCForecaster (MCMC), _BaseForecaster
│   ├── evaluate.py          # backtest(), eval_crps / eval_mae / eval_rmse
│   ├── metrics.py           # empirical CRPS core (jnp, jit/vmap-able)
│   ├── util.py              # zero_data_like, shift_loc, prefix_condition, concat_future, fourier_features, periodic_repeat
│   ├── typing.py            # Array / Metric / ModelFactory / ForecasterFactory aliases (breaks import cycles)
│   ├── datasets.py          # load_bart_od wrapper + univariate/hierarchical splits (time-at-(-2) layout)
│   ├── models/              # validated example models as importable library code
│   │   ├── __init__.py
│   │   ├── univariate.py
│   │   └── hierarchical.py
│   └── py.typed
├── tests/
│   ├── conftest.py          # rng_key, synthetic univariate/panel fixtures, fast SVI/MCMC params
│   ├── test_forecaster.py   # ABC contract, predict() train vs future-forecast, guide-not-resized, shapes
│   ├── test_evaluate.py     # backtest windows + metrics
│   ├── test_metrics.py      # CRPS vs known closed-form cases
│   ├── test_util.py         # shift_loc, fourier/periodic features, prefix_condition
│   └── test_examples.py     # reduced-step smoke tests on BART data (both models)
├── examples/
│   ├── forecasting_univariate.ipynb
│   └── hierarchical_forecasting.ipynb
├── .github/workflows/{ci.yml, claude.yml, claude-code-review.yml}
├── pyproject.toml
├── .pre-commit-config.yaml
├── CLAUDE.md, CONTRIBUTING.md, README.md, LICENSE, .gitignore
└── (uv.lock generated)
```

---

## Source-verified mechanism (what the port must reproduce) — and the loophole, closed

I read Pyro's actual source (`forecaster.py`, `util.py`, `evaluate.py`, `__init__.py`) and the **ground-truth** notebook cells of both examples (confirmed: train via `svi.run(rng, steps, covariates_train, y_train)` at length `t`, then `posterior(rng, covariates)` at full length `t+f`, with `covariates = jnp.zeros_like(y)` dummy zeros, no `future` arg, no masking). Analysis:

- **Why Pyro needs its machinery, and what dissolves.** `predict()`'s `reshape_batch`/unsqueeze exists only because (Pyro's own comment) "Pyro [does not use] name dimensions internally" — NumPyro's plate stack removes the need, so it is **dropped**. `PrefixReplayMessenger` (posterior on the in-sample latent prefix + prior on the forecast suffix) and `PrefixConditionMessenger`/`prefix_condition` (condition `(t+f)`-noise on the `t`-prefix) exist to support **fit-once / forecast-any-`f`** plus marginalized `GaussianHMM` noise.
- **The loophole (and why the examples' literal pattern is fragile).** Training at length `t` then re-running the guide/model at length `t+f` makes the in-sample `drift`/`time` plate **resize**, which `AutoNormal` does not support (its variational-param shapes are frozen from the training trace). If a framework silently re-sampled the whole `drift` trajectory from the prior at predict time, forecasts would **not condition on the data**. So the literal "train short, predict long" relies on autoguide/`Predictive` resize behavior we will not depend on.
- **Loophole closed — the robust, faithful, idiomatic mechanism (a `future` argument + separate in-sample/forecast latent sites + cumulative-sum continuation).** This is the pattern NumPyro's own SGT / Holt-Winters time-series tutorials use, and it reproduces Pyro's `PrefixReplay` semantics exactly with plain site-name separation:
  - in-sample time latents are sampled under `numpyro.plate("time", t)` with the **fixed** name (e.g. `drift`) → identical shape in train and predict, so **`AutoNormal` never resizes**;
  - when `future > 0`, the forecast-horizon latents are sampled under a **separate** site (e.g. `drift_future`, plate size `future`); this site is **not in the guide**, so `Predictive` draws it from the **prior** — i.e. posterior prefix + prior suffix, exactly like `PrefixReplay`;
  - a cumulative sum runs over the concatenated `[drift, drift_future]` so the forecast level **continues from the inferred end-level** → forecasts are correctly conditioned on the data (a `jnp.cumsum` over the time axis; mathematically identical to a `jax.lax.scan` random-walk accumulation, and the form the shipped models use);
  - this restores Pyro's **fit-once / forecast-any-`f` UX** (pass `future=f` at call time; no refit) while staying pure/functional.

## Core library design

### Array & API conventions (one contract, both examples conform)
- **Array layout = Pyro's:** time at axis **`-2`**, the single observation/event dim at **`-1`**, arbitrary batch dims to the left. `data: Float[Array, "*batch t obs"]`, `covariates: Float[Array, "*batch duration cov"]` with `duration = t + future`. The univariate example is `(t, 1)` natural; the hierarchical example is **re-expressed** from the tutorial's `(origin, destin, time)` to `(origin, time, destin)` so origin is batch `-3`, time `-2`, destin event `-1` — same generative structure/results, one consistent contract. Stated as an explicit, documented design decision.
- **`future` is never an argument** — it is derived from shapes (`future = covariates.shape[-2] - data.shape[-2]`), exactly as Pyro derives `t_obs`/`t_cov`. The public model signature stays Pyro-identical: `model(self, zero_data, covariates)`.
- **Type aliases (in `numpyro_forecast/typing.py`):** `Array = jax.Array`; `Metric = Callable[[Array, Array], float]`; `ModelFactory = Callable[[], ForecastingModel]`; `ForecasterFactory` = a `typing.Protocol` with `__call__(model: ForecastingModel, data: Array, covariates: Array, **opts: Any) -> _BaseForecaster`. Array params annotated with `jaxtyping` shape strings; everything else fully hinted for `ty`.

### `ForecastingModel` (ABC) — `forecaster.py`
Keeps Pyro's contract: subclass and implement `model(self, zero_data, covariates) -> None`, calling `self.predict(...)` **exactly once**. The framework wraps the instance into a **pure** NumPyro model fn `fn(covariates: Array, data: Array | None = None) -> None` (no module-global state; `data` threaded via a per-call closure set synchronously during the trace — safe under `jit`/`vmap`). State exposed to subclasses as read-only properties: `self.duration: int`, `self.t_obs: int`, `self.future: int`.

- `zero_data: Array = util.zero_data_like(data, covariates)` → zeros of shape `(*batch, duration, obs)`; exposes shape/dtype only (so `prediction` can't depend on observed values, matching Pyro). Tolerates an empty covariate channel (`cov == 0`, Pyro's `empty(duration, 0)` convention).
- **In-sample vs forecast latents** — `self.time_series(name: str, dist_fn: Callable[[], dist.Distribution], *, reparam: Reparam | None = None) -> Array`: samples `name` under `numpyro.plate("time", self.t_obs, dim=-2)` and, when `self.future > 0`, `f"{name}_future"` under a size-`future` plate; applies `reparam` (e.g. `LocScaleReparam`) to **both** sites via `numpyro.handlers.reparam`; returns the concatenation along axis `-2`. This is the crux that keeps guide shapes fixed and makes `Predictive` draw the suffix from the prior. (No `future` parameter — read from state.)
- `self.predict(noise_dist: dist.Distribution, prediction: Array) -> None`: `noise_dist` is **zero-centered** (model passes `loc=0`), `prediction` the deterministic mean over `duration`. We **drop Pyro's event_dim 0/1/2 + `reshape_batch` logic** (plates align dims). Behavior:
  - re-center to `obs_dist` via `util.shift_loc` (single-dispatch over Normal/StudentT/Independent; TransformedDistribution is Future work);
  - **training** (`future == 0`): `numpyro.sample("obs", obs_dist, obs=data)`;
  - **forecasting** (`future > 0`): observe prefix `numpyro.sample("obs", obs_dist[..., :t, :], obs=data)` and sample suffix `numpyro.sample("obs_future", prefix_condition(obs_dist, data))`, exposed as `numpyro.deterministic("forecast", ...)`.
- `prefix_condition` (single-dispatch, `util.py`): for i.i.d.-in-time noise = the trivial future-slice (`obs_dist[..., t:, :]`); for time-correlated noise (MultivariateNormal/Independent/Transformed) = the genuine Gaussian conditional. `GaussianHMM` is the documented v2 path.

### `Forecaster` (SVI) and `HMCForecaster` (MCMC) — `forecaster.py`
Thin subclasses of a shared `_BaseForecaster`, mirroring Pyro's two-class split but **stateless/functional**. Pyro asserts `data.size(-2) == covariates.size(-2)` at fit; we keep that (**fit is in-sample**, `future == 0`, guides/samples sized to `t`).
- `Forecaster(model: ForecastingModel, data: Array, covariates: Array, *, guide: AutoGuide | None = None, optim: _NumPyroOptim | None = None, num_steps: int = 1001, num_particles: int = 1, rng_key: Array) -> None`: defaults `guide=AutoNormal(model_fn)`, `optim=Adam(0.01)`; runs `SVI(...).run(...)`; stores `guide`, `params: dict[str, Array]`, `losses: Array`.
- `HMCForecaster(model, data, covariates, *, num_warmup=1000, num_samples=1000, num_chains=1, rng_key) -> None`: runs `MCMC(NUTS(...))`; stores `posterior_samples: dict[str, Array]`. No `torch.multinomial` replay — `Predictive(posterior_samples=...)` draws directly.
- Both: `__call__(self, data: Array, covariates: Array, num_samples: int, *, rng_key: Array, batch_size: int | None = None) -> Float[Array, "sample *batch future obs"]` — asserts `data.size(-2) < covariates.size(-2)`, returns the `"forecast"` site. Matches Pyro's `forecaster(data, covariates_extended, num_samples)` signature.
  - **Two-step forecast (closes a guide-retrace sub-loophole):** an `AutoNormal` guide re-traces the model each call, so running the guide at `future>0` would demand variational params for the absent `_future` sites → error. So (mirroring Pyro trace-guide→replay-model): (1) `samples = guide.sample_posterior(rng, params, sample_shape=(num_samples,))` at in-sample shapes; (2) `Predictive(model_fn, posterior_samples=samples)(rng, data, covariates)` — prefix sites substituted, `_future` sites from the prior. MCMC skips step 1.
  - `batch_size` chunking (Pyro's OOM guard) → `jax.lax.map` over RNG-key chunks.

### `evaluate.py`
- `backtest(data: Array, covariates: Array, model_fn: ModelFactory, *, forecaster_fn: ForecasterFactory = Forecaster, metrics: Mapping[str, Metric] = DEFAULT_METRICS, transform: Callable[[Array, Array], tuple[Array, Array]] | None = None, train_window: int | None = None, min_train_window: int = 1, test_window: int | None = None, min_test_window: int = 1, stride: int = 1, seed: int = 1234567890, num_samples: int = 100, batch_size: int | None = None, forecaster_options: Mapping[str, Any] | Callable[..., Mapping[str, Any]] = {}) -> list[BacktestResult]`: same window semantics as Pyro (expanding when `train_window is None`; `t0<t1<t2`). **No `clear_param_store()`** — each window is a pure `forecaster_fn(...)` call. RNG seeding via explicit `random.PRNGKey(seed)` per window (replaces `pyro.set_rng_seed`).
- Returns a list of `@dataclass(frozen=True) BacktestResult` with typed fields `t0/t1/t2/seed/num_samples: int`, `train_walltime/test_walltime: float`, `metrics: dict[str, float]`, `params: dict[str, float]` — typed and directly assertable in tests (cleaner than Pyro's flat untyped dict; a `.to_dict()` keeps Pyro-style access).
- Metrics (`Metric` type, ported to `jnp`): `eval_mae(pred, truth) -> float` (sample median), `eval_rmse(pred, truth) -> float` (sample mean), `eval_crps(pred, truth) -> float` (empirical). `DEFAULT_METRICS = {"mae": eval_mae, "rmse": eval_rmse, "crps": eval_crps}`.

### `metrics.py` — empirical CRPS
Port `pyro.ops.stats.crps_empirical` to JAX. Public, exported, jit/vmap-friendly: `crps_empirical(pred: Float[Array, "sample *batch"], truth: Float[Array, "*batch"]) -> Float[Array, "*batch"]` = `E|X − y| − ½·E|X − X'|` via the sorted-sample O(n log n) form (`jnp.sort` + weighted diffs). `eval_crps` (in `evaluate.py`) = `float(crps_empirical(pred, truth).mean())`.

### `util.py`, `typing.py`, `datasets.py`
- `util.py`: `zero_data_like`, `shift_loc` (single-dispatch), `slice_time` (single-dispatch), `prefix_condition` (single-dispatch), `concat_future` (the low-level prefix/suffix concat used by `ForecastingModel.time_series`), `fourier_features` and `periodic_repeat` (repeat a `24×7` season to `t_max`), all `jaxtyping`-typed. (`time_series` itself is a **method** of `ForecastingModel`, since it reads model state.) (`pad_to`/`time_mask` are not shipped — neither target example needs masking; see Future work.)
- `typing.py`: the `Array`/`Metric`/`ModelFactory`/`ForecasterFactory` aliases (avoids import cycles between `forecaster.py` and `evaluate.py`).
- `datasets.py`: thin `load_bart_od()` wrapper returning typed arrays + the two splits, **arranged to the `(*, time, obs)` convention** — univariate weekly `log` totals shaped `(t, 1)` (417/52); hierarchical `jnp.log1p(permute_dims(counts, (1, 0, 2)))` → `(origin, time, destin)` with `T0 = T1 − 24*90`, `T1 = T2 − 24*7*2`.

---

## Pyro → NumPyro/JAX porting map

| Pyro construct | Port |
| --- | --- |
| `torch.Tensor` ops, `.cpu().item()` | `jnp` ops, `float(...)` |
| `pyro.plate("time", T, dim=-1)` | `numpyro.plate("time", T)` |
| `GaussianHMM` noise + `.prefix_condition` | cumulative-sum (random-walk) latent level + i.i.d. obs noise (a `jnp.cumsum`, equivalent to a `jax.lax.scan` accumulation) |
| `PrefixReplayMessenger` (posterior prefix + prior suffix) | **separate `name`/`name_future` sites** (horizon derived from shapes) — guide covers prefix, `Predictive` draws suffix from prior; cumulative sum concatenates |
| `PrefixConditionMessenger` + `prefix_condition` | separate `obs`/`obs_future` sites (i.i.d. ⇒ trivial time-slice); `singledispatch prefix_condition` kept for correlated noise (v2) |
| `predict()` `reshape_batch`/unsqueeze (no named dims) | **dropped** — plate stack aligns dims |
| `DCTAdam`, Haar/DCT `time_reparam` | `LocScaleReparam` (as both examples use); DCTAdam out of scope |
| `pyro.clear_param_store()` per backtest window | **dropped** — each window is a pure call returning its own params |
| manual SVI loop / `Trace_ELBO` | `numpyro.infer.SVI(...).run(...)`, `Trace_ELBO` |
| `HMCForecaster` NUTS + `torch.multinomial` replay | `numpyro` `MCMC(NUTS)` + `Predictive(posterior_samples=...)` |
| guide-trace `poutine.replay` | `Predictive(model, guide=guide, params=params)` |
| `batch_size` OOM retry loop | `jax.lax.map` / `vmap` over RNG-key chunks |
| `self._data` instance state (reset in `finally`) | per-call closure threading the data into `predict` |

## Functional / JAX idioms (the "FP flavor")
- **Pure model functions**, explicit immutable SVI/MCMC state, no global param store.
- **Explicit `PRNGKey` threading** everywhere via `jax.random.split`; forecasters take `rng_key`.
- **`jnp.cumsum`** for the random-walk latent level/drift (both examples; equivalent to a `jax.lax.scan` accumulation); **`jax.vmap`** for batched forecast sampling and for vectorizing `eval_crps` over trailing dims; optional **`vmap`/`lax.map` over equal-length backtest windows** as a JAX-native alternative to Pyro's Python loop.
- **`jax.jit`** on hot pure helpers (CRPS, transition fns); **`functools.singledispatch`** for `shift_loc`/`prefix_condition` (functional dispatch instead of messenger subclassing).
- **`jaxtyping`** annotations (`Float[Array, "time feature"]`) on array params, consistent with the examples and enforcing the type-hint standard.

---

## Example models — `numpyro_forecast/models/` (reused by notebooks + tests)

Ship the two validated models as library code (so both notebooks and smoke tests import them), each expressed through the `ForecastingModel`/`predict` API and using `self.time_series("drift", ...)` so they forecast via the `future` mechanism (no plate resize):
- **`univariate`** (reproduces the blog): `bias + level_t + (weight·covariates)`, random-walk `drift`→`level` via a cumulative sum + `LocScaleReparam(centered)`, `StudentT(df=nu, loc=mu, scale=sigma)`; priors exactly as verified (`bias~N(0,10)`, `weight~N(0,0.1)`, `drift_scale~LogNormal(-20,5)`, `nu~Gamma(10,2)`, `sigma~LogNormal(-5,5)`, `centered~U(0,1)`). Covariates = `fourier_features` (52 terms). SVI/`AutoNormal`/`Adam(5e-3)`; CRPS on weekly BART totals (417/52).
- **`hierarchical`** (reproduces the tutorial, re-expressed in `(origin, time, destin)` layout so time is `-2`): `origin` (dim −3) / `destin` (dim −1) / `hour_of_week` plates, per-destination `drift` (reparam'd) via `self.time_series` → cumulative-sum level, `origin_seasonal+destin_seasonal` via `periodic_repeat`, `pairwise` affinity, `scale=origin_scale+destin_scale`, `Normal` likelihood; SVI; CRPS across station pairs (90-day/2-week). Same generative structure/results as the tutorial.

---

## Infrastructure (full probcast parity)

- `pyproject.toml`: hatchling build; `requires-python = ">=3.12"`; deps `jax`, `numpyro`, `arviz>=1.0.0`, `matplotlib`, `tqdm`, `jaxtyping`; `dataframes` extra (`pandas`/`polars`) for BART loading; `dev` extra (`pytest`, `pytest-cov`, `pytest-xdist`, `ruff`, `prek`, `ty`, `nbmake`); ruff config (line 99, `D`/`DOC` with numpy convention, select B/D/DOC/E/F/I/RUF/S/UP/W, per-file test ignores), pytest cov on `numpyro_forecast`, `[tool.ty]` config. Adapt from probcast's `pyproject.toml`, renaming `probcast`→`numpyro_forecast` and replacing the mypy config with `ty`.
- `.pre-commit-config.yaml`: ruff-check/ruff-format, **`ty` via `astral-sh/ty-pre-commit`** (replaces mirrors-mypy), standard hooks; install/run via **prek** (`prek install`, `prek run --all-files`) — prek is a drop-in pre-commit runner using the same config file.
- `.github/workflows/ci.yml`: `lint` job (`uv sync --all-extras`, `ruff check`, `ruff format --check`, `uv run ty check numpyro_forecast/`) then `test` job matrix 3.12/3.13 (`pytest`, includes reduced-step example smoke tests). Plus `claude.yml` + `claude-code-review.yml` adapted from probcast.
- uv for env management; commit `uv.lock`. Add `py.typed`. Copy `conftest.py` fixture style from probcast.

---

## Staged execution

1. **Scaffold + toolchain.** Create package/test/example dirs, `pyproject.toml`, `.pre-commit-config.yaml`, CI workflows, `py.typed`, `CLAUDE.md`/`CONTRIBUTING.md`, update `README.md`. `uv sync --all-extras`; `prek install`. Verify `ruff`, `ty`, empty `pytest` run clean.
2. **Core library (TDD).** `metrics.py` → `util.py` (incl. `time_series`, `shift_loc`, `prefix_condition`) → `ForecastingModel`/`predict` (`future` mechanism) → `Forecaster`/`HMCForecaster` → `evaluate.backtest`. Unit tests for shapes, **fit-once/forecast-any-`f` (guide not resized; suffix from prior)**, forecast conditions on data, CRPS correctness.
3. **Univariate model + example.** `models/univariate.py` + notebook reproducing the blog; reduced-step smoke test asserting forecast shape + CRPS in a sane band.
4. **Hierarchical model + example.** `models/hierarchical.py` + `datasets.py` splits + notebook reproducing the tutorial; reduced-step smoke test.
5. **Polish.** Docstrings, README usage, ensure CI green on both Python versions.

---

## Verification

- `uv sync --all-extras` succeeds; `prek run --all-files` clean.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check numpyro_forecast/` pass (zero `ty` errors; all functions fully type-hinted, all public APIs have numpy docstrings).
- `uv run pytest` green (unit + smoke), incl. `tests/test_examples.py` running both models on `load_bart_od()` with reduced SVI steps and asserting forecast shapes + finite CRPS comparable to the published tutorials.
- `uv run pytest --nbmake examples/` executes both notebooks (reduced steps) without error.
- Manual: full notebooks reproduce CRPS values close to the published univariate (~matching Pyro) and hierarchical tutorials.

---

## Resolved design risks (loopholes closed)
- **AutoNormal shape resize** (guide trained on `t` cannot be re-run at `t+f`) → in-sample latents keep a fixed site name/shape (`plate("time", t)`); the forecast horizon uses **separate `_future` sites** → the guide is never resized. Pyro's fit-once / forecast-any-`f` UX is **preserved**.
- **In-sample latent trajectory must inform the forecast** → `Predictive` substitutes the guide's posterior for the prefix sites and draws the `_future` sites from the prior; a cumulative sum concatenates so the forecast continues from the inferred end-level (≡ Pyro `PrefixReplay`). Forecasts are provably conditioned on the data — not re-sampled from the prior.
- **Partial / autoregressive observation noise** → i.i.d. case via separate `obs`/`obs_future` sites (trivial slice); time-correlated case via the ported single-dispatch `prefix_condition` (genuine Gaussian conditional).
- **No global param store** → backtest windows and forecasters are pure; nothing to `clear`.
- **`covariates = jnp.zeros_like(y)` in the examples** is a dummy placeholder (models without real covariates) — `predict`/`zero_data_like` must tolerate a zero/empty covariate channel, matching Pyro's `torch.empty(duration, 0)` convention.

## Public API (`numpyro_forecast/__init__.py`)
Export the exact Pyro surface for drop-in familiarity — `ForecastingModel`, `Forecaster`, `HMCForecaster`, `backtest`, `eval_crps`, `eval_mae`, `eval_rmse` — plus our additions `BacktestResult`, `DEFAULT_METRICS`, `prefix_condition`, `crps_empirical`, and the `models`/`datasets` submodules. `__all__` is set explicitly.

## Future work
- Marginalized `GaussianHMM`/correlated observation noise via the ported `prefix_condition` + an explicit NumPyro `PrefixReplay`-style messenger (only needed for non-i.i.d. observation noise). The cumulative-sum random-walk + `future` mechanism covers both target examples.
- DCTAdam / Haar–DCT time reparameterization (use `LocScaleReparam` as the examples do).
- `TransformedDistribution` support in `shift_loc`/`slice_time`/`prefix_condition` (the shipped single-dispatch covers `Normal`/`StudentT`/`Independent`, which both target examples use).
- `pad_to`/`time_mask` masking helpers (not needed by either target example).
