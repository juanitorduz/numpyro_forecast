# Refactor proposal: a functional NumPyro forecasting library (issue #75)

Status: **proposal for discussion, nothing here is implemented yet.** This is the plan for addressing the API feedback in [#75](https://github.com/juanitorduz/numpyro_forecast/issues/75), rewritten after @theorashid's review of the first draft. Feedback welcome, especially on the `ssoe` design, the naming decisions, and the open questions at the bottom.

## Context

Issue #75 reviews the package API. Core critique: the package feels like a framework (Forecaster classes, fit wrappers, model wrapper) when its real value is a small set of model-building functions (the `_future`-site trick, prefix conditioning with the exact MVN conditional, horizon bookkeeping) that work inside any plain NumPyro model under standard NumPyro inference. It also identifies a missing third building block: six of the thirteen example notebooks (arma, exponential_smoothing_state_space, croston, tsb, availability_tsb, censored_demand) hand-roll the same scan-based recursion with a hand-made future-error site and a hand-registered `"forecast"` deterministic. The issue counted five; censored_demand landed afterwards, and its two-error form is the one to check the `ssoe` signature against, since it is the least obviously covered.

The first draft of this document kept the OOP API and the `fit_*` wrappers as a maintainer constraint. That constraint is withdrawn. The review made the case that a Pyro-shaped class hierarchy is the wrong thing to be faithful to, and it is more valuable to integrate cleanly with the JAX ecosystem than to ease a Pyro migration that few users will make. So:

- **No OOP.** `Forecaster`, `HMCForecaster`, `PathfinderForecaster`, and `ForecastingModel` are removed. Inference method is independent of the model.
- **No fit wrappers.** `fit_svi` and `fit_mcmc` are removed. They hide two lines of standard NumPyro from the user for no gain.
- **The word "primitive" goes.** See the next section.

`numpyro_forecast` becomes a faithful port of the *ideas* of `pyro.contrib.forecast` (the `_future`-site trick, prefix conditioning, horizon bookkeeping, backtesting) into a functional, JAX-native library. It is not a port of Pyro's class hierarchy. Inference is whatever plain NumPyro you write.

Per the review's suggestion, the first implementation PR is a **draft PR that strips the package to the strictest reading of the above**. Only after reading the rewritten notebooks do we decide what conveniences, if any, come back. This document says what that draft removes, what it deliberately keeps, and why.

## Terminology: these are not primitives

`numpyro.primitives` means something specific: `sample`, `param`, `deterministic`, `plate`, `factor`, the statements that emit messages to the effect-handler stack and that handlers intercept. What this package provides is not that, and it is not effect handlers either. `innovations`, `markov_series`, `ssoe`, and `predict` are **plain Python functions that call `numpyro.sample` and `numpyro.deterministic` on your behalf** inside your model function, sizing the sites from a `Horizon` value and splitting the in-sample window from the forecast horizon.

Calling them "primitives" mislabels them for exactly the audience most likely to read the docs. The package-wide vocabulary becomes **"model building blocks"** in headings and "model functions" in prose, and the docs say in one sentence what they are and are not. This sweeps through this document, the README, every docstring in the models module, the `great-docs.yml` section titles, and `AGENTS.md`.

## What the package stops doing

| Removed | What you write instead |
|---|---|
| `forecaster.py` in full: `ForecastingModel`, `_BaseForecaster`, `Forecaster`, `HMCForecaster`, `PathfinderForecaster` | a plain `(covariates, data=None)` model function that derives its `Horizon` via `nf.horizon` |
| `forecasting_model` (the functional wrapper) | `h = nf.horizon(covariates, data)` as the first line of the model (spelled `Horizon.from_data` until the PR-C renames land) |
| `fit_svi`, `SVIFit`, `resolve_optimizer`, `resolve_guide` | `SVI(model, guide, optim, Trace_ELBO()).run(key, num_steps, covariates, data)` |
| `fit_mcmc`, `MCMCFit`, `resolve_kernel`, `_validate_kernel_run_config` | `MCMC(NUTS(model), ...).run(key, covariates, data)` |
| the `_draw_posterior_impl` singledispatch and its three registrations | `mcmc.get_samples()` for MCMC; the guide-based `draw_posterior` for variational fits |
| `register_fit` (the public sampler seam the first draft proposed) | nothing to register: the concept dies with the fit types |
| `OptimizerResolutionError`, `GuideResolutionError`, `KernelResolutionError`, `GuideSampleArgsError` | NumPyro's own errors |
| `OptimizerLike`, `GuideLike`, `KernelLike`, `ForecasterFactory` | concrete NumPyro types at the two call sites that still need them |

Two loose ends from the first draft close themselves:

- The proposed naming asymmetry between a GLM-form functional `predict` and a Pyro-form `ForecastingModel.predict` dissolves. With no class there is exactly one `predict`, the GLM form (`predict(h, obs_dist_fn, latent)`), and the zero-centered-noise variant is deleted.
- The "vanilla-inference invariant" (every model must be fittable with nothing but jax and NumPyro) stops being an invariant to guard, because it becomes the only path. The ordinary SVI and MCMC round-trip tests are that guard.

## What the package keeps, and why

The bar every survivor has to clear: **it does something plain NumPyro cannot do, or cannot do without materializing more memory than an accelerator has.**

### Model building blocks

`Horizon`, `horizon()`, `innovations` (renamed from `time_series`), `markov_series` (renamed from `markov_time_series`), `ssoe` (new, see below), `predict` (GLM form), and `Transition`. These are the package. They carry the train/forecast split, the `_future`-site convention that keeps `AutoNormal` from resizing, and the horizon derived from covariate shapes.

`Horizon` is kept deliberately, since it is the value every building block is parameterized by: it derives the train/forecast split from shapes once per model call, instead of letting each block take `(covariates, data)` and re-derive it independently, which would reopen exactly the mismatch that the rejected `obs=`-on-`predict` idea would (see the docs section). `forecasting_model` is **not** kept: it was a two-line wrapper that saved exactly one line, `h = nf.horizon(covariates, data)` at the top of the model, and it fails the same bar that removed `fit_svi`, while hiding the standard NumPyro `(covariates, data=None)` signature behind a bespoke `(h, covariates)` body signature. Without it the model on the page is a function any NumPyro user already knows how to read.

With the wrapper gone, the `(covariates, data=None)` calling convention the drivers rely on is carried by documentation and by the `ForecastModel` type. PR-A upgrades `ForecastModel` in `typing.py` from the loose `Callable[..., None]` alias to a `@runtime_checkable` Protocol whose `__call__` is `(self, covariates: Array, data: Array | None = None, /) -> None`, positional-only because the drivers invoke positionally, so user models keep free parameter names. What this buys: ty checks the signature structurally at every call site where a user passes a model into `forecast`, `predict_in_sample`, or `to_datatree`, catching a forgotten `data=None` default statically. What it does not buy: at runtime the beartype hook's isinstance check on a runtime-checkable Protocol reduces to member existence, i.e. `callable(model)`, no stronger than the current alias, because Python runtime protocols never check signatures (a model missing the default still fails loudly, with a `TypeError` at the first `data=None` driver call). The alias's docstring has to be rewritten in any case: it currently justifies the looseness by the OOP/functional duality, and this refactor deletes both halves of that sentence.

### Producing draws

Three functions survive, all of them fit-agnostic and all of them memory-bounded:

```python
draw_posterior(rng_key, guide, params, num_samples, *, batch_size=None, device=None)
forecast(rng_key, model, posterior, data, covariates, *, batch_size=None, parallel=True, device=None)
predict_in_sample(rng_key, model, posterior, covariates, *, batch_size=None, parallel=True, device=None)
```

`forecast` and `predict_in_sample` keep their exact current signatures; they already take a posterior dict rather than a fit. They own the `"forecast"` and `"obs"` site conventions and the chunked, device-offloaded predictive driver that compiles `Predictive` exactly once.

`draw_posterior` loses the fit argument in favor of the guide it was hiding. This is the survivor most exposed to the minimality argument, so the case for it in full: `AutoGuide.sample_posterior` is eager and unjitted, and it materializes every latent and deterministic site, plus their intermediates, for every draw at once. On a wide panel that is the single largest allocation of the whole workflow. `draw_posterior` wraps it in a per-guide cached `jax.jit` with a static `sample_shape` so XLA can plan and reuse buffers, then draws in chunks, moving each chunk off the accelerator before the next is drawn. That is not two lines of NumPyro, and [fresh_retail_stockout.ipynb](docs/examples/fresh_retail_stockout.ipynb) does not run on a GPU without it.

Three branch decisions the deleted dispatch layer was hiding, now explicit:

- **MCMC gets no branch and needs none.** `mcmc.get_samples()` already returns materialized draws, so there is no drawing peak to bound; the old `MCMCFit` implementation only thinned an index grid. Deleted, with no `thin` helper added back unless a notebook asks for one.
- **Hand-written guides get no branch either.** `guide` is annotated `AutoGuide`. The old implementation was a bare `Predictive(guide, params=..., num_samples=...)` call with no jit caching and no chunking benefit, and reconstructing it here would need optional `covariates=`/`data=` parameters meaningful on exactly one branch. Users with a hand-written guide write that one `Predictive` line. `GuideSampleArgsError` goes with it.
- `_require_positive_num_samples` moves up into `draw_posterior`, since it currently lives in the dispatch implementations and only guards the chunked path directly.

### The device and batching invariant

This is the package's reason to exist and the pass/fail criterion for the whole refactor. Every knob below survives with identical semantics and identical argument names:

| Knob | Where | Status |
|---|---|---|
| `batch_size=`, `device=` on posterior drawing | `draw_posterior` | survives, guide-based signature |
| `batch_size=`, `parallel=`, `device=` on predictive sampling | `forecast`, `predict_in_sample` | survives unchanged |
| `predictive_batch_size=`, `predictive_device="host"` | `to_datatree` | survives unchanged |
| `batch_size=` on metrics | `eval_crps`, `eval_coverage` | untouched by this refactor |
| `batch_size=` per backtest window | `backtest` and both closure contracts | survives, see below |
| device resolution, transfer, chunk stitching, OOM advice | the private offload module | survives verbatim |
| `"host"` working when `set_platform("cuda")` leaves no CPU backend | device resolution | survives verbatim |
| NumPy leaves flowing through every downstream signature | `Array \| np.ndarray` annotations | must be re-checked at every new signature |

The last row is the one that fails silently. The jaxtyping import hook checks annotations at runtime, so any new signature that types draws as `Array` instead of `Array | np.ndarray` rejects host-offloaded draws at the exact moment a GPU user needs them, while passing every CPU test that never sets `device=`. A CI test therefore walks the full `device="host"` path end to end (draw, in-sample predict, forecast, `to_datatree`) so the NumPy-leaf contract is enforced rather than merely intended.

### ArviZ export

`to_datatree` stays: `arviz_base.from_numpyro` cannot build the forecast groups or 3-D covariate dims, and seven notebooks use it. Its signature drops the fit:

```python
to_datatree(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, Array | np.ndarray],
    data: Array,
    covariates: Array,
    *,
    num_chains: int = 1,
    predictive_batch_size: int | None = None,
    predictive_device: jax.Device | str | None = "host",
    ...
) -> xarray.DataTree
```

The `Array | np.ndarray` on `posterior` is load-bearing, not cosmetic: host-offloaded draws are NumPy, and that is how the stockout notebook calls it. The `_posterior_reshape` singledispatch and the `_reshape_predictive` helper are replaced by a plain `num_chains` reshape, with a divisibility check the old fit-owned value got for free (a user-supplied `num_chains` that disagrees with the sample-axis length would otherwise raise a raw NumPy error or silently mis-slice). The docs note that NumPyro's flattened `get_samples()` is chain-major, so the reshape is valid, and point at `get_samples(group_by_chain=True)` as the unambiguous alternative.

`num_predictive_samples` disappears, because the posterior now arrives already drawn. One consequence worth knowing before re-running notebooks: `to_datatree` no longer draws its own posterior, so a notebook that exports a tree *and* scores draws separately now shares one draw between them instead of drawing twice from different keys. That is more coherent, and it shifts every number those notebooks print.

### Backtesting

`backtest` is currently built on the Forecaster protocol, so it needs a new contract. It takes user closures:

```python
ForecastFn  = (rng_key, model, train_data, train_covariates, full_covariates, num_samples, *, batch_size=None) -> draws
InSampleFn  = (rng_key, model, train_data, train_covariates, num_samples, *, batch_size=None) -> draws
```

The `batch_size` keyword is not garnish. `backtest` has a public `batch_size` today and threads it into both the forecast call and the in-sample scoring; a closure contract without it would silently remove per-window memory bounding from every backtest, which is exactly the regression this refactor must not ship. `backtest` keeps the parameter and keeps forwarding it. `forecast_fn` returns a jax `Array`; a closure that offloads internally brings the draws back before returning, since metrics are jitted.

Knock-on changes, all breaking and all listed in the release notes:

- `BacktestResult.params` is removed rather than left permanently empty, because `results_to_dataframe` emits `param_<name>` columns from it and documents them. This changes the `to_dict()` shape.
- `eval_train=True` requires an `in_sample_fn` and raises at the entry point rather than per window.
- `forecaster_options` is superseded by closure capture.
- `train_walltime`/`test_walltime` no longer correspond to anything the closure exposes, since fitting and forecasting happen inside one call. Either document the new meaning or collapse them into one `walltime`.

`backtest_vectorized` survives with a much smaller change: it never used `Forecaster`, only the guide and optimizer resolvers, so it now takes a concrete `AutoGuide` instance and a concrete NumPyro optimizer (callers building an optax chain call `optax_to_numpyro` themselves). The hand-rolled vmapped SVI loop stays, with its existing docstring about why: vmap cannot host `svi.run`'s host-side conveniences, `AutoGuide` needs an eager warm-up init, and chunked or offloaded draw and predict are not vmappable. Making a fit helper vmap-friendly is not an option here, because there is no fit helper any more.

One consistency trap to settle rather than inherit: annotating `guide: AutoGuide` makes the import hook reject a bad guide before the body's explicit `VectorizedGuideError` can fire, leaving that exception dead but exported. Keep the loose annotation and the explicit raise (the error exists because its message is better), or tighten the annotation and delete the exception. This document recommends the former.

The issue's joblib/multiprocessing suggestion stays rejected in-package: windows are embarrassingly parallel at the process level and `backtest`'s plain-Python loop is already orchestratable externally, which the docs will say.

### contrib/blackjax

The BlackJAX integration is explicitly kept, and it gets *better* out of this refactor rather than worse. With `fit_mcmc` gone, a BlackJAX kernel is visibly just an `MCMCKernel`:

```python
MCMC(BlackjaxMCLMCKernel(model, num_tuning_steps=10_000), num_warmup=0, num_samples=10_000, chain_method="sequential").run(key, covariates, data)
```

`fit_pathfinder` also stays, and the asymmetry with the deleted `fit_svi`/`fit_mcmc` is deliberate: a BlackJAX `PathfinderState` is not a NumPyro posterior and cannot be consumed without the model, its data, and the unconstraining transform, so `PathfinderFit` carries irreducible state rather than wrapping two lines of NumPyro. It also needs the `bfgs_sample` underflow patch, without which every path ELBO is `-inf` beyond a few hundred parameters.

It stops registering on the deleted singledispatch and gains a companion:

```python
pathfinder_samples(rng_key, fit, num_samples, *, batch_size=None, device=None) -> dict[str, Array | np.ndarray]
```

The memory knobs there are mandatory. The registered implementation ends in `jax.vmap(transform)` over every draw at once, and it is bounded today only because `draw_posterior`'s chunk loop calls it once per chunk. A bare `pathfinder_samples` would silently strip chunking from the backend most likely to be run on an accelerator. The chunk-and-transfer loop is factored out of `draw_posterior` so both callers share it.

One guardrail cannot be fully preserved, and this document would rather say so than pretend. `_validate_kernel_run_config` currently rejects `chain_method != "sequential"` for BlackJAX kernels (their instance-held step and postprocess functions capture tracers under vmap/pmap) and warns when `num_warmup > 0` (adaptation lives in `kernel.init`, so warmup is discarded work). With `fit_mcmc` gone there is no entry point to check this: a kernel has no visibility into `chain_method`, and detecting tracers inside `init` false-positives because NumPyro jits the legitimate sequential path too. Options: (a) document the constraints prominently in the three kernel docstrings and the notebook, accepting a worse error message than today, or (b) keep a small public `check_kernel_config(kernel, num_chains, chain_method, num_warmup)` for users to call, which is a wrapper by another name. This document recommends (a), plus a test pinning the failure mode so the degradation is known rather than discovered. The `AIES`/`ESS` ensemble check is dropped as NumPyro's own domain. `KernelConfigError` survives for whichever route is chosen.

## The new `ssoe` building block

Six notebooks hand-roll the same error-feedback pattern: a raw `jax.lax.scan` filter, a hand-made `eps_future` site, a forecast scan feeding errors back, and a hand-registered `"forecast"` deterministic. That is roughly 40 to 60 lines of scan plumbing each.

```python
class SSOEResult(NamedTuple):
    mu: Array            # full-horizon one-step-ahead means, time at axis -2
    y_future: Array      # sampled future values (empty when h.future == 0)
    final_carry: Any     # carry after the in-sample filter

def ssoe(
    h: Horizon,
    name: str,
    y: Array,                                    # REQUIRED driving series over t_obs, time at axis -2
    init_carry: Any,
    predict_fn: Callable[[Any], Array],
    advance: Callable[[Any, Array, Array, Array | None], Any],
    noise_dist: dist.Distribution,               # zero-centered error dist (eps_t := y_t - predict_fn(carry))
    xs: Array | None = None,                     # full-horizon exogenous inputs, split at h.t_obs
    *,
    obs: Array | None = ...,                     # defaults to h.data (None means the site is sampled)
    obs_site: str = "obs",
    obs_mask: Array | None = None,               # likelihood-only, via dist.mask
) -> SSOEResult:
```

Semantics, following the `markov_series` conventions:

1. In-sample: raw `jax.lax.scan`, no sample sites inside. `mu_t = predict_fn(carry)`, `eps_t = y_t - mu_t`, `carry = advance(carry, y_t, eps_t, x_t)`.
2. Likelihood: `numpyro.sample(obs_site, shift_loc(noise_dist, mu_in)[.mask(obs_mask)], obs=obs)`. With the default `obs=h.data` the site is observed while fitting and sampled under `predict_in_sample` or a prior predictive, which is what the drivers need. Multi-channel models (croston, tsb) pass per-channel derived series and distinct `obs_site` names, avoiding both site collisions and double-counted likelihoods.
3. Future errors: site `f"{name}_future"` from `noise_dist` under `plate("time_future", h.future, dim=-2)`. The guide never sees it, since fitting always happens with `future == 0`. Panel batching comes from `noise_dist`'s batch shape.
4. Forecast: raw `jax.lax.scan` consuming the pre-drawn errors. Registers `numpyro.deterministic(f"{name}_forecast", y_future)` and `f"{name}_mu"`. It never registers a bare `"forecast"`, because two components would collide. Single-component models add the one-liner themselves; composed models build their product from the returned `y_future` values (croston: `z.y_future * p.y_future`), which a mu-only return could not express.
5. Shape contract: scanned rows are `(*batch, obs)`, `init_carry` must already be broadcast against them (a scalar carry breaks `lax.scan`'s carry-type invariant), and `ssoe` validates that `predict_fn`'s output carries the trailing obs axis, mirroring the existing markov step-distribution validation. A scalar-shaped mu silently produces a `(t, t)` log-prob otherwise.
6. Frozen-carry semantics: the forecast scan updates the carry by default. Reproducing the notebooks' flat intermittent-demand forecasts requires future gate rows to be `False`. Croston gets this for free (future sales covariate rows are zero), but TSB's "update every period" gate and availability_tsb's scenario availability must be explicitly zeroed over the horizon. This recipe is documented loudly and tested; silently drifting p-channels would corrupt the scenario analysis.

Why `y` is a required argument rather than read from `h.data`: `predict_in_sample` calls the model with `data=None`, `to_datatree` unconditionally calls `predict_in_sample`, availability_tsb runs a prior predictive with `data=None`, and arma's backtest uses the same path. The `"obs"` site must be unobserved in those calls, which is why the current notebooks route the driving series through `covariates`. That routing is load-bearing, not a smell.

One docstring sentence positions SSOE against Kalman marginalization: linear-Gaussian members such as ARMA and additive ETS can be marginalized exactly, and SSOE is the right form for the nonlinear members. The vocabulary keeps "innovations" for the building block of that name and calls SSOE's eps "errors", matching the acronym.

Tests mirror `test_markov.py`: shape invariants, guide-never-resizes, SVI and MCMC round trips (ARMA(1,1) recovery), mask semantics, frozen-versus-updating carry gating, panel shapes, and a two-component composed model exercising `predict_in_sample`, `forecast`, and `to_datatree` end to end.

## Target layout

```
numpyro_forecast/
  __init__.py       # flat top-level exports; convert stays eager (see dependency slimming)
  models/           # zero inference imports
    __init__.py     #   flat re-exports
    _horizon.py     #   Horizon, horizon()
    _innovations.py #   innovations
    _markov.py      #   markov_series, Transition
    _ssoe.py        #   ssoe, SSOEResult
    _predict.py     #   predict (GLM form)
  predictive.py     # draw_posterior, forecast, predict_in_sample
  _offload.py       # device resolution, transfer, chunk stitching, OOM advice
  _validation.py    # shared argument checks
  convert.py evaluate.py metrics.py surgery.py arrays.py acf.py features.py datasets.py
  contrib/blackjax.py  exceptions.py optional.py typing.py
```

`functional/` was named as the counterpart to `forecaster.py`; with the OOP module gone the name means nothing, and there is no inference left in the package to justify an `inference/` subpackage either. All three draw-producing functions live in one `predictive.py`: they share the offload helpers, they share the chunk-and-transfer loop that `pathfinder_samples` also uses, and putting them together retires a private cross-module import (the prediction module currently reaches into the posterior module for `_index_tree`).

`models` is a package rather than a single file because the post-refactor single-file estimate is around 590 lines with no headroom, and the scan-heavy `markov_series` and `ssoe` pairs deserve their own homes. Submodules are private; the public dotted path stays `numpyro_forecast.models.ssoe` via flat re-exports. One subtlety this creates: `tests/test_docs_reference.py`'s walk collects only symbols whose `__module__` matches the module they are found in and skips `_`-prefixed modules, so re-exported names would silently escape the "every public symbol is documented" enforcement. The walk is extended in the same PR to also collect names re-exported via `__all__` from public package `__init__` modules.

Top-level exports: `horizon`, `Horizon`, `innovations`, `markov_series`, `ssoe`, `predict`, `Transition`, `SSOEResult`, `draw_posterior`, `forecast`, `predict_in_sample`, plus the existing evaluation, metrics, convert, and surgery names. Canonical docs spelling: `import numpyro_forecast as nf`. The `install_import_hook` submodule list in `__init__.py` and the CI base-import leak list move in lockstep.

LOC honesty: the package gets smaller, but not dramatically, since the fit wrappers were thin. The win is a smaller API surface to learn, one obvious way to run inference, and roughly 40 to 60 lines of scan plumbing removed from each SSOE notebook.

## Dependency slimming

Orthogonal to everything above and landable independently.

- **arviz becomes the optional extra `numpyro_forecast[arviz]`, with `convert` staying an eager import.** A PEP 562 `__getattr__` was considered and dropped: direct `from numpyro_forecast.convert import ...` (used by tests, `pkgutil` walks, and great-docs) bypasses `__getattr__`, making beartype instrumentation nondeterministic. Instead `convert.py` keeps its place in the `install_import_hook` block, and its module-top `arviz_base`/`xarray` imports move to lazy `optional.require(..., extra="arviz")` calls inside the public functions, with a `TYPE_CHECKING` block for annotations. The already-string-quoted `"xarray.DataTree"` annotations are no-ops for beartype either way.
- Wiring: the `arviz` extra is added to `all`, `all_cuda`, and `dev`, since CI and docs run `uv sync --extra all` and both `tests/test_convert.py` and the notebooks need arviz present. The docs environment keeps arviz regardless, because `scripts/api_snapshot.py` imports old tags whose `__init__` eagerly imports convert.
- The CI base-import leg gains `arviz`, `arviz_base`, `xarray`, and `pandas` in its leak-signal list, and the comment justifying pandas' absence is rewritten (its premise dies with the hard dependency). Same for `tests/test_package.py`.
- matplotlib is dropped from hard dependencies (it is referenced only in a docstring) and added to `dev`.

## Docs, README, notebooks

- README leads with plain NumPyro: one model function built from `nf.horizon`, `nf.innovations`, and `nf.predict`, then `SVI` or `MCMC` and `Predictive`. It keeps the "what each building block replaces in raw NumPyro" table (which also enumerates the sites each one registers: `drift_future`, `obs_future`, `forecast`), the taxonomy paragraph (`innovations` is sampling outside any loop, `ssoe` is an iid error plate plus a deterministic scan consuming them, `markov_series` is sampling inside `contrib.control_flow.scan`), the `handlers.scope` component-reuse pattern, and a paragraph on dims beyond `(time, obs)` (batch dims stack leftward, plus the `markov_series` `plates=` idiom flip). The "Pyro-style API" section is deleted.
- The `_future`-suffix convention is kept. The never-resizes property comes from fitting with `future == 0`, but the suffix is load-bearing as a convention across prediction, `to_datatree`, and roughly 510 tests; `scope` is documented as the composition tool, not a replacement.
- Rejected API-shape asides from the issue, for the record: `obs=` as an explicit `predict` parameter (data flows via `Horizon`; a separate `obs=` reopens the shape-derived-horizon invariant to mismatch), and name-first argument order (`nf.innovations("drift", dist_fn, h)`) in favor of the existing h-first convention. `ssoe` does get an `obs=` override, for its multi-channel case.
- `great-docs.yml` `reference:` is reordered and pruned: Model building blocks / Distribution surgery / Producing draws / Evaluation / ArviZ export / Extensions (contrib) / the rest. The "Forecasters" and "Functional core: fitting" sections are deleted.
- Guard rails move in lockstep per PR: `tests/test_docs_reference.py`, `tests/test_api_snapshot.py`, `tests/test_package.py`, and the CI leak list.
- Tests deleted: `test_forecaster.py`, `test_functional_svi.py`, `test_functional_mcmc.py`, `test_functional_interop.py`, `test_guides.py`, `test_optim.py`. `test_functional_posterior.py` is kept and retargeted to the guide-based `draw_posterior`, since its chunking, `AutoDelta`, and device cases are the regression net for the memory machinery. `test_kernels.py` shrinks to the BlackJAX guard. `tests/example_models.py` is rewritten as functional models, which cascades into `test_examples.py`. `.test_durations` is refreshed at the end.
- `AGENTS.md` rides along: the module-layout description and the extras list. Its rng-key-first rule carves out one exception for `functools.singledispatch` generics and names `draw_posterior`/`_draw_posterior_impl` as the example; that clause is **deleted** rather than re-pointed, because this refactor removes both rng-consuming dispatch generics (the other is `convert._posterior_reshape`) and `surgery`'s dispatchers take no `rng_key`. "`rng_key` first, always" becomes unqualified.

### Notebooks

Three notebooks are rewritten in the draft PR, chosen because between them they exercise everything the refactor touches:

- **`forecasting_univariate.ipynb`**: the OOP/functional twin structure collapses into one functional model. `Forecaster(...)` becomes explicit `AutoNormal` plus `SVI.run` plus `guide.sample_posterior`; the backtest cells gain a local `forecast_fn` closure; `backtest_vectorized` takes an explicit guide and optimizer.
- **`inference_methods_comparison.ipynb`**: the notebook that proves the thesis. Four engines, one model, all driven by standard NumPyro and BlackJAX: `MCMC(NUTS(...))`, explicit `SVI` with `optax_to_numpyro` (previously hidden inside `resolve_optimizer`), `fit_pathfinder` plus `pathfinder_samples`, and `MCMC(BlackjaxMCLMCKernel(...))`. The BlackJAX kernel integration reads better without a wrapper around it.
- **`fresh_retail_stockout.ipynb`**: the GPU and device-handling acceptance test, and the reason `draw_posterior` survives. Its scoring cell changes by exactly one argument, `svi_fit` becoming `guide, params`, while `batch_size=250` and `device="host"` stay exactly as they are through drawing, in-sample prediction, forecasting, and the DataTree export. The `ScaledForecaster` subclass becomes a `scaled_forecast_fn` closure, which is shorter and clearer than the subclass was.

The remaining ten notebooks are handled in a follow-up PR (PR-E), which must land on `refactor3` before the integration merge: notebooks are published from their committed outputs, so the branch's PR previews render removed API until it does, while the public dev site keeps tracking `main` and never shows the interim state.

## PR sequencing

**Branching.** All implementation PRs land on an integration branch, not on `main`, so `main` never carries a half-refactored state between PRs:

- When implementation starts, cut `refactor3` from the tip of `main` and push it. The name marks the target: the branch integrates everything that becomes v0.3.0.
- First commit on `refactor3`, before any PR opens: extend the CI triggers so the branch is testable. `ci.yml` gets `branches: [main, refactor3]` on both `push` and `pull_request`; `docs.yml` adds `refactor3` to its `pull_request` branches only, so PRs into the branch get docs previews while `push` stays `[main]` and no site build runs on integration-branch pushes (production deploys are impossible from it regardless: the `deploy-main` job is guarded to `main` and release events). The wiring lives on `refactor3` rather than `main` on purpose: `pull_request` workflows read the workflow file from the merge ref, which includes the base branch, and `push` workflows read it from the pushed branch, so `main` needs nothing until the integration merge carries the change over; the integration PR may drop `refactor3` from the lists again.
- PR-A through PR-F all target `refactor3`, and `main` is merged into `refactor3` regularly so the integration branch stays current.
- A final integration PR merges `refactor3` into `main`, and the v0.3.0 tag follows immediately (item 8 below).
- Dependency slimming (item 7) targets `main` directly: it is non-breaking and stays releasable even if the refactor stalls, and `refactor3` picks it up through the regular `main` merges.

1. **PR-A, the minimal draft** (opened as a draft PR, per the review's suggestion). Every removal above, the reshaped `draw_posterior`, `to_datatree`, `backtest`, and contrib contracts, the test sweep, and the three notebook rewrites. No renames, no `ssoe`, no layout move: the diff should be about what disappears. If the notebooks make it unwieldy, the stockout rewrite splits into its own PR immediately behind, but PR-B does not start before it lands, because it is the acceptance test for the device machinery.
2. **PR-B, readback.** Having read the three rewritten notebooks, decide which conveniences come back. Candidates to re-litigate: `forecast` and `predict_in_sample`, `draw_posterior`'s name and signature, `backtest`'s closure shape, `to_datatree`'s ergonomics, and whether the `forecasting_model` decorator earns its way back.
3. **PR-C, layout move, renames, and the terminology sweep.** `models/` package plus `predictive.py`, `innovations`/`markov_series`/`horizon()`, top-level exports, docs-reference and api-snapshot sweep.
4. **PR-D, the `ssoe` block**, its tests, and an example model.
5. **PR-E, the remaining ten notebooks.** Must land on `refactor3` before the integration merge; until it does, the branch's PR previews render removed API, while the public dev site keeps tracking `main` and never shows the interim state.
6. **PR-F, README and docs restructure**, parallel to PR-E.
7. **Dependency slimming**, independent, any time; targets `main` directly (see Branching above).
8. **Integration merge and release 0.3.0.** After PR-E and PR-F, one PR merges `refactor3` into `main`, and the v0.3.0 tag follows immediately. The tight coupling is a release blocker, not a nicety: `scripts/api_snapshot.py` resolves the *current* `reference:` against the old tag's worktree, so the moment the reference changes reach `main` every `functional.*` and `forecaster.*` entry stops resolving for v0.2.3 and the stable site's API reference is gutted until a new tag becomes the stable root. With those changes confined to `refactor3`, that window opens at the integration merge and closes at the tag, minutes rather than PRs. Invalidate `.great-docs-cache/` at the merge (the "tags are immutable so snapshots never go stale" assumption no longer holds once snapshot content depends on current reference entries; keying the cache by a hash of the reference section is a possible follow-up).

## Verification

Per PR: `uv run ruff check . && uv run ruff format --check .`, `uv run ty check numpyro_forecast/`, `make tests` (which includes the docs-reference, api-snapshot, and package tests), plus the extras CI leg for contrib changes.

PR-A additionally has to prove the device machinery survived:

- Every row of the device and batching table above holds, checked knob by knob.
- The retargeted posterior tests still cover chunked drawing, `device="host"` returning NumPy leaves, the `AutoDelta` tiling, and reproducibility per `(rng_key, batch_size)`.
- A new end-to-end `device="host"` test walks draws from `draw_posterior` through `predict_in_sample`, `forecast`, and `to_datatree` with NumPy leaves throughout, so the `Array | np.ndarray` contract is enforced by CI rather than by care.
- `backtest(..., batch_size=...)` demonstrably forwards into both closures. A silently ignored keyword is the failure mode, so assert the closure receives it.
- `fresh_retail_stockout.ipynb` re-executes end to end with `batch_size=250, device="host"` intact, and every prose number is re-derived from the new outputs (the shared-posterior change in `to_datatree` shifts them), each backed by a cell that prints it.
- Ideally one real GPU run of that notebook before merge. Every check above is a CPU proxy, and no CPU test proves the thing the package exists for.

PR-D: the `ssoe` tests as specified, including the composed two-channel model through `predict_in_sample`, `forecast`, and `to_datatree`, and the frozen-carry TSB recipe.

Dependency slimming: the base CI leg imports `numpyro_forecast` without arviz, xarray, or pandas and asserts they did not leak, and `to_datatree` without arviz raises the actionable `optional.require` message.

Final: `make docs` builds, the README quickstart runs as written, and after the 0.3.0 release the stable site's API reference is complete again.

## Open questions for reviewers

1. Replacement vocabulary for "primitive": "model building blocks", "model functions", or something better?
2. `draw_posterior`, `forecast`, and `predict_in_sample` are the least NumPyro-like surface left, and also the only thing here that plain NumPyro cannot do on a wide panel. Is keeping all three the right line? And does `draw_posterior` want a guide-shaped name now that it takes a guide?
3. `backtest`'s closure contract: one `forecast_fn`, or a `(fit_fn, forecast_fn)` pair so `backtest` can also drive the in-sample scoring without a second user closure?
4. `markov_series` as the name for the scan-based latent. Bare `markov` was rejected because `pyro.markov` and funsor's `markov` are established enumeration primitives and a false friend for anyone arriving from Pyro.
5. The `ssoe` signature: is the explicit required `y` argument acceptable, or is there a cleaner way to reconcile the filter's need for history with the `data=None` driver calls?
