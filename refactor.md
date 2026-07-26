# Refactor proposal: library, not framework (issue #75)

Status: **proposal for discussion, nothing here is implemented yet.** This is the scaffolding plan for addressing the API feedback in [#75](https://github.com/juanitorduz/numpyro_forecast/issues/75). Feedback welcome, especially on the `ssoe` primitive design and the naming decisions.

## Context

Issue #75 reviews the package API. Core critique: the package feels like a framework (Forecaster classes, fit wrappers, model wrapper) when its real value is a small set of model-building primitives (the `_future`-site trick, prefix conditioning with the exact MVN conditional, horizon bookkeeping) that work inside any plain numpyro model under standard numpyro inference. It also identifies a missing third primitive: 5 of 12 example notebooks (arma, exponential_smoothing, croston, tsb, availability_tsb) hand-roll the identical SSOE (single-source-of-error / error-feedback) pattern: raw `jax.lax.scan` filter, hand-made `eps_future` site, forecast scan feeding errors back, hand-registered `"forecast"` deterministic.

Maintainer constraints (they override the issue where they conflict): keep the OOP API (this stays a Pyro port) but make it a thin wrapper over the functional API; split model from inference while keeping `contrib/blackjax.py` plug-and-play for different samplers; keep the package lightweight; all 12 notebook use cases stay supported. Consequence: `fit_svi`/`fit_mcmc` are NOT removed (the issue suggested it); they are the inference layer the OOP wraps and 7 notebooks use them; they are repositioned as optional conveniences.

**Vanilla-inference invariant (promoted to a design guarantee):** every model built from the primitives must be fittable and forecastable with NOTHING but jax + numpyro: `SVI(model, guide, optim, Trace_ELBO()).run(...)` or `MCMC(NUTS(model)).run(key, covariates, data)`, then `guide.sample_posterior(...)` / `mcmc.get_samples()`, then `Predictive(model, posterior)(key, covariates_full, data)["forecast"]` (the issue's exact proposed flow). `fit_svi`/`fit_mcmc`/`draw_posterior`/`forecast` are conveniences layered on top (progress bars, guide/optimizer/kernel resolution, chunked and device-offloaded sampling), never requirements. This already holds structurally (the model primitives have zero inference imports) but becomes explicit: a new `tests/test_vanilla_numpyro.py` asserts parity between the vanilla flow and the fit-helper flow for an `innovations` model, a `markov_series` model, and an `ssoe` model, and the README's lead example shows the vanilla flow next to the convenience flow.

Proposed decisions:

1. **Renames**: `time_series` -> `innovations`, `markov_time_series` -> `markov_series` (bare `markov` rejected: `pyro.markov` / funsor `markov` is an established enumeration primitive and a false-friend for a Pyro-port audience), new free function `horizon(covariates, data)` (keep `Horizon.from_data` too). Hard break, no deprecation shims (repo precedent), swept through tests/docs/notebooks in lockstep.
2. **Merge predict**: the functional core gets ONE `predict` in GLM form (`predict(h, obs_dist_fn, latent)`, today's `predict_glm` renamed); the zero-centered-noise variant is deleted from the functional core. `ForecastingModel.predict(noise_dist, prediction)` keeps the Pyro signature (via `shift_loc`; probe-verified bitwise-identical sampling, so the interop tests and the univariate notebook's bitwise assert survive). The OOP class also keeps `predict_glm`, delegating to the functional `predict`. This is a deliberate naming asymmetry (same name, two forms); both docstrings cross-reference each other, the README table gets a row disambiguating the two forms, and a new interop test asserts OOP `predict_glm` and functional `predict` register identical sites.
3. **Keep `PathfinderForecaster`**, thinned to the shared base implementation.
4. **Layout**: dissolve `functional/` into a top-level `models/` package + `inference/` subpackage; promote primitives to top-level exports.

This plan was adversarially reviewed (ssoe correctness against the actual notebook code, internal consistency against the issue, feasibility probes against the installed environment); the confirmed findings are folded in below and marked where they changed the design.

## Codebase facts that constrain the design

- `functional/models.py` (417 LOC) has zero imports from svi/mcmc/posterior/prediction; the model/inference split already exists at file level. `predict` is literally `predict_glm(h, lambda mu: shift_loc(noise_dist, mu), prediction)`.
- `forecaster.py` (625 LOC): `ForecastingModel` is a pure façade; `_BaseForecaster._draw_posterior` is implemented identically three times in the subclasses; fitting happens in `__init__` (Pyro behavior, kept).
- `contrib/blackjax.py`: kernels plug via `kernel=` into `fit_mcmc` (import-free MRO-name detection); `fit_pathfinder` registers `PathfinderFit` on the private `_draw_posterior_impl` singledispatch at import time. Nothing in core imports contrib.
- `evaluate.py`: `backtest` defaults `forecaster_fn=Forecaster` (the one core-to-OOP import edge). `backtest_vectorized` hand-rolls the SVI loop deliberately (vmap cannot host `svi.run`'s host-side conveniences; AutoGuide needs an eager warm-up init; chunked/offloaded draw and predict are not vmappable).
- `convert.py` imports `arviz_base`/`xarray` at module top, so arviz>=1.2 is a hard dependency; matplotlib is a hard dependency referenced only in a docstring. `to_datatree` must stay (6 notebooks use it; `arviz_base.from_numpyro` cannot build the forecast groups or 3-D covariate dims).
- **Driver contracts that constrain `ssoe`**: `predict_in_sample` calls the model with `data=None`, so `h.data` is `None` there; `to_datatree` unconditionally calls `predict_in_sample`; availability_tsb runs a prior predictive with `data=None`; arma's backtest uses `eval_train=True` (same path). The `"obs"` site must therefore become unobserved in those calls, which is why the current notebooks route the driving series through `covariates`. That routing is load-bearing, not a smell.
- Guard rails updated in lockstep with any symbol change: `great-docs.yml` `reference:`, `tests/test_docs_reference.py`, `tests/test_api_snapshot.py`, `tests/test_package.py` (hardcoded `__all__` set), `tests/test_functional_interop.py`, README, `.github/workflows/ci.yml` (base-import leak list).

## Target layout

```
numpyro_forecast/
  __init__.py           # tiered exports; convert stays EAGER (see dependency slimming)
  models/               # <- functional/models.py split into private submodules, re-exported flat;
    __init__.py         #    zero inference imports (invariant, enforced by test_vanilla_numpyro)
    _horizon.py         #    Horizon, horizon(), forecasting_model
    _innovations.py     #    innovations
    _markov.py          #    markov_series, Transition
    _ssoe.py            #    ssoe, SSOEResult
    _predict.py         #    predict (glm form)
  inference/            # <- functional/ minus models.py
    __init__.py         # fit_svi, SVIFit, fit_mcmc, MCMCFit, draw_posterior, register_fit,
    svi.py  mcmc.py     #   forecast, predict_in_sample, resolve_optimizer, resolve_guide, resolve_kernel
    posterior.py  prediction.py  _offload.py  _validation.py
  forecaster.py         # thinned OOP wrapper (Pyro-compat layer)
  contrib/blackjax.py   # registration switches to public register_fit
  surgery.py  arrays.py # unchanged (core)
  evaluate.py  metrics.py  convert.py  acf.py  features.py  datasets.py   # extras tier (framing via docs, no extras/ dir)
  exceptions.py  optional.py  typing.py
```

`models` is a package, not a single file. The single-file estimate post-refactor is ~590 LOC (417 today + ~170 for `ssoe` + ~40 for `horizon()`/`SSOEResult` - ~40 for the deleted predict wrapper), which is workable but leaves no headroom for future primitives, and the scan-heavy `markov_series`/`ssoe` pairs deserve their own homes. Splitting during the layout-move PR costs nothing extra. Submodules are private; the public dotted path stays `numpyro_forecast.models.ssoe` etc. via flat re-exports. One subtlety this creates: `tests/test_docs_reference.py`'s walk collects only symbols whose `__module__` equals the module they are found in, and it skips `_`-prefixed modules, so re-exported primitives would silently escape the "every public symbol is documented" enforcement. Fix in the same PR: extend the walk to also collect names re-exported via `__all__` in public package `__init__` modules.

New top-level exports: `horizon`, `Horizon`, `innovations`, `markov_series`, `ssoe`, `predict`, `forecasting_model`, `Transition`, `fit_svi`, `fit_mcmc`, `SVIFit`, `MCMCFit`, `draw_posterior`, `forecast`, `predict_in_sample`, `register_fit` (plus the existing OOP/eval names). Canonical docs spelling: `import numpyro_forecast as nf`.

LOC honesty: the package stays ~5,000 LOC. The win is user-code LOC (each SSOE notebook drops roughly 40-60 lines of scan plumbing; the user still writes `predict_fn`/`advance`/gate construction) and a smaller API surface to learn.

Vocabulary note: the taxonomy must not use "innovations" for both the renamed `time_series` AND ssoe's error draws. Reserve "innovations" for the primitive; call ssoe's eps "errors" (matching the SSOE acronym) in all docs and docstrings.

## The new `ssoe` primitive (~140 LOC + docstring)

Redesigned after adversarial review (the first draft could not express croston/tsb/availability_tsb and broke the `predict_in_sample`/`to_datatree` drivers):

```python
class SSOEResult(NamedTuple):
    mu: Array            # full-horizon one-step-ahead means, time at axis -2
    y_future: Array      # sampled future values (empty when h.future == 0)
    final_carry: Any     # carry after the in-sample filter

def ssoe(
    h: Horizon,
    name: str,
    y: Array,                                    # REQUIRED driving series over t_obs, time at axis -2.
    init_carry: Any,                             #   Usually sliced from covariates: the drivers call the
    predict_fn: Callable[[Any], Array],          #   model with data=None (predict_in_sample, prior
    advance: Callable[[Any, Array, Array, Array | None], Any],  # predictive), so h.data CANNOT drive the filter.
    noise_dist: dist.Distribution,               # zero-centered error dist (eps_t := y_t - predict_fn(carry))
    xs: Array | None = None,                     # full-horizon exogenous inputs, split at h.t_obs (as markov_series)
    *,
    obs: Array | None = ...,                     # what the likelihood observes; defaults to h.data (None -> site
    obs_site: str = "obs",                       #   is sampled, which is what predict_in_sample needs)
    obs_mask: Array | None = None,               # boolean, likelihood-only via dist.mask (masked steps still
) -> SSOEResult:                                 #   feed advance; gating the STATE is xs's job)
```

Semantics (follows the `markov_series` conventions):

1. In-sample: raw `jax.lax.scan` (no sample sites inside): `mu_t = predict_fn(carry)`, `eps_t = y_t - mu_t`, `carry = advance(carry, y_t, eps_t, x_t)`.
2. Likelihood: `numpyro.sample(obs_site, shift_loc(noise_dist, mu_in)[.mask(obs_mask)], obs=obs)`. With the default `obs=h.data` the site is observed while fitting and sampled under `predict_in_sample`/prior predictive (`h.data is None`), exactly matching today's driver contracts. Multi-channel models (croston/tsb) pass per-channel derived series and distinct `obs_site` names ("z_obs", "p_obs"), avoiding both site collisions and double-counted likelihoods; no `handlers.scope` required (though it still composes).
3. Future errors: site `f"{name}_future"` from `noise_dist` under `plate("time_future", h.future, dim=-2)`; the guide never sees it (fitting always happens with `future == 0`); panel batching comes from `noise_dist`'s batch shape (probe-verified: batch `(series,)` under the plate yields `(future, series)`; availability_tsb needs no `plates=`).
4. Forecast: raw `jax.lax.scan` consuming the pre-drawn errors: `mu = predict_fn(carry)`, `y = mu + eps`, `carry = advance(carry, y, eps, x_t)`. Registers `numpyro.deterministic(f"{name}_forecast", y_future)` and `numpyro.deterministic(f"{name}_mu", mu_full)`. It NEVER registers bare `"forecast"` (two components would collide; NumPyro raises on duplicate sites). Single-component models add the one-liner `numpyro.deterministic("forecast", res.y_future)`; composed models build their product from the returned `y_future` values (croston: `z.y_future * p.y_future`), which a mu-only return could not express.
5. Shape contract: scanned rows are `(*batch, obs)`; `init_carry` must already be broadcast against them (scalar carries break `lax.scan`'s carry-type invariant), and `ssoe` validates that `predict_fn`'s output carries the trailing obs axis, mirroring `_validate_markov_step_dist` (a scalar-shaped mu silently produces a `(t, t)` log-prob otherwise).
6. Frozen-carry semantics: the forecast scan updates the carry by default. Reproducing the notebooks' flat intermittent-demand forecasts requires future gate rows to be `False`: croston gets this for free (future sales covariate rows are zero), but TSB's "update every period" gate and availability_tsb's scenario availability must be explicitly zeroed over the horizon (`xs = concat([gate_in, zeros(future)])`). This recipe is documented loudly and tested; silently drifting p-channels would corrupt the scenario analysis.
7. Doc notes: `ssoe` owns its likelihood, so it is mutually exclusive with `predict` on the same `obs_site`; `f"{name}_mu"` has length `t_obs` in `fit.samples` but `duration` under a forecast-time `Predictive` (coordinate-alignment note; arma's `posterior_dims={"mu_t": ...}` sweeps to the new name); one docstring sentence positions SSOE against Kalman marginalization (linear-Gaussian members like ARMA and additive ETS can be marginalized exactly; SSOE is the right form for the nonlinear members), answering the issue's dynestyx aside.

Tests: new `tests/test_ssoe.py` mirroring `test_markov.py`: shape invariants, guide-never-resizes, SVI + MCMC round trips (ARMA(1,1) recovery), mask semantics, frozen-vs-updating carry gating (the TSB recipe), panel shapes, a two-component composed model exercising `predict_in_sample` + `forecast` + `to_datatree` end to end, plus an example model in `tests/example_models.py` and interop coverage.

## OOP thinning (forecaster.py)

- `_BaseForecaster.__init__` gains a `fit` parameter stored as `self._fit`; the base implements `_draw_posterior` concretely as `draw_posterior(rng_key, self._fit, ...)`. Delete the three identical subclass overrides (~90 LOC).
- Keep `Forecaster`, `HMCForecaster`, `PathfinderForecaster` as thin constructors: `super().__init__(model, t_obs, fit_xxx(...))` plus convenience attributes. Fitting stays in `__init__` (Pyro behavior).
- Add classmethod `from_fit(model, fit, t_obs)`: defined on the private base but documented on `Forecaster`; returns an instance of the class it is called on; convenience attrs (`guide`/`params`/`losses`/`posterior_samples`/`elbo`) populated defensively via `getattr` on the fit, so `Forecaster.from_fit` accepts any registered fit type.
- `ForecastingModel` methods renamed to mirror the primitives (`innovations`, `markov_series`, `ssoe`, `predict_glm`) with the one documented exception: `predict` keeps Pyro's `(noise_dist, prediction)` form (decision 2).

## Public sampler seam (inference/posterior.py)

```python
def register_fit(fit_type: type, draw_fn: Callable[[Array, Any, int], dict[str, Array]]) -> None:
    """Register a posterior drawer for a custom fit type. draw_fn(rng_key, fit, num_samples)."""
```

Adapts the rng-key-first `draw_fn` into the private `_draw_posterior_impl` singledispatch (preserves the private-generic/public-wrapper convention from AGENTS.md). `contrib/blackjax.py` switches to `register_fit(PathfinderFit, ...)`. Exported top-level; documented in a new "Extending inference" reference section with the fit-protocol contract (works with `draw_posterior`, `forecast`, `from_fit`, and `backtest`'s `forecaster_fn`). The BlackJAX kernel MRO-name detection in `inference/mcmc.py` stays as-is.

## Dependency slimming

- **arviz becomes the optional extra `numpyro_forecast[arviz]`, with `convert` staying an EAGER import** (design changed by review). A PEP 562 `__getattr__` was considered and dropped: direct `from numpyro_forecast.convert import ...` (used by tests, `pkgutil` walks, and great-docs) bypasses `__getattr__`, making beartype instrumentation nondeterministic. Instead: `convert.py` keeps its place in the `install_import_hook` block; inside it, the module-top `import arviz_base` / `import xarray` move to lazy `optional.require("arviz_base", extra="arviz")` calls inside the three public functions, with a `TYPE_CHECKING` block for annotations. Probe-verified: the module imports cleanly under the hook without arviz/xarray installed, and the already-string-quoted `"xarray.DataTree"` annotations are no-ops for beartype either way. `convert.py`'s module docstring (currently asserting arviz is a hard dep) and `optional.py`'s extras enumeration get updated.
- Wiring: the `arviz` extra is added to `pyproject.toml` `all`/`all_cuda` AND to `dev` (CI and docs run `make setup` which is `uv sync --extra all`; `tests/test_convert.py` and the notebooks need arviz present). The docs environment keeps arviz regardless, because `scripts/api_snapshot.py` imports old tags whose `__init__` eagerly imports convert.
- `.github/workflows/ci.yml` base-import leg: rewrite the comment justifying pandas' absence from the leak list (its premise dies with the hard dep) and add `arviz`, `arviz_base`, `xarray`, `pandas` to the leak-signal list; same for `tests/test_package.py`'s list. The no-arviz import check folds into this existing base leg.
- Drop matplotlib from hard deps (docstring-only reference); add matplotlib to the `dev` extra.
- Keep `to_datatree` (confirmed necessary); the issue's "why not `arviz_base.from_numpyro`" question gets answered in the arviz-extra docs page.

## evaluate/backtest

- Kill the core-to-OOP edge: `backtest`'s signature becomes `forecaster_fn: ForecasterFactory | None = None`, resolved to `Forecaster` via a local import inside the body before `_run_backtest` (which keeps the non-optional type). `typing.py`'s `ForecasterFactory` docstring and the `backtest` param docs get updated ("None resolves to `Forecaster`").
- `backtest_vectorized`: kept as a documented fast path. Docstring and docs state honestly why the SVI loop is hand-rolled (vmap cannot host `svi.run`'s conveniences; AutoGuide prototype-trace warm-up; chunked/offloaded draw and predict are not vmappable) and point to `backtest` as the general tool. Hygiene: the scan update loop moves into `inference/svi.py::_svi_scan_steps(svi, state, num_steps)` so the step logic has one home. Making `fit_svi` vmap-friendly is explicitly rejected (it would recreate the same duplication in a worse place).
- Issue asides, explicitly dispositioned: the joblib/multiprocessing suggestion is rejected in-package (windows are embarrassingly parallel at the process level; `backtest`'s plain-Python loop is already orchestratable externally, and the docs will say so); dims-beyond-(time, obs) gets a README paragraph (batch dims stack leftward, plus the `markov_series` `plates=` idiom flip, documented loudly per the issue's ask).

## Docs / README / notebooks

- README restructure: (1) lead with plain-numpyro + primitives: the model built from `nf.horizon` + `nf.innovations` + `nf.predict`, shown fit BOTH ways side by side, vanilla `SVI`/`MCMC` + `Predictive` (the vanilla-inference invariant, the issue's exact flow) and the `nf.fit_svi`/`nf.forecast` conveniences, with one sentence on what the conveniences add; (2) a "what each primitive replaces in raw NumPyro" table, which also enumerates the sites each primitive registers (`drift_future`, `obs_future`, `forecast`, per the issue's "sites that are never explicitly typed" aside); (3) the taxonomy paragraph: `innovations` = sampling outside any loop; `ssoe` = iid error plate + deterministic scan consuming them; `markov_series` = sampling inside `contrib.control_flow.scan`; (4) a "Pyro-style API" section (all four classes as the thin compat wrapper); (5) extras last. The `handlers.scope` component-reuse pattern is documented; the `_future`-suffix convention is kept (the never-resizes property comes from fitting with `future == 0`, but the suffix is load-bearing as a convention across prediction, `to_datatree`, ~370 tests; `scope` is documented as the composition tool, not a replacement).
- Remaining API-shape asides, dispositioned: `obs=` as an explicit `predict` parameter is rejected (data flows via `Horizon`; a separate `obs=` reopens the shape-derived-horizon invariant to mismatch), though `ssoe` gets an `obs=` override for its multi-channel case; the issue's name-first argument order (`nf.innovations("drift", dist_fn, h)`) is rejected in favor of the existing h-first convention (consistency across all primitives and the OOP delegations).
- `great-docs.yml` `reference:` reordered: Model primitives / Distribution surgery / Inference helpers / Extending inference (`register_fit`, contrib.blackjax) / Pyro-style forecasters / Evaluation / ArviZ export / the rest.
- Notebooks: the 5 SSOE notebooks are rewritten around `nf.ssoe` (the driving series keeps arriving via `covariates`; that routing is load-bearing for `predict_in_sample`/`to_datatree`, see driver contracts above); `forecasting_univariate` is trimmed from the full OOP/functional twin to functional-led with a short OOP coda; all 12 get the mechanical rename/import sweep; `cast("Array", ...)` is swept to `jnp.asarray(...)`. Re-executed via jupytext; `.test_durations` refreshed after the sweep.
- AGENTS.md updates ride along: the `draw_posterior`/`_draw_posterior_impl` convention path, the module-layout description, the extras list. The dissolved `functional/__init__` architecture docstring moves to `models/__init__.py` + `inference/__init__.py`.

## PR sequencing (each leaves CI green; reference:/api-snapshot/test_package/tests updated in lockstep per PR)

1. **PR1 (independent): OOP thinning + public seam.** Base `_fit` + concrete `_draw_posterior`, `from_fit`, thin all three forecaster classes, `register_fit`, contrib switch, `backtest` lazy default. Files: `forecaster.py`, `functional/posterior.py`, `contrib/blackjax.py`, `evaluate.py`, `typing.py`, `__init__.py`, `great-docs.yml`, `tests/test_forecaster.py`, `tests/test_package.py`, `tests/contrib/test_blackjax.py`.
2. **PR2 (independent): dependency slimming.** Lazy arviz inside `convert.py` (module stays eager), pyproject (`[arviz]` extra + `all`/`all_cuda`/`dev`, drop matplotlib), `.github/workflows/ci.yml` leak lists + comment, `tests/test_package.py`, `optional.py` docstring, AGENTS.md extras note, no-arviz import test in the base CI leg.
3. **PR3 (independent): `ssoe` primitive** (in the current `functional/models.py` location, with the redesigned signature above) + `ForecastingModel.ssoe` + `tests/test_ssoe.py` + `tests/example_models.py` + reference entry + `tests/test_vanilla_numpyro.py` (the vanilla-inference parity guard for all three primitives).
4. **PR4a (after 1-3): pure layout move.** Split `functional/models.py` into the `models/` package (private submodules, flat re-exports), `functional/` -> `inference/`; import-path sweep; extend the `test_docs_reference.py` walk to cover `__all__` re-exports from public package `__init__` modules; NO symbol renames. Bitwise-neutral, trivially reviewable despite size.
5. **PR4b (after 4a): renames + predict merge + export promotion.** `innovations`/`markov_series`/`horizon()`; functional `predict` = glm form (OOP keeps Pyro form); top-level exports including `SVIFit`/`MCMCFit`/`Transition`; `test_package.py` `__all__`; full docs-reference/api-snapshot sweep; `.test_durations` refresh if test files were renamed. Blast radius measured at ~37 files for 4a+4b combined, which is why they are split.
6. **PR5 (immediately after 4b, ideally branched from it): notebook rewrites** (5 SSOE + univariate trim + rename sweep of all 12). Must not lag: the dev docs render notebooks from the current checkout, so between 4b and 5 the published dev site would show deleted names.
7. **PR6 (after 4b, parallel to 5): README + docs restructure** + the primitive-vs-raw-numpyro table + backtest_vectorized limitations prose.
8. **Release 0.3.0 immediately after PR5/PR6** (review blocker): `scripts/api_snapshot.py` resolves the CURRENT `great-docs.yml` `reference:` against the old tag's worktree, so after PR4b every `functional.*` entry stops resolving for v0.2.3 and the stable site's API reference is gutted until a new tag becomes the stable root. Accept the short interim degradation, keep the PR4b-to-release window tight, and invalidate `.great-docs-cache/` after PR4b (the "tags are immutable so snapshots never go stale" assumption in `scripts/build_docs.py` no longer holds once snapshot content depends on current reference entries; keying the cache by a hash of the reference section is a possible follow-up).

## Verification

- Per PR: `uv run ruff check . && uv run ruff format --check .`, `uv run ty check numpyro_forecast/`, `make tests` (includes `test_docs_reference.py`, `test_api_snapshot.py`, `test_package.py`, `test_functional_interop.py`), plus the extras CI leg for contrib changes.
- PR2: the base CI leg imports `numpyro_forecast` without arviz/xarray/pandas and asserts they did not leak; `to_datatree` without arviz raises the actionable `optional.require` message.
- PR3: `tests/test_ssoe.py` as specified above, including the composed two-channel model through `predict_in_sample` + `forecast` + `to_datatree`, and the frozen-carry TSB recipe. `tests/test_vanilla_numpyro.py` passes: plain `SVI`/`MCMC` + `Predictive` produce the same sites (bitwise where the sampling paths coincide) as the fit-helper flow.
- PR4b: an interop test asserts OOP `predict_glm` and functional `predict` register identical sites; the bitwise interop tests stay green (`shift_loc` equivalence is probe-verified for Normal, StudentT, and MVN under both parametrizations).
- PR5: `uv run jupytext --to notebook --execute` for each rewritten notebook; ty on the notebooks before execution; prose numbers have printed backing; the rewritten SSOE notebooks reproduce the old forecasts distributionally (spot-check CRPS/coverage against the committed outputs).
- Final: `make docs` builds; after the 0.3.0 release the stable site's API reference is complete again; the README quickstart runs as-is.

## Open questions for reviewers

1. `ssoe` signature: is the explicit required `y` argument (driving series, usually sliced from covariates) acceptable, or is there a cleaner way to reconcile the filter's need for history with `predict_in_sample`/prior-predictive calls where `data=None`?
2. `markov_series` vs another name for the scan-based latent (bare `markov` was rejected as a `pyro.markov` false-friend).
3. The predict naming asymmetry (functional `predict` in GLM form, `ForecastingModel.predict` in Pyro form): acceptable trade-off for port fidelity, or should the OOP method also switch?
4. Anything missing from the vanilla-inference invariant so the package works as "just functions inside your numpyro model"?
