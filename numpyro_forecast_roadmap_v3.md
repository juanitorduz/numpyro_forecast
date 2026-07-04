# `numpyro_forecast` — Implementation Roadmap (v3, final)

**Implementation-ready specification: function scaffolds, helper contracts,
ordered checklists, named tests with assertions, and the closed risk
register.** Supersedes v1 (feature survey) and v2 (audit + risk revision).

Status: final for implementation · Target: `numpyro_forecast >= 0.2`
Spike evidence: S-M (scan/replay/carry) and S-V (vmap-SVI) both executed
GREEN on jax 0.10.2 / numpyro 0.21.0; scripts live in `spikes/` and are
converted to permanent regression tests (§11.3).

---

## 0. Conventions, envelope, layout

### 0.1 Package-wide conventions (normative)

1. **Axis contract:** batch dims left, time at `-2`, observation/event dim
   at `-1`, for every array crossing a public boundary. The single
   exception is the *stored* layout of scan sites (C6): time leading, as
   `contrib.control_flow.scan` produces and `Predictive` replays them.
2. **Fit contract:** any object registered with `_draw_posterior_impl`
   yields `dict[str, Array]` with a leading sample axis over in-sample
   latent sites only (never `_future` sites, never `forecast`).
3. **Resolution at the boundary:** every widened parameter
   (`optim`, `guide`, `kernel`) is normalized by a `resolve_*` pure
   function called first thing in the `fit_*` body; nothing downstream
   sees the widened union type.
4. **Errors:** misuse fails at the earliest call with a message naming the
   accepted forms; unsupported dispatch fails with registration
   instructions. No silent fallbacks anywhere.
5. **Registration by import** for extension modules; every internal
   registry has a public registration path (§6.1 principle).
6. **Docstrings:** numpydoc; every public function documents its PRNG
   consumption ("splits `rng_key` into N streams for …") — a convention
   that has already prevented one key-reuse bug in `backtest`'s
   `eval_train` path and is now normative.

### 0.2 Compatibility envelope (tested floors)

| Dependency | Pin | Assumed surface (⚙ = canary test) |
|---|---|---|
| jax | `>= 0.10` (spike-tested 0.10.2) | pytree `block_until_ready`; `jax.monitoring` listener ⚙; vmap-of-`lax.scan` semantics used by §4.4 |
| numpyro | `>= 0.21` (spike-tested) | `initialize_model`/`constrain_fn`; scan replay + carry threading ⚙ (spike tests); reparam-inside-scan placement ⚙; plate-wraps-scan rejection ⚙; `AIES`/`ESS` |
| optax (soft) | `>= 0.2` | structural `init`/`update` only |
| blackjax (extra) | `>= 1.2, < 2` | `window_adaptation`, `mclmc_find_L_and_step_size`, `vi.pathfinder.{approximate,sample}` ⚙ |
| arviz-base (extra) | `>= 0.9` (ArviZ 1.0 line) | `dict_to_dataset`; DataTree schema; `sample_dims` attrs ⚙ |
| xarray (with arviz extra) | `>= 2024.11` | `DataTree` |
| pandas (extra) | `>= 2.0` | — |

`pyproject.toml` extras and the `require(module, *, extra)` /
`_api_canary(module, attrs)` helpers as specified in v2 §0.1 (unchanged).

### 0.3 Final module layout

```
numpyro_forecast/
├── __init__.py          # re-exports incl. register_elementwise, to_datatree
├── typing.py            # + OptimizerLike, GuideLike, KernelLike, BuildFn, Metric
├── functional.py        # resolve_optimizer, resolve_guide, resolve_kernel,
│                        #   fit_svi, fit_mcmc, draw_posterior (+PathfinderFit reg
│                        #   lives in contrib), forecast (+padding),
│                        #   predict, predict_glm, time_series, markov_time_series
├── forecaster.py        # shims: Forecaster, HMCForecaster, PathfinderForecaster
├── util.py              # dispatchers, elementwise registry, require()
├── metrics.py           # + _pinball, eval_pinball, eval_interval_score, make_mase
├── evaluate.py          # _timed fix, backtest (+reuse_model, per_window_metrics),
│                        #   backtest_vectorized, results_to_dataframe
├── convert.py           # to_datatree, add_forecast, to_inferencedata (shim)
└── contrib/
    ├── __init__.py
    └── blackjax.py      # _BlackjaxKernel, BlackjaxNUTSKernel,
                         #   BlackjaxMCLMCKernel, BlackjaxCustomKernel,
                         #   fit_pathfinder, PathfinderFit
tests/
├── conftest.py          # univariate/hierarchical fixtures, count_compilations
├── test_optim.py  test_guides.py  test_kernels.py  test_functional.py
├── test_util.py  test_metrics.py  test_evaluate.py  test_backtest_vectorized.py
├── test_markov.py  test_markov_spike_invariants.py  test_vmap_svi_invariants.py
├── test_convert.py  test_consistency.py
└── contrib/test_blackjax.py
spikes/                  # executed spike scripts, kept verbatim for provenance
```

---

## 1. Optimizer resolution

**Goal.** `fit_svi` accepts `None` / positive scalar / optax
`GradientTransformation` / `_NumPyroOptim`. **Files:** `functional.py`,
`typing.py`, `forecaster.py` (annotation only).

### 1.1 Scaffold (final; v2 §1.1 with no further changes)

```python
_DEFAULT_LEARNING_RATE: float = 0.01


def resolve_optimizer(
    optim: "OptimizerLike",
) -> _NumPyroOptim:
    """Normalize an optimizer specification into a NumPyro optimizer.

    Accepted forms: ``None`` (``Adam(0.01)``); a finite positive scalar
    learning rate (``float``/``int``/NumPy scalar/0-d array) giving
    ``Adam(lr)``; an ``optax.GradientTransformation`` (wrapped via
    ``numpyro.optim.optax_to_numpyro`` — imported lazily, so optax stays a
    soft dependency); a ``_NumPyroOptim`` (identity).

    Raises
    ------
    TypeError
        For ``bool`` (bool <: int would silently mean ``Adam(1.0)``) and any
        other unrecognized type; the message lists the accepted forms.
    ValueError
        For a non-finite or non-positive learning rate.
    """
    # order: None -> _NumPyroOptim -> bool guard -> scalar -> structural optax -> TypeError
```

Body exactly as v2 §1.4 (bool guard before the numeric branch; `ndim == 0`
probe for array scalars; `hasattr(optim, "init") and hasattr(optim, "update")`
structural check with lazy `optax_to_numpyro` import).

### 1.2 `fit_svi` — final full signature (C5)

```python
def fit_svi(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    optim: "OptimizerLike" = None,
    guide: "GuideLike" = None,
    num_steps: int = 1_001,
    stable_update: bool = False,
) -> SVIFit:
    """Fit with stochastic variational inference.

    PRNG: splits ``rng_key`` once — one stream for SVI, none retained.

    Notes
    -----
    ``optim`` and ``guide`` accept the widened forms documented on
    :func:`resolve_optimizer` and :func:`resolve_guide`. The returned
    :class:`SVIFit` carries the in-sample ``data``/``covariates`` (needed to
    draw from hand-written guides and by :func:`~numpyro_forecast.convert.to_datatree`).
    """
    _require_equal_duration(data, covariates)
    optimizer = resolve_optimizer(optim)
    resolved_guide = resolve_guide(guide, model)
    svi = SVI(model, resolved_guide, optimizer, Trace_ELBO())
    result = svi.run(rng_key, num_steps, covariates, data,
                     progress_bar=False, stable_update=stable_update)
    return SVIFit(guide=resolved_guide, params=result.params,
                  losses=result.losses, data=data, covariates=covariates)
```

### 1.3 Implementation checklist
1. `typing.py`: `OptimizerLike` string alias (TYPE_CHECKING-guarded optax import).
2. `functional.py`: `resolve_optimizer`; swap into `fit_svi`; extend `SVIFit`.
3. `forecaster.py`: widen `Forecaster.__init__` annotations; forward.
4. Docs: one optax example in the `fit_svi` docstring (cosine schedule + clip).

### 1.4 Tests — `tests/test_optim.py`
As v2 §1.2 (all of v1's plus bool rejection, 0-d array acceptance,
non-finite rejection, optax-import-poisoned soft-dependency proof), plus:
- `test_svifit_carries_training_args` — `fit.data is data`,
  `fit.covariates is covariates` (identity, not equality — no copies).

**Acceptance:** all resolver branches covered; `fit_svi` with
`optax.chain(clip, adam(schedule))` reaches the same loss magnitude as the
Adam baseline on the univariate fixture (loose statistical assert).

---

## 2. Guide resolution

**Goal.** `guide=` accepts instance / class / `functools.partial` factory /
hand-written function; MAP (`AutoDelta`) forecasts correctly; misuse fails
loudly. **Files:** `functional.py`, `typing.py`, `forecaster.py`.

### 2.1 Scaffolds (final, C4 reconciled)

```python
def resolve_guide(
    guide: "GuideLike",
    model: ForecastModel,
) -> "AutoGuide | Callable[..., None]":
    """Normalize a guide specification against ``model``.

    Resolution: ``None`` -> ``AutoNormal(model)``; ``AutoGuide`` instance ->
    identity; ``AutoGuide`` subclass or ``functools.partial`` of one ->
    called with ``model``; any other callable -> hand-written guide, after
    :func:`_probe_handwritten_guide` (which rejects single-positional-arg
    callables — the mistyped-factory shape — with an error explaining both
    interpretations). Anything else -> ``TypeError``.
    """


def _probe_handwritten_guide(guide: Callable[..., object]) -> None:
    """Reject callables whose signature matches a guide *factory*.

    Uses ``inspect.signature``; exactly one required positional parameter
    and no defaults => raise ``TypeError`` with the dual-interpretation
    message (v2 §2.1). Signatures that ``inspect`` cannot resolve
    (builtins, some partials) pass the probe — the probe is a tripwire for
    the common mistake, not a gatekeeper.
    """


def _ensure_sample_axis_for_delta(
    samples: dict[str, Array], num_samples: int
) -> dict[str, Array]:
    """Tile AutoDelta point estimates to a leading sample axis.

    Called **only** when the fit's guide is an ``AutoDelta`` (guide-type
    dispatch, never shape inspection — C4): every leaf is broadcast to
    ``(num_samples, *leaf.shape)`` unconditionally. For all other Auto
    guides ``sample_posterior`` already returns the axis and this function
    is never invoked.
    """
```

`_draw_posterior_impl` for `SVIFit` (final):

```python
@_draw_posterior_impl.register
def _(fit: SVIFit, num_samples: int, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    if isinstance(fit.guide, AutoDelta):
        point = fit.guide.sample_posterior(rng_key, fit.params)
        return _ensure_sample_axis_for_delta(point, num_samples)
    if isinstance(fit.guide, AutoGuide):
        return fit.guide.sample_posterior(
            rng_key, fit.params, sample_shape=(num_samples,)
        )
    if fit.covariates is None:
        raise ValueError(_HANDWRITTEN_GUIDE_NEEDS_ARGS_MSG)   # v2 §2.2 text
    predictive = Predictive(fit.guide, params=fit.params, num_samples=num_samples)
    return predictive(rng_key, fit.covariates, fit.data)
```

### 2.2 Implementation checklist
1. `typing.py`: `GuideLike` alias.
2. `functional.py`: the three functions above; `SVIFit` fields
   `data`/`covariates` with `None` defaults; `fit_svi` populates them.
3. `forecaster.py`: `Forecaster.__init__(..., guide: GuideLike = None)`.
4. Docs: table of the five accepted forms; note that a hand-written guide
   must use the model's `(covariates, data=None)` signature and cover
   **only in-sample sites** (the `_future` invariant).

### 2.3 Tests — `tests/test_guides.py`
v2 §2.3 in full (resolution matrix incl. `partial`; lambda-factory
rejection R4; manual-`SVIFit`-without-args ValueError; the AutoDelta
dimension-coincidence regression now trivially passing because dispatch is
by type; hand-written mean-field guide end-to-end; per-guide-flavor
shape-contract forecasts). Release gate unchanged:
- `test_guide_never_sees_future_sites[AutoNormal|AutoMVN|AutoDelta|handwritten]`
  — inspect the guide's sites/params; assert no `*_future` names.

**Acceptance:** `guide=AutoMultivariateNormal` (bare class) fits and
forecasts the univariate fixture; `guide=AutoDelta` forecast has sample
axis with *varying* values (obs noise) over tiled latents.

---

## 3. Sampler backends

**Goal.** Every NumPyro `MCMCKernel` through `fit_mcmc`; blackjax full
samplers as kernels (incl. user-written); Pathfinder as a fit type; scope
of "all samplers" explicit. **Files:** `functional.py`, `typing.py`,
`contrib/blackjax.py`, `forecaster.py`.

### 3.1 `resolve_kernel` + validation (final)

```python
def resolve_kernel(
    kernel: "KernelLike",
    model: ForecastModel,
    kernel_kwargs: Mapping[str, Any] | None,
) -> MCMCKernel:
    """Normalize a kernel specification.

    ``None`` -> ``NUTS(model, **kernel_kwargs)`` (kwargs become NUTS
    options — no need to name the default class to tune it); kernel class
    -> ``kernel(model, **kernel_kwargs)``; kernel instance -> identity,
    and combining an instance with non-empty ``kernel_kwargs`` raises
    ``ValueError`` (ambiguous). Anything else -> ``TypeError``.
    """


def _validate_kernel_run_config(
    kernel: MCMCKernel, num_chains: int, chain_method: str, num_warmup: int
) -> None:
    """Entry-point checks for constraints NumPyro surfaces late or never.

    - ``AIES``/``ESS``: require ``num_chains > 1`` AND
      ``chain_method == "vectorized"`` (the ensemble is the chain batch).
    - ``_BlackjaxKernel`` subclasses: require ``chain_method == "sequential"``
      (instance-held step/postprocess functions capture tracers under
      vmap/pmap tracing — K5); warn via ``warnings.warn`` if
      ``num_warmup > 0`` (adaptation lives in ``kernel.init``; warmup steps
      are discarded work).
    Each violation raises ``ValueError`` naming the constraint and the fix.
    """
```

`fit_mcmc` — final full signature (C5):

```python
def fit_mcmc(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    kernel: "KernelLike" = None,
    kernel_kwargs: Mapping[str, Any] | None = None,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 1,
    chain_method: str = "sequential",
) -> MCMCFit:
    """Fit with MCMC. PRNG: consumed entirely by :class:`~numpyro.infer.MCMC`.

    ``MCMCFit`` gains ``num_chains`` (frozen field, default 1) so chain
    structure survives into :func:`~numpyro_forecast.convert.to_datatree`.
    Samples are stored flattened (``group_by_chain=False``) — the §5.2
    reshape contract.
    """
```

**Scope table** (normative docs content, from v2 §3.1): NUTS/HMC/SA/
BarkerMH — class or instance path; AIES/ESS — class path + validated run
config; MixedHMC/DiscreteHMCGibbs/HMCGibbs/HMCECS — **instance** path
(constructor needs more than the model), each with a docs example and one
smoke test; NestedSampler — out of scope (not an `MCMCKernel`; recorded as
a possible `contrib.jaxns` fit type).

### 3.2 `contrib/blackjax.py` — kernels (final)

Class set: `_BlackjaxKernel` (base; owns `initialize_model` plumbing,
`sample_field = "position"`, the `_BJState(position, inner, rng_key)`
namedtuple, and an **inner-state validation** in `init`: assert the built
state exposes `.position` with the same key set as the initial position,
raising a named-keys `TypeError` for malformed `build_fn` results);
`BlackjaxNUTSKernel(num_adaptation_steps, target_acceptance_rate)`;
`BlackjaxMCLMCKernel(num_tuning_steps)`; `BlackjaxCustomKernel(build_fn)`.
Bodies exactly as v2 §3.2–3.3; the only v3 change is that the state
validation moved from "custom-kernel test wish" into the base `init` so all
three concrete kernels share it.

Postprocessing contract (release gate, unchanged): key set of
`MCMC.get_samples()` for a blackjax kernel on the reparam-bearing
`UnivariateForecaster` **equals** the NUTS key set on the same model.

### 3.3 Pathfinder (final; v2 §3.4 with R7 fixes locked in)

`PathfinderFit(state, model, covariates, data, elbo)` — plain-data frozen
dataclass, picklable. `fit_pathfinder(...)` per v2. Draw registration vmaps
`constrain_fn` per sample (no `batch_ndims` reliance). Two ⚙ canaries: the
`approximate` return structure (where the ELBO path lives) and `sample`'s
return arity — pinned by the first implementation PR, guarded thereafter.
`PathfinderForecaster` shim: constructor mirrors `Forecaster` minus
`optim`/`guide`, plus `num_elbo_samples`/`ftol`.

### 3.4 Implementation checklist
1. `typing.py`: `KernelLike`, `BuildFn`.
2. `functional.py`: `resolve_kernel`, `_validate_kernel_run_config`,
   `fit_mcmc` rewrite, `MCMCFit.num_chains` field.
3. `contrib/__init__.py` + `contrib/blackjax.py` (kernels, then pathfinder,
   then the `_draw_posterior_impl` registration — import-order-free since
   all are in one module).
4. `forecaster.py`: `HMCForecaster(kernel=..., kernel_kwargs=...)`;
   new `PathfinderForecaster`.
5. Docs: "Samplers" page with the scope table + one example per row;
   "How extension modules work" (registration-by-import contract).

### 3.5 Tests — `tests/test_kernels.py`, `tests/contrib/test_blackjax.py`
All of v2 §3.6, organized:
- resolver: none/class/instance/instance+kwargs-raises/typeerror; NUTS
  kwargs through the `None` path.
- run-config validation: AIES single-chain; AIES non-vectorized; blackjax
  non-sequential; blackjax warmup warning.
- NumPyro kernels: `@parametrize(NUTS, HMC, BarkerMH, SA)` fit+forecast
  shape contract; `AIES` vectorized smoke; `HMCGibbs`/`HMCECS` instance
  smokes.
- blackjax (skip-marked, extras CI leg): known-posterior recovery
  (Normal–Normal closed form, MC tolerance); **key-set equality vs NUTS on
  the reparam model** (release gate); MCLMC end-to-end forecast finite +
  CRPS within 2× NUTS; custom-kernel happy path + malformed-state named
  error; pathfinder constrained-support test (sigma > 0), forecast
  composition, picklability, backtest-as-`forecaster_fn` structural test;
  ⚙ API canaries.

**Acceptance:** `fit_mcmc(kernel=BlackjaxMCLMCKernel, kernel_kwargs=...,
num_warmup=0)` → `forecast` on the univariate fixture, green on the extras
leg; base leg unaffected (no blackjax import at package import time).

---

## 4. Performance

### 4.1 Walltime (final; v2 §4.1)

`_timed` gains `jax.block_until_ready(result)`; `_block_object(obj)`
materializes `vars(obj)` leaves (pytree-generic, slots-safe no-op). Call
sites: training timing wraps `_block_object(forecaster_fn(...))`; forecast
timing already returns arrays. Release note: reported walltimes step up
(K6). Test: `test_walltime_includes_compute` (`@slow`), plus
`test_block_object_handles_frozen_dataclass_and_shim`.

### 4.2 Chunk padding (final; v2 §4.2)

`_pad_posterior(posterior, batch_size) -> (padded, num)` with the three
`ValueError` guards (empty dict; `batch_size <= 0`; sample-axis
disagreement across leaves, error names offending sites) and the PRNG-
semantics note promoted onto `forecast`'s docstring ("chunking is a memory
knob, not a reproducibility knob; reproducibility is per `(key,
batch_size)`"). Revised chunk loop:

```python
if batch_size is not None:
    padded, num = _pad_posterior(posterior, batch_size)
    chunks = [
        _predict(rng, model, _index_tree(padded, s, batch_size), cov, parallel=parallel)
        for rng, s in zip(random.split(rng_key, num_padded // batch_size),
                          range(0, num_padded, batch_size))
    ]
    return jnp.concatenate(chunks, axis=0)[:num]
```

Tests: chunked-vs-unchunked statistical closeness + shape equality;
property sweep `num_samples ∈ {1, b-1, b, b+1, 3b}`; **exactly one**
`_predict` compilation across the sweep for fixed shapes
(compile-count harness); the three guard errors.

### 4.3 `backtest_vectorized` — full scaffold (C1, C2; K2 retired)

```python
def backtest_vectorized(
    rng_key: Array,
    data: Array,
    covariates: Array,
    model_fn: ModelFactory,
    *,
    train_window: int,
    test_window: int,
    stride: int = 1,
    num_steps: int = 1_001,
    optim: "OptimizerLike" = None,
    guide: "GuideLike" = None,
    num_samples: int = 100,
    metrics: Mapping[str, Metric] | None = None,
    keep_predictions: bool = False,
) -> VectorizedBacktestResult:
    """Rolling-window backtest with all windows fitted in one vmapped SVI run.

    Estimator-equivalent to :func:`backtest` with rolling windows; differs
    only in PRNG stream layout and float reduction order (documented — the
    equivalence test is statistical, not bitwise). Model, guide, and SVI
    compile once; expect order-of-magnitude wall-clock wins for tens of
    windows on small models.

    Constraints (each its own ``ValueError``): fixed ``train_window`` and
    ``test_window`` >= 1; ``stride`` >= 1; the resolved guide must be an
    ``AutoGuide`` (hand-written guides: use :func:`backtest`);
    ``model_fn`` is called exactly once (per-window model variation: use
    :func:`backtest`); no per-window ``forecaster_options``.

    PRNG: ``fold_in(rng_key, -1)`` for the warm-up init (discarded),
    ``fold_in(rng_key, i)`` per window ``i`` for fit, and a split of each
    window key for posterior draws and forecasting.
    """
    starts = jnp.arange(0, duration - train_window - test_window + 1, stride)
    train_d, train_c, hor_c, truth = jax.vmap(_slice_window)(starts)  # dynamic_slice_in_dim, §v1

    model = model_fn()
    resolved_guide = resolve_guide(guide, model)
    if not isinstance(resolved_guide, AutoGuide):
        raise ValueError(_VECTORIZED_NEEDS_AUTOGUIDE_MSG)
    svi = SVI(model, resolved_guide, resolve_optimizer(optim), Trace_ELBO())

    # K11 — MANDATORY eager warm-up on concrete window-0 arrays: AutoGuide
    # caches its prototype on the instance at first init; if that first
    # init is traced under vmap, the instance holds leaked tracers and any
    # later eager use raises UnexpectedTracerError (spike-pinned).
    svi.init(random.fold_in(rng_key, -1),
             train_c[0], train_d[0])

    def fit_one(key, d, c):
        state = svi.init(key, c, d)
        def step(s, _):
            s, loss = svi.update(s, c, d)
            return s, loss
        state, losses = lax.scan(step, state, length=num_steps)
        return svi.get_params(state), losses

    window_keys = jax.vmap(lambda i: random.fold_in(rng_key, i))(
        jnp.arange(starts.shape[0]))
    params, losses = jax.jit(jax.vmap(fit_one))(window_keys, train_d, train_c)

    post_keys, fc_keys = ...  # vmapped split of window_keys
    posterior = jax.vmap(
        lambda k, p: resolved_guide.sample_posterior(k, p, sample_shape=(num_samples,))
    )(post_keys, params)                                   # spike: bitwise pure

    def forecast_one(key, post_w, d, hc):
        pred = Predictive(model, posterior_samples=post_w,
                          return_sites=["forecast"])
        return pred(key, hc, d)["forecast"]                # spike F3: nests fine

    predictions = jax.jit(jax.vmap(forecast_one))(fc_keys, posterior, train_d, hor_c)

    metric_values = _vmapped_metrics(metrics or DEFAULT_METRICS, predictions, truth)
    return VectorizedBacktestResult(
        t0=starts, t1=starts + train_window, t2=starts + train_window + test_window,
        losses=losses, metrics=metric_values,
        predictions=predictions if keep_predictions else None,
    )
```

Private helpers: `_slice_window(t0)` (four `dynamic_slice_in_dim`s over the
closed-over full arrays); `_vmapped_metrics` (vmap each jitted metric
kernel over the window axis; single device→host transfer at the end,
mirroring `evaluate_forecast`'s fast path). `VectorizedBacktestResult`
dataclass as v2, plus `to_dataframe()`-compatibility via §10.

Implementation checklist:
1. `evaluate.py`: result dataclass, validators (each constraint one test),
   `_slice_window`, warm-up line **with the K11 comment verbatim**,
   the three vmapped stages, `_vmapped_metrics`.
2. `spikes/spike_vmap_svi.py` → `tests/test_vmap_svi_invariants.py`
   (§11.3): warm-up-fix test, contamination counterfactual
   (`pytest.raises(UnexpectedTracerError)`), `sample_posterior` purity,
   plan-B continuation (documents the fallback stays viable).

Tests — `tests/test_backtest_vectorized.py` (v2 set, now unconditional):
window-index equality with loop `backtest`; per-window CRPS/MAE
statistical closeness (rtol 1e-1, generous SVI budgets); every validator;
window-count formula property test over a `(duration, train, test, stride)`
grid; **compile discipline: exactly one SVI-fit compilation for
`num_windows ∈ {2, 5, 10}`** (the feature's raison d'être, CI-enforced);
`@slow` 2× speedup sanity for 10 windows.

**Acceptance:** univariate fixture, 8 rolling windows: metrics match the
loop path statistically; compile count 1; no `UnexpectedTracerError` on a
subsequent eager `fit_svi` with a fresh guide in the same process.

### 4.4 Loop-`backtest` model reuse (final; v2 §4.5)

`reuse_model: bool = True` effective only for rolling windows; docstring
states precisely what it buys (forecast/predict kernel cache hits; SVI
still recompiles per window — §4.3 is the fix for that). Compile-count
test asserts forecast-kernel compilations == 1 and SVI compilations ==
`num_windows` across a 5-window run.

### 4.5 Compile-count harness (final; v2 §4.6)

`count_compilations` conftest fixture: `jax.monitoring.register_event_listener`
primary; suite-local `Lowered.compile` wrap fallback; harness
unavailability = `xfail(strict=False)` with a warning, never silent skip;
⚙ canary named `test_compile_harness_backend_available`.

### 4.6 Removed/deferred
Buffer donation stays out (v2 R9); `scripts/bench_memory.py` remains the
instrument for revisiting.

---

## 5. ArviZ export (DataTree-first)

**Goal.** One-call conversion to a schema-compliant `xarray.DataTree`
(ArviZ ≥ 1.0), with a forecast group and a deprecated 0.x shim.
**Files:** `convert.py`, `functional.py` (`MCMCFit.num_chains` — done in §3).

### 5.1 Scaffolds (final)

```python
def to_datatree(
    rng_key: Array,
    fit: object,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    num_predictive_samples: int | None = None,
    coords: Mapping[str, Sequence] | None = None,
    time_coord: Sequence | None = None,
) -> "xarray.DataTree":
    """Convert a fit into an ArviZ-schema DataTree.

    Groups: ``posterior`` (``(chain, draw, ...)``; true chains for
    :class:`MCMCFit`, one pseudo-chain + ``variational: true`` attrs for
    SVI/Pathfinder), ``posterior_predictive`` (in-sample ``obs`` via
    :func:`predict_in_sample`, dims ``(chain, draw, time, obs_dim)``),
    ``observed_data`` (``(time, obs_dim)``), ``constant_data``
    (covariates, ``(time, covariate_dim)``). Tree attrs:
    ``inference_library``, ``creation_library``, ``sample_dims``.

    PRNG: one stream, for the posterior-predictive draws.

    Raises
    ------
    ImportError
        Without ``arviz-base`` (``pip install 'numpyro_forecast[arviz]'``).
    """


@singledispatch
def _posterior_reshape(fit: object, samples: dict[str, Array]) -> dict[str, Array]:
    """(num_samples, ...) -> (chain, draw, ...) per fit type.

    Default: single pseudo-chain, ``leaf[None]``. ``MCMCFit`` override:
    ``leaf.reshape(fit.num_chains, -1, *leaf.shape[1:])`` — valid because
    ``get_samples(group_by_chain=False)`` concatenates chains in order
    (pinned by ``test_mcmc_chain_reshape_roundtrip`` against
    ``group_by_chain=True``).
    """


def add_forecast(
    tree: "xarray.DataTree",
    forecast_samples: Array,          # (num_samples, future, obs) from forecast()
    covariates_future: Array,
    *,
    time_coord: Sequence | None = None,
) -> "xarray.DataTree":
    """Attach ``predictions`` + ``predictions_constant_data`` groups; the
    ``time`` coordinate continues the in-sample one (integer continuation
    by default, explicit values via ``time_coord``). Returns a new tree."""


def to_inferencedata(*args, **kwargs) -> "arviz.InferenceData":
    """Legacy shim: ``to_datatree`` -> ``InferenceData.from_datatree``.
    Requires classic arviz (< 1.0) with xarray >= 2024.11; FutureWarning
    pointing at :func:`to_datatree`; actionable ImportError otherwise."""
```

Group construction rule (normative): **every group goes through
`arviz_base.dict_to_dataset`** (never hand-rolled `xarray.Dataset`s — it
owns dim naming, `sample_dims` handling, and schema evolution), assembled
with `xarray.DataTree.from_dict`. Dims plumbing: pass
`dims={"obs": ["time", "obs_dim"], ...}` and merged `coords` (user
`coords` > generated `time` > defaults).

### 5.2 Implementation checklist
1. `convert.py`: the four callables + `_group_datasets(fit, ...)`
   assembler; lazy `require("arviz_base", extra="arviz")`.
2. `__init__.py`: export `to_datatree`, `add_forecast`.
3. CI: extras leg runs full convert suite; a *legacy-arviz* leg (classic
   `arviz` 0.x + xarray ≥ 2024.11) runs only the shim test.
4. Docs: diagnostics page — fit → `to_datatree` → `arviz_stats.ess/rhat`
   → `add_forecast` → plotting teaser.

### 5.3 Tests — `tests/test_convert.py`
v2 §5.5 verbatim (schema/dims per fit type; 2-chain reshape round-trip;
variational metadata; `ess`/`rhat` execute on the posterior node —
the acceptance test; exact observed/constant round-trips; time-coord
threading; `add_forecast` groups + time continuation; shim on the legacy
leg), plus:
- `test_all_groups_via_dict_to_dataset` — monkeypatch-spy asserting no
  group bypasses the constructor (guards the normative rule).

**Acceptance:** `az.rhat(to_datatree(...)["posterior"].dataset)` runs
warning-free for a 2-chain MCMC fit of the univariate fixture.

---

## 6. Elementwise registry, noise families, `predict_glm`

**Goal.** Open, sound time-axis surgery for elementwise families; count
observations. **Files:** `util.py`, `functional.py`.

### 6.1 Registry + structural check (final; C9)

```python
_ELEMENTWISE_FAMILIES: set[type[dist.Distribution]] = set()
_ELEMENTWISE_CHECKED: set[type[dist.Distribution]] = set()


def register_elementwise(family: type[dist.Distribution]) -> type[dist.Distribution]:
    """Declare a family safe for generic time-axis surgery (v2 §6.1
    docstring: definition, what it unlocks in slice_time/prefix_condition,
    when to register concrete implementations instead, decorator usage).
    Membership is by exact type — a subclass may add structural parameters,
    so elementwise-ness is never inherited (deliberate divergence from
    singledispatch's MRO behavior, documented)."""
    _ELEMENTWISE_FAMILIES.add(family)
    return family


def _check_elementwise(noise_dist: dist.Distribution) -> None:
    """First-use structural verification per family (K7 mitigation).

    Cheap, trace-time-only: asserts ``event_shape == ()`` and that every
    ``arg_constraints`` parameter broadcasts against ``batch_shape``
    (``jnp.broadcast_shapes`` raises otherwise). On failure: raise
    ``TypeError`` naming the family, the offending parameter, and the
    remedy ("this family is not elementwise; register concrete
    implementations instead"). Caches the pass in ``_ELEMENTWISE_CHECKED``.
    """
```

Generic defaults (final): `slice_time` — membership check →
`_check_elementwise` → broadcast-and-slice body (as today);
`prefix_condition` — membership check → `slice_time(d, slice(data.shape[-2], None))`
(independence ⇒ future marginal; the R1 fix). Redundant Normal/StudentT
`prefix_condition` registrations deleted in the same PR under the
trace-equivalence test. `shift_loc` stays pure dispatch; new registrations:
Laplace, Cauchy, AsymmetricLaplace, Gumbel (bodies as v1 §6.1).
Registered elementwise at import: Normal, StudentT, Laplace, Cauchy,
AsymmetricLaplace, Gumbel, Poisson, NegativeBinomial2.

### 6.2 `predict_glm` (final; body as v1 §6.2 + v2 additions)

```python
def predict_glm(
    h: Horizon,
    obs_dist_fn: Callable[[Array], dist.Distribution],
    latent: Array,
) -> None:
    """GLM-style observation/forecast sites (docstring: v1 §6.2 — the
    link-function contract, prefix/suffix mirroring of predict, the
    slice_time requirement).

    Additional validation (v2): if the observation distribution's support
    is discrete, ``h.data`` must be integer-dtyped — checked here with a
    targeted message (the most common user error for count models).
    """
```

`predict` refactor: delegate to
`predict_glm(h, lambda mu: shift_loc(noise_dist, mu), prediction)`, gated by
`test_predict_predict_glm_trace_equivalence` (seeded `handlers.trace`
comparison — site names, values, log_probs identical). `ForecastingModel`
gains the `predict_glm` forwarder (shim discipline).

### 6.3 Implementation checklist
1. `util.py`: registry + check + generic rewrites + `shift_loc`
   registrations + deletions; export `register_elementwise`.
2. `functional.py`: `predict_glm`; `predict` delegation.
3. `forecaster.py`: forwarder.
4. Docs: "Supported noise families" table generated from the registries
   (doctest-checked so the table can't drift); count-model tutorial stub
   (Poisson local level).

### 6.4 Tests — `tests/test_util.py`, `tests/test_functional.py`, `tests/test_consistency.py`
v1 §6 + v2 §6.3 sets, consolidated: `shift_loc` log_prob-shift identity
over all registered families incl. `Independent(Laplace)`; slice
correctness per elementwise family vs manual slicing; Poisson
`prefix_condition` == future marginal (R1 regression); Dirichlet →
actionable error; user-family decorator registration; Normal-subclass
non-inheritance; `_check_elementwise` failure on a deliberately structural
fake family (named param in the error); discrete-dtype validation both
directions; Poisson end-to-end (integer, non-negative forecast samples;
rate recovery on a step series); trace equivalence. **Registry-consistency
test** (owns invariant I4, §11.1): no family in a partial registration
state across the three dispatchers + the elementwise set.

**Acceptance:** the Poisson local-level example from the docs runs green
as a doctest; `predict` behavior bit-identical pre/post refactor.

---

## 7. `markov_time_series` (spike-validated, final)

**Goal.** State-space latents with carried state; forecast scan seeded by
the posterior-conditioned final in-sample carry. All three spike questions
GREEN (C3); the design below is buildable as written. **Files:**
`functional.py`, `forecaster.py`.

### 7.1 Layout contract (C6, C7 — normative)

- **Storage layout (sites):** scan layout, time leading —
  `(t, *plate_batch, obs)` for the in-sample site, `(f, ...)` for the
  future site. Posterior dicts hold scan layout; `Predictive` replays them
  unmodified (spike S-M1b), so **no conversion happens on the replay
  path**.
- **Model-body layout (return value):** package layout — the primitive
  returns `moveaxis(concat, 0, -2)` so downstream `predict`/`predict_glm`
  see `(*plate_batch, duration, obs)`.
- **Per-step shape rule (C7):** the transition's distribution must carry
  the trailing observation dim — per-step shape `(*plate_batch, obs)`,
  `obs >= 1`. This makes the single `moveaxis(0, -2)` unambiguous in every
  case (unbatched `(t, obs)` is already package layout; batched
  `(t, B, obs)` → `(B, t, obs)`). Validated at trace time: a per-step
  event/batch shape of `()` raises with "add the observation dimension,
  e.g. loc=...[..., None]".

### 7.2 Scaffold

```python
Transition = Callable[[PyTree, Array | None], tuple[dist.Distribution, Callable[[Array], PyTree]]]
"""(carry, x_t) -> (dist_t, carry_fn) where carry_fn(z_t) builds the next
carry from the *sampled* latent. The wrapper owns the sample statement, so
user code cannot break the Markov structure by resampling (v2 spike option
A, now fixed as the contract)."""


def markov_time_series(
    h: Horizon,
    name: str,
    init_carry: PyTree,
    transition: Transition,
    xs: Array | None = None,
    *,
    plates: Sequence[tuple[str, int]] = (),
    reparam_config: Mapping[str, "numpyro.infer.reparam.Reparam"] | None = None,
) -> Array:
    """Sample a Markov (state-space) latent over the full horizon.

    In-sample steps run in a ``numpyro.contrib.control_flow.scan`` with
    site ``name``; when forecasting, horizon steps run in a second scan
    with site ``f"{name}_future"`` **seeded by the final in-sample carry**
    — the guide never sees the future site (same invariant as
    :func:`time_series`), and under posterior replay the carry is a
    deterministic function of the replayed draws, so the forecast is
    conditioned through the state (spike-verified: forecast step k tracks
    ``phi^k * z_last`` on the AR(1) closed form).

    Parameters
    ----------
    h, name, init_carry, transition, xs
        As documented on :data:`Transition`; ``xs`` covers the full horizon
        with time at axis ``-2`` and is split/moved into scan layout
        internally; ``None`` for autonomous dynamics.
    plates
        ``(name, size)`` pairs opened **inside** the scan body around the
        sample statement. NumPyro rejects ``plate`` *wrapping* ``scan``
        outright (spike-pinned); this argument makes the only working
        placement the only expressible one. An enclosing user plate is
        detected at trace time and re-raised with guidance.
    reparam_config
        Site-name -> Reparam mapping applied **inside** the scan body.
        Outside placement fails in NumPyro (the handler intercepts the
        scan-level stacked message — spike-pinned); inside placement is
        fully functional: the trace exposes ``{name}_decentered`` as the
        sample site and ``name`` as deterministic, SVI fits the decentered
        site, and replay recomputes ``name`` (spike max posterior-mean
        error 0.029 on near-noiseless data).

    Returns
    -------
    Array
        The latent over the full horizon in package layout
        ``(*plate_batch, duration, obs)``.

    Raises
    ------
    ValueError
        If forecasting without observed data; if the per-step shape lacks
        the observation dim (C7); if an enclosing plate is detected.
    """
    def _body(site_name):
        def body(carry, x_t):
            dist_t, carry_fn = transition(carry, x_t)
            ctx = reparam(config=dict(reparam_config)) if reparam_config else nullcontext()
            with ctx, _plate_stack(plates):
                z = numpyro.sample(site_name, dist_t)
            return carry_fn(z), z
        return body

    xs_scan = None if xs is None else jnp.moveaxis(xs, -2, 0)
    final_carry, zs = scan(_body(name), init_carry,
                           None if xs_scan is None else xs_scan[: h.t_obs],
                           length=h.t_obs if xs_scan is None else None)
    if h.future == 0:
        return jnp.moveaxis(zs, 0, -2)
    _, zf = scan(_body(f"{name}_future"), final_carry,
                 None if xs_scan is None else xs_scan[h.t_obs :],
                 length=h.future if xs_scan is None else None)
    return jnp.moveaxis(jnp.concatenate([zs, zf], axis=0), 0, -2)
```

Private helpers: `_plate_stack(plates)` (nested `numpyro.plate` context
manager, innermost-last); `_reject_enclosing_plates()` (inspects
`numpyro.primitives._PYRO_STACK` for plate messengers — private API, so it
is (a) wrapped in try/except falling back to NumPyro's own error and (b)
covered by a ⚙ canary); per-step shape validation on the first traced
step.

OOP shim: `ForecastingModel.markov_time_series(...)` forwards with
`self._horizon` as always.

### 7.3 Implementation checklist
1. `functional.py`: `Transition` alias, the primitive, helpers.
2. `forecaster.py`: forwarder.
3. `spikes/spike_scan_replay.py` → `tests/test_markov_spike_invariants.py`
   (§11.3): layout, replay, carry-threading (extreme-state), batched
   carry-threading, reparam placement (both the inside-works and the
   outside-raises halves), plate-wrap rejection.
4. Docs: state-space tutorial — local level via `markov_time_series`,
   exponential smoothing on Victoria electricity, the §7.4 guide note.

### 7.4 Guide note (v2 §7.2, unchanged, now docs content)
Auto guides fit the stacked scan site as one factorized block —
statistically valid mean-field over the joint path, not a filtering
posterior; MCMC unaffected; the AR(1) closed-form test quantifies quality.

### 7.5 Tests — `tests/test_markov.py`
- `test_ar1_forecast_matches_closed_form` — posterior-mean forecast decays
  as `phi^k * z_last` (tight tol; SVI and NUTS parametrized).
- `test_local_level_equivalent_to_time_series_cumsum` — same model both
  ways, statistical equality of forecast distributions.
- `test_carry_threading_extreme_state` — the loud S-M2 regression:
  horizon-1 forecast near `phi * z_last`, far from the unconditional mean.
- `test_batched_plates_argument` — `(B,)` plate: shapes
  `(B, duration, obs)` out; site stored `(t, B, obs)`.
- `test_missing_obs_dim_raises_with_guidance` (C7).
- `test_enclosing_plate_rejected_with_guidance`.
- `test_reparam_config_end_to_end` — decentered site fitted, replay tracks
  data.
- `test_xs_threading` — driven dynamics: xs sliced/moved correctly (an
  xs-dependent mean recovered).
- `test_exponential_smoothing_victoria` (`@slow`) — the notebook
  construction as a regression target.

**Acceptance:** AR(1) closed-form test green under both SVI and NUTS; the
spike-invariant suite green (it is the NumPyro-version canary for this
feature — K10).

---

## 8. Metrics (final; v2 §8)

`_pinball` (jitted, quantile static — one compile per distinct level,
documented); `eval_pinball(quantile=0.5)` with `(0,1)` validation;
`eval_interval_score(alpha=0.9)` (Winkler; closed-form tested);
`make_mase(train_data, seasonality=1)` factory (validates
`seasonality >= 1`, `train length > seasonality`, nonzero scale **at
factory time**); `backtest(per_window_metrics=...)` hook merging over
`metrics`. Tests as v2 §8 (identity/asymmetry/closed-form/scale/
constant-series/hook-key-presence/large-sample dtype finiteness).
`eval_pinball_multi` stays demand-gated (§11.5).

---

## 9. `MultivariateNormal` registrations (final; v2 §9)

Same-PR set: `shift_loc` (loc shift), `slice_time` (marginal block: slice
`loc`; slice both covariance axes), `prefix_condition` (Cholesky-solve
Gaussian conditional, symmetrized + jitter floor). v1 layout restriction:
`obs == 1`, time as the event dim; anything else raises naming the
restriction. Tests: closed-form conditional vs `numpy.linalg` on an
AR-structured 5-step covariance; `MVN(mu, sigma^2 I)` conditional ==
`Independent(Normal)` slice (log_prob grid equality); near-singular
`Sigma_oo` (cond 1e10) → valid distribution via jitter; sliced-MVN
log_prob vs scipy; end-to-end GP-noise model through `fit_svi` →
`forecast`. Registry-consistency test (I4) covers the all-three-or-none
rule automatically.

---

## 10. `results_to_dataframe` (final; v2 §10)

One overloaded free function accepting `Sequence[BacktestResult]` or
`VectorizedBacktestResult`, one row per window, prefix-namespaced columns
(`metric_*`, `train_metric_*`, `param_*`, plus `t0/t1/t2`,
`num_samples`, walltimes where available — vectorized results have no
per-window walltimes; the columns are simply absent, documented).
Predictions excluded. pandas via `require`. Tests: schema per input type,
row counts, empty-metrics windows, missing-pandas error, loop/vectorized
column compatibility (same metric columns for the same metric set).

---

## 11. Cross-cutting invariants, spike provenance, risk register, delivery plan

### 11.1 Invariants (each owned by one named test — C10)

| ID | Invariant | Owning test |
|----|-----------|-------------|
| I1 | No guide (any flavor) ever contains a `*_future` site | `test_guide_never_sees_future_sites[*]` |
| I2 | Every fit type's posterior dict keys == the key set `Predictive` expects for the model (incl. reparam `_decentered` sites) | `test_blackjax_keyset_equals_nuts`, `test_pathfinder_forecast_composes`, guide suite |
| I3 | Fixed shapes ⇒ fixed compile counts: 1 forecast kernel across padded chunks and rolling windows; 1 vmapped-SVI compile regardless of window count | `test_single_compile_while_chunking`, `test_rolling_backtest_predict_cache_hits`, `test_single_svi_compilation` |
| I4 | No noise family in a partial registration state (elementwise set + three dispatchers) | `test_registry_consistency` |
| I5 | Scan-site storage layout is scan layout; conversion happens only in `markov_time_series`'s return | `test_markov_spike_invariants::test_replay_layout` + `test_batched_plates_argument` |
| I6 | AutoGuide instances are never first-initialized under a transformation trace (warm-up rule, K11) | `test_vmap_svi_invariants::test_contamination_counterfactual` + code-review rule: any new vmapped-SVI path must cite this test |
| I7 | Every public array-crossing function documents its PRNG consumption; `eval_train` and warm-up streams are `fold_in`-isolated | docstring lint (docs build) + `test_eval_train_key_isolation` |
| I8 | Optional deps never imported at package import time | base CI leg (`test_base_import_no_extras`) |
| I9 | `predict` ≡ `predict_glm∘shift_loc` (post-refactor) | `test_predict_predict_glm_trace_equivalence` |

### 11.2 Risk register — final state

| ID | Risk | Status / handling |
|----|------|-------------------|
| K1 | blackjax API drift | open, mitigated: pin `>=1.2,<2`; ⚙ canaries on extras leg name the exact symbols |
| K2 | vmap-over-SVI unsound | **retired** (spike S-V: F1 green for AutoNormal + AutoMVN; F2 bitwise pure; F3 green; plan-B validated as unused fallback) |
| K3 | scan replay/carry unsound | **retired** (spike S-M: replay unmodified; carry threads to <0.1% of closed form; batched case green) |
| K4 | ArviZ 0.x/1.x bifurcation | open, mitigated: DataTree primary; shim + legacy CI leg |
| K5 | blackjax kernel state under non-sequential chains | validated away at entry; stateless-state redesign queued post-soak |
| K6 | walltime step-change | accepted; release-noted |
| K7 | user elementwise misdeclaration | mitigated: `_check_elementwise` first-use structural check (§6.1) |
| K8 | numpyro utility drift | open, mitigated: floor `>=0.21`; key-set gate + spike-invariant suites are the contract |
| K9 | AutoDelta tiling misread as posterior spread | accepted; docs + `variational: true` DataTree attrs |
| K10 | scan behaviors are NumPyro-internals-sensitive (reparam placement, plate rejection, replay layout) | mitigated: spike scripts converted to permanent canary tests (§11.3) |
| K11 | **AutoGuide prototype contamination under transformation tracing** (new, from S-V): first `svi.init` inside vmap leaves leaked tracers on the instance; later eager use raises `UnexpectedTracerError` | mitigated by the mandatory warm-up rule in §4.3 (comment-in-code + I6 counterfactual test). Note this risk applies to *any* future vmapped-inference feature, hence invariant status |

### 11.3 Spike provenance → permanent tests

`spikes/spike_scan_replay.py` and `spikes/spike_vmap_svi.py` are kept
verbatim (provenance, with the recorded jax 0.10.2 / numpyro 0.21.0
environment) and *ported* — not linked — into
`tests/test_markov_spike_invariants.py` and
`tests/test_vmap_svi_invariants.py`: same assertions, fixture-sized,
deterministic seeds, each test docstring citing the spike finding it pins.
These two files are the ⚙ canaries for K8/K10/K11: a NumPyro or JAX
upgrade that breaks a spike-established behavior fails a named test whose
docstring explains exactly which design assumption just broke.

### 11.4 Delivery plan (PR-sized, ordered by dependency)

**Phase 1 — resolution layer & core hardening** (independent, revertable)
| PR | Contents | Gate |
|----|----------|------|
| P1 | `resolve_optimizer` + `fit_svi` signature + `SVIFit` fields (§1) | test_optim green |
| P2 | `resolve_guide` + probe + AutoDelta dispatch (§2) | I1 green all flavors |
| P3 | `resolve_kernel` + validation + `MCMCFit.num_chains` (§3.1) | scope-table smokes |
| P4 | walltime fix + `_block_object` (§4.1) | release note drafted |
| P5 | chunk padding + posterior validation (§4.2) | I3 chunk test |
| P6 | elementwise registry + families + generics rewrite + deletions (§6.1) | I4 + R1 regression |
| P7 | metrics + `per_window_metrics` (§8) | closed-form tests |
| P8 | `results_to_dataframe` (§10, loop-result half) | schema tests |

**Phase 2 — extras & new modeling surface** (needs P1–P6)
| PR | Contents | Gate |
|----|----------|------|
| P9 | compile-count harness + I3 tests retrofit (§4.5) | harness canary |
| P10 | `contrib/blackjax` kernels incl. custom (§3.2) | I2 key-set gate |
| P11 | Pathfinder + forecaster shim (§3.3) | constrained-support + pickle |
| P12 | `to_datatree` + `add_forecast` + shim + CI legs (§5) | ess/rhat acceptance |
| P13 | `predict_glm` + `predict` refactor (§6.2) | I9 |

**Phase 3 — flagship features** (spike-cleared; needs P2, P5, P9)
| PR | Contents | Gate |
|----|----------|------|
| P14 | spike-invariant test suites (§11.3) | both green on pinned floors |
| P15 | `backtest_vectorized` + vectorized dataframe half (§4.3, §10) | I3 + I6 + equivalence |
| P16 | loop-backtest `reuse_model` (§4.4) | I3 rolling test |
| P17 | `markov_time_series` + tutorial (§7) | AR(1) closed form, both inference paths |
| P18 | MVN registrations (§9) | I4 auto-covers; numerics tests |

P14 lands *before* P15/P17 deliberately: the invariant suites must be in
CI before the features that depend on them, so a dependency upgrade in the
gap cannot silently invalidate the spike evidence.

### 11.5 Out of scope (final)
Buffer donation (benchmark-gated); NestedSampler backend (recorded:
`contrib.jaxns` following the Pathfinder pattern); FlowMC / bayeux;
stringly-typed sampler dispatch; `eval_pinball_multi` (demand-gated);
blackjax stateless-state redesign for parallel chains (queued post-soak,
K5); filtering-flavored guides for scan sites (research direction, §7.4).
