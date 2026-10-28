# Scaffolding: forecast log densities and model stacking

Companion to `docs/dev/stacking_design.md`. That document says what and why; this one says where every piece lives, what its exact signature and type contract is, how each kernel is to be implemented, and which test pins which property. Everything described here exists on the `stacking-scaffold` branch: the public surface is declared with full signatures, docstrings and jaxtyping annotations, the plumbing that needs no algorithm is implemented, every kernel is a `raise NotImplementedError`, and the two new test modules are written against the declared surface with a strict `xfail(raises=NotImplementedError)` on each test that reaches a stub. The branch passes `ruff check`, `ruff format --check`, `ty check` (the same pre-existing diagnostics as `main`, none new), `tests/test_docs_reference.py`, `tests/test_docstring_markup.py`, `tests/test_package.py`, `tests/test_api_snapshot.py`, and the existing `tests/test_evaluate.py`, `tests/test_backtest_vectorized.py` and `tests/test_predictive.py`; the new tests report 7 passes (the implemented validation paths and the trace-level mechanism check) and 84 strict xfails. Implementing a kernel turns its xfails into failures until the mark is removed, which is the intended workflow: delete the `@scaffold` mark of a test when its kernel lands and make it pass.

## 1. Inventory of the branch

| File | Change | State |
| --- | --- | --- |
| `numpyro_forecast/typing.py` | `Reduction`, `GroupPrior`, `DensityReader` aliases; `ForecastOutput` NamedTuple; `ForecastFn.__call__` may return `ForecastOutput` | complete |
| `numpyro_forecast/exceptions.py` | `DensitySiteError(NumpyroForecastError, ValueError)`, `StackingShapeError(NumpyroForecastError, ValueError)` | complete |
| `numpyro_forecast/predictive.py` | module docstring paragraph; `_FUTURE_OBS_SITE`; `_site_log_prob`, `predict_density`, `ssoe_density`, `_log_density`, `forecast_log_density`, `reduce_log_density`, `log_density_ess`, `conditional_log_density` | signatures and docstrings complete; `predict_density` raises `DensitySiteError` on a missing site then stubs; all other bodies stubbed |
| `numpyro_forecast/evaluate.py` | `BacktestResult.log_density`, `.log_density_reduction`; `VectorizedBacktestResult.log_density`, `.log_density_reduction`; `backtest(log_density=, density_reader=)`; `_run_window` wiring; `_unpack_forecast_output`; `backtest_vectorized(log_density=, density_reader=)`; `_vectorized_log_density` | wiring and `_unpack_forecast_output` implemented; `_vectorized_log_density` stubbed |
| `numpyro_forecast/stacking.py` | new module: `MixturePer`, `StackingCells`, `stacking_cells`, `window_design_effect`, `log_score`, `stacking_weights`, `pseudo_bma_weights`, `mixture_crps_terms`, `crps_stacking_weights`, `rectified_features`, `hierarchical_weights`, `stacking_factor`, `stack_forecasts` | signatures and docstrings complete; all bodies stubbed |
| `numpyro_forecast/__init__.py` | imports and `__all__` entries for every new public name | complete |
| `great-docs.yml` | sections "Forecast log densities" and "Model stacking"; new entries under "Typing" and "Exceptions" | complete |
| `tests/test_density.py` | 38 tests for readers, driver, reductions, chain-rule route, backtest hook | written; 6 pass, 32 xfail |
| `tests/test_stacking.py` | 53 tests for cells, solvers, blocks, mixture draws | written; 1 passes, 52 xfail |
| `tests/test_evaluate.py`, `tests/test_package.py` | the pinned `BacktestResult` field set and the pinned `__all__` extended | complete |
| `docs/dev/stacking_design.md`, `docs/dev/stacking_scaffolding.md` | this pair | complete |

Not on the branch, to be done per PR: the `skill:` block of `great-docs.yml`, the `AGENTS.md` conventions (Section 7 below), the notebooks, and `.test_durations` after the slow tests land.

## 2. Type contracts

### 2.1 Shape vocabulary

The jaxtyping shape strings reuse the package's names and add a few. Every string has the leading space the `F722` rule requires.

| Axis name | Meaning |
| --- | --- |
| `sample` | posterior draws, always first on a per-draw array |
| `*batch` | the model's batch dimensions (series plates), left of time, possibly empty |
| `future` | the forecast horizon, `t2 - t1` in a backtest window; time sits at axis `-2` as everywhere in the package |
| `obs` | the observation axis, `-1` |
| `_future`, `_obs` | "`future` or `1`" and "`obs` or `1`": the underscore marks an axis a correlated family collapses to one (jaxtyping treats a leading underscore as an anonymous, unbound dimension, so the annotation documents the collapse without asserting equality to the input) |
| `cells` | flattened `(window, horizon, series)` rows of a stacking matrix |
| `models` | the model axis, always last on stacking inputs and weights |
| `features` | continuous stacking features; `2*features` after `rectified_features` |
| `_cells_or_units` | one entry per cell, or per unit when `unit` is given to `log_score` |

### 2.2 Runtime enforcement

The package is imported under `jaxtyping.install_import_hook("numpyro_forecast", "beartype.beartype")`, so every annotation on a public or private function is checked at call time. Three consequences the scaffold relies on, each pinned by a passing test today:

- A rank mismatch is a `jaxtyping.TypeCheckError` before any body runs: `stacking_weights(np.zeros(10))` fails on `" cells models"` (`test_stacking_weights_reject_a_vector_at_the_type_boundary`). Tests of shape **semantics** (negative weights, mismatched windows) go through `StackingShapeError` from the body; tests of shape **rank** go through `TypeCheckError` from the hook.
- `Literal` arguments are enforced: `reduce_log_density(x, "mean")` is a `TypeCheckError` (`test_reduce_rejects_unknown_reduction_at_the_type_boundary`), so no kernel needs a "unknown reduction" branch and no docstring lists a `ValueError` for it.
- `Array | np.ndarray` return annotations are checked too, which is how the review found the pre-existing `forecast()` violation on the MVN layout.

Public inputs that may be NumPy or host-committed are annotated `Float[ArrayLike, ...]` or `Num[ArrayLike, ...]`, as `crps_empirical` does; jitted cores and block internals use `Float[Array, ...]`. Sampled values are narrowed with `jnp.asarray(numpyro.sample(...))` inside the blocks, per `AGENTS.md`.

### 2.3 Conventions carried over

- `rng_key` first and positional on everything that consumes randomness: `forecast_log_density`, `pseudo_bma_weights` (where it may be `None` for the no-bootstrap path, still first), `stack_forecasts`. `conditional_log_density` consumes none and takes none. The blocks take none: randomness comes from the enclosing `seed` handler, as for `innovations()`.
- `StackingCells` is a frozen dataclass of host (NumPy) arrays, like `BacktestResult`: it is a result container, not a pytree, and the solvers take its arrays, not the object. `ForecastOutput` is a `NamedTuple` because it is returned from inside closures that `backtest_vectorized` may vmap one day.
- `num_groups` on `hierarchical_weights` is a Python `int` (or a sequence of them) because it sizes a `numpyro.plate`; a traced `groups.max() + 1` cannot.
- Static arguments of the jitted kernels: `_log_density` is `jax.jit(static_argnums=(1, 5), static_argnames=("parallel",))` so `model` and `reader` are static; a `functools.partial(predict_density, site=...)` is hashable by identity, so a user should build the reader once and reuse it, or every call recompiles. The docstring of `predict_density` says so.
- No hard line breaks in prose, no em-dashes, `$...$` math, `\text{Normal}` not `\mathcal{N}`, raw strings for any docstring with a backslash: `tests/test_docstring_markup.py` passes on the branch.

## 3. Implementation notes, kernel by kernel

What follows is the algorithm each stub has to implement, with the JAX idiom to use and the test that pins it. Signatures are as on the branch; only the bodies are missing.

### 3.1 `predictive._site_log_prob(fn, truth) -> Array`

Branch on `fn.event_shape` first.

- `()`: return `fn.log_prob(truth)`, shape `(*batch, future, obs)`.
- `(future,)` with `truth.shape[-1] == 1` (the time-leading MVN layout the surgery module supports): `fn.log_prob(truth[..., 0])` has shape `(*batch,)`; return it reshaped to `(*batch, 1, 1)`.
- `(obs,)` matching `truth.shape[-1]` (an `ssoe()` noise distribution with an event over the observation axis, the VAR case): `fn.log_prob(truth)` has shape `(*batch, future)`; return `[..., None]`.
- Anything else: `ValueError` naming `fn.event_shape` and the three supported layouts.

Pinned by `test_predict_density_known_normal_matches_closed_form`, `test_mvn_obs_returns_the_joint_block_density` (which also checks the value against the conditional MVN computed by hand) and `test_unsupported_event_shape_raises`.

### 3.2 `predictive.predict_density(model_trace, truth, *, site="obs_future")`

The missing-site branch is implemented. The rest is `return _site_log_prob(model_trace[site]["fn"], truth)`. A discrete family with a float `truth` must raise the `predict()` message ("integer-dtyped"): check `_has_discrete_support(fn)` from `models.py` against `truth.dtype` here, so the error is raised once per call rather than per draw (`test_discrete_family_requires_integer_truth`). `test_predict_density_reads_a_custom_observed_site` pins the `site=` path, `test_poisson_density_matches_closed_form` the discrete path.

### 3.3 `predictive.ssoe_density(name) -> DensityReader`

Return a closure that reads `noise = model_trace[f"{name}_future"]["fn"]` and `mu = model_trace[f"{name}_mu_future"]["value"]`, raising `DensitySiteError` naming the missing site and the one-line deterministic when either is absent, and returns `_site_log_prob(noise, truth - mu)`. `noise` is the distribution expanded under the `time_future` plate, so its batch shape already carries `future`. Pinned by `test_ssoe_density_matches_ar1_closed_form` (the AR(1) closed form at $S = 20{,}000$, tolerance 0.03) and `test_ssoe_density_requires_mu_future_deterministic`.

### 3.4 `predictive._log_density(rng_key, model, posterior, data, covariates, reader, truth, *, parallel)`

```python
num_samples = _sample_axis_size(posterior)
keys = random.split(rng_key, num_samples)


def one(key, draw):
    conditioned = condition(substitute(model, draw), {_FUTURE_OBS_SITE: truth})
    model_trace = trace(seed(conditioned, key)).get_trace(covariates, data)
    return reader(model_trace, truth)


if parallel:
    return jax.vmap(one)(keys, dict(posterior))
return lax.map(lambda kd: one(*kd), (keys, dict(posterior)))
```

Conditioning `obs_future` is harmless for a reader that does not use it (`ssoe_density`, a custom `site`): the site is simply absent. `substitute` also pins deterministic sites present in the posterior (`drift`), as `Predictive` does. Pinned by `test_condition_substitute_seed_pattern_pins_only_in_sample_sites` (passes today, at trace level), `test_future_latents_are_drawn_from_the_prior_and_in_sample_sites_are_pinned`, `test_time_reparam_model_matches_the_plain_model_marginal`, `test_batch_dims_follow_the_series_plate`, and `test_parallel_false_matches_parallel_true_without_future_randomness`.

### 3.5 `predictive.forecast_log_density(...)`

In order: `_require_covariates_extend_data(data, covariates)`; check `truth.shape == (*data.shape[:-2], covariates.shape[-2] - data.shape[-2], data.shape[-1])` and raise `ValueError` mentioning `truth` otherwise; reject posterior keys with `"_future" in key or key in {"obs_future", "forecast"}` with a `ValueError` listing them (`test_rejects_posterior_carrying_future_sites`); then

```python
def predict_fn(key, post):
    return _log_density(key, model, post, data, covariates, reader, truth, parallel=parallel)


return _chunked_draws(rng_key, predict_fn, posterior, batch_size, _resolve_device(device))
```

`_chunked_draws` gives chunking, the compile-once index blocks, the OOM advice and the device placement for free. Pinned by `test_chunked_matches_unchunked_without_future_randomness` (equality for a model without future randomness, and at most one compilation across the chunk loop via the `count_compilations` fixture) and `test_host_placement_and_numpy_posterior` (`assert_host_resident` and `assert_numpy_host`, NumPy posterior leaves).

### 3.6 `predictive.reduce_log_density(log_density, reduction)` and `log_density_ess`

```python
ld = _device_view(log_density)
if reduction == "joint":
    ld = ld.sum(axis=-2, keepdims=True)
return logsumexp(ld, axis=0) - jnp.log(ld.shape[0])
```

ESS: with `log_w = ld - logsumexp(ld, axis=0)`, return `jnp.exp(-logsumexp(2 * log_w, axis=0))`. Pinned by `test_reduce_marginal_is_log_mean_exp_per_cell`, `test_reduce_joint_sums_over_time_before_averaging`, `test_reductions_coincide_at_horizon_one`, `test_reduce_is_stable_for_tiny_densities` (a shift of $-800$), `test_reduce_accepts_numpy_input`, `test_log_density_ess_bounds_and_extremes` (equal densities give $S$, one dominant draw gives 1) and `test_marginal_equals_logsumexp_identity_on_the_driver_output`.

### 3.7 `predictive.conditional_log_density(model, posterior, data, covariates, truth, *, batch_size, device)`

`t = data.shape[-2]`; require `covariates.shape[-2] >= t + truth.shape[-2]`; reject any posterior leaf whose shape has `t` at axis `-2` with a `ValueError` naming the site (a time-indexed latent that cannot be substituted into a longer window; `test_conditional_log_density_rejects_time_indexed_latents` matches `drift_decentered`); then `full = jnp.concatenate([data, truth], axis=-2)` and, per chunk through `_chunked_draws` with a dummy key,

```python
log_likelihood(model, dict(post), covariates[..., : full.shape[-2], :], full)["obs"][..., t:, :]
```

`numpyro.infer.util.log_likelihood` needs every sample site substituted, which the leaf check guarantees for `ssoe()` models. Pinned by `test_conditional_log_density_equals_full_window_obs_slice` and `test_conditional_marginal_at_horizon_one_matches_forecast_density` (the chain-rule and simulated-predecessor scores agree at one step, tolerance 0.02 at $S = 4{,}000$).

### 3.8 `evaluate._vectorized_log_density(rng_key, model, posterior, train_data, horizon_covariates, truth, *, reader, reduction)`

```python
keys = jax.vmap(lambda i: random.fold_in(random.fold_in(rng_key, i), 3))(jnp.arange(n))


def one(key, post_w, d, hc, y):
    return reduce_log_density(_log_density(key, model, post_w, d, hc, reader, y), reduction)


return jax.jit(jax.vmap(one))(keys, posterior, train_data, horizon_covariates, truth)
```

`fold_in(parent, 3)` is disjoint from the three streams `_window_key_streams` splits off the same parent. Pinned by `test_backtest_vectorized_log_density_shape_and_label`.

### 3.9 `stacking.stacking_cells(results, data)`

Validate: at least two models (`"two"` in the message); every result has `log_density` (`"log_density"`); every list has the same `(t0, t1, t2)` sequence (`"window"`); every result has the same `log_density_reduction` (`"reduction"`); predictions all present or all absent, with equal sample counts (`"prediction"`). Then for window $w$ with `(t1, t2)` and truth `data[..., t1:t2, :]`, flatten `log_density` of shape `(*batch, future_or_1, obs)` to rows in `(future, series)` order where `series` is the flattened `(*batch, obs)` index; `horizon` is the step index, or `-1` for the joint reduction; `window` is $w$. Stack models along the last axis; `draws` becomes `(models, sample, cells)` from `prediction` with the same flattening. Pinned by `test_stacking_cells_layout_is_window_major_then_horizon`, `test_stacking_cells_joint_reduction_has_one_cell_per_window`, `test_stacking_cells_collects_draws_when_every_model_kept_them` and the five rejection tests, which build `BacktestResult` objects by hand.

### 3.10 `stacking.window_design_effect(log_density, window)`

Sort models by their total log density; `d = L[:, best] - L[:, second]`; per-window means by `jax.ops.segment_sum` over `window` divided by counts; `deff = var(window means, ddof=1) / (var(d, ddof=1) / mean cells per window)`, clipped below at one. Pinned by `test_window_design_effect_is_one_for_independent_cells` (iid noise, 0.8 to 1.3) and `test_window_design_effect_grows_with_a_shared_window_term` (above 3).

### 3.11 `stacking.log_score(log_density, weights, *, unit)`

`scores = logsumexp(L + log(weights), axis=-1)` with `weights` broadcast from `(K,)` or `(cells, K)`; with `unit`, return `segment_sum(scores, unit) / segment_sum(1, unit)` over the sorted unique units (`jnp.unique` is not jittable with a dynamic size; use `num_segments = int(unit.max()) + 1` on the host side, since `unit` is a host array). Pinned by `test_log_score_sum_is_the_stacking_objective`, `test_log_score_per_unit_means_and_equal_weights_baseline`, `test_log_score_accepts_per_cell_weights`, and the end-to-end `test_held_out_comparison_carries_a_paired_standard_error_across_windows`.

### 3.12 `stacking.stacking_weights(log_density, *, cell_weights, dirichlet_alpha, tol, max_iter)`

Require `K >= 2` (`StackingShapeError`). With `eta` of shape `(K - 1,)`, `log_w = log_softmax(concatenate([eta, 0]))`, and

```python
def negative_objective(eta):
    log_w = jax.nn.log_softmax(jnp.concatenate([eta, jnp.zeros(1)]))
    per_cell = logsumexp(L + log_w, axis=-1)
    prior = (dirichlet_alpha - 1.0) * log_w.sum()
    return -(jnp.sum(pi * per_cell) + prior)


result = jax.scipy.optimize.minimize(
    negative_objective, jnp.zeros(K - 1), method="BFGS", tol=tol, options={"maxiter": max_iter}
)
return jax.nn.softmax(jnp.concatenate([result.x, jnp.zeros(1)]))
```

`jax.scipy.optimize.minimize` is jittable and differentiable, so the whole function is one `jax.jit` and vmaps over a leading axis (`test_stacking_weights_vmaps_over_a_leading_horizon_axis`). Pinned further by `test_stacking_weights_match_scipy_reference` (the ArviZ SLSQP algorithm inline, tolerance $2 \times 10^{-3}$), `test_stacking_weights_match_arviz_compare` (skipped without `arviz_stats`), `test_duplicated_models_combined_weight_is_invariant`, `test_dirichlet_alpha_above_one_splits_duplicates_evenly`, `test_stacking_weights_dominant_model_takes_all` (above 0.99, not exactly one), `test_stacking_weights_invariant_to_a_common_shift`, `test_zero_cell_weights_remove_cells`, and `test_log_score_is_concave_in_the_weights`.

### 3.13 `stacking.pseudo_bma_weights(rng_key, log_density, *, unit, num_bootstrap)`

Per-unit totals `T` of shape `(units, K)` by `segment_sum` (every cell its own unit when `unit is None`). Without bootstrap: `softmax(T.sum(0))`. With bootstrap: `alpha = random.dirichlet(rng_key, jnp.ones(units), (num_bootstrap,))`, `z = alpha @ T * units`, `softmax(z, axis=-1).mean(0)`, the ArviZ `bb-pseudo-bma` recipe with units in place of points. Pinned by `test_pseudo_bma_without_bootstrap_is_softmax_of_totals` and `test_pseudo_bma_bootstrap_is_on_the_simplex_and_less_extreme`.

### 3.14 `stacking.mixture_crps_terms(draws, truth)`

`a[k] = mean_s |X_k - y|`. For `B[k, l]` with $k \ne l$, the pair identity in $O(S \log S)$: sort `x = draws[k]` along the sample axis, `c = cumsum(x)`; for each `y_j` in `draws[l]`, `r = searchsorted(x, y_j)`; `sum_i |x_i - y_j| = (r y_j - c[r]) + ((c[-1] - c[r]) - (S - r) y_j)`; average over `j`. Vectorize with `jax.vmap` over cells and `jnp.searchsorted` along the sample axis; the diagonal is zero by construction and the matrix is symmetrized. Pinned by `test_mixture_crps_diagonal_is_the_single_model_crps` ($a_k - \tfrac12 B_{kk}$ equals `crps_empirical`; note $B_{kk}$ is the within-model term, so compute it with the same routine on `(x, x)`), `test_mixture_crps_terms_are_symmetric_with_zero_diagonal` and `test_mixture_crps_identity_against_a_proportional_pool` (three weight vectors, tolerance $2 \times 10^{-3}$ at 1,000 draws).

### 3.15 `stacking.crps_stacking_weights(a, b, *, cell_weights, tol, max_iter)`

Mean terms `A = sum_c pi_c a_c`, `Bm = sum_c pi_c B_c`; objective `f(w) = A @ w - 0.5 w @ Bm @ w`, gradient `A - Bm @ w`; projected gradient with step `1 / L`, `L = max |eigenvalue(Bm)|`, and the Euclidean projection onto the simplex (sort, cumulative sum, threshold), run with `lax.while_loop` until the step is below `tol` or `max_iter`. Exact zeros on the boundary are expected. Pinned by `test_crps_stacking_weights_two_models_match_a_grid_search`, `test_crps_stacking_objective_is_convex_on_the_simplex` (midpoint convexity on random simplex points) and `test_crps_stacking_weights_select_a_perfect_model`.

### 3.16 `stacking.rectified_features(x, *, center)`

`center = median(x, axis=0)` unless given; return `concatenate([relu(x - center), relu(center - x)], axis=-1), center`. Pinned by `test_rectified_features_identities` and `test_rectified_features_reuse_the_training_center`.

### 3.17 `stacking.hierarchical_weights(name, num_models, groups, num_groups, *, features, group_prior, tau_mu, tau_sigma, tau_beta)`

Normalize `groups`, `num_groups` and `group_prior` to tuples of equal length (`StackingShapeError` otherwise; also for `num_models < 2`). Sites, with `K1 = num_models - 1`:

```python
with numpyro.plate(f"{name}_models", K1):
    mu = jnp.asarray(numpyro.sample(f"{name}_mu", dist.Normal(0.0, tau_mu)))
alpha = jnp.broadcast_to(mu, (cells, K1))
for i, (g, J, prior) in enumerate(zip(groups, num_groups, group_prior)):
    with numpyro.plate(f"{name}_models_{i}", K1):
        sigma = jnp.asarray(numpyro.sample(f"{name}_sigma_{i}", dist.HalfNormal(tau_sigma)))
        if prior == "exchangeable":
            with numpyro.plate(f"{name}_groups_{i}", J, dim=-2):
                z = jnp.asarray(numpyro.sample(f"{name}_z_{i}", dist.Normal(0.0, 1.0)))
        else:
            with numpyro.plate(f"{name}_groups_{i}", J - 1, dim=-2):
                eps = jnp.asarray(numpyro.sample(f"{name}_z_{i}", dist.Normal(0.0, 1.0)))
            z = jnp.concatenate([jnp.zeros((1, K1)), jnp.cumsum(eps, axis=0)], axis=0)
            z = z - z.mean(axis=0, keepdims=True)
    offset = jnp.where((g >= 0)[:, None], sigma * z[jnp.clip(g, 0)], 0.0)
    alpha = alpha + offset
if features is not None:
    with numpyro.plate(f"{name}_models_beta", K1), numpyro.plate(f"{name}_features", M, dim=-2):
        beta = jnp.asarray(numpyro.sample(f"{name}_beta", dist.Normal(0.0, tau_beta)))
    alpha = alpha + features @ beta
log_w = jax.nn.log_softmax(jnp.concatenate([alpha, jnp.zeros((cells, 1))], axis=-1), axis=-1)
numpyro.deterministic(name, jnp.exp(log_w))
return log_w
```

The `jnp.where` on `g >= 0` with a clipped gather is what makes `-1` the population group without a branch. Shapes, as the test pins them: `{name}_mu` `(K1,)`, `{name}_sigma_i` `(K1,)`, `{name}_z_i` `(J, K1)` for the exchangeable prior and `(J - 1, K1)` for the random walk, `{name}_beta` `(M, K1)`, deterministic `name` `(cells, K)`. Pinned by `test_hierarchical_weights_sites_and_shapes`, `test_hierarchical_weights_zero_spread_is_complete_pooling` (condition `sigma` to zero), `test_hierarchical_weights_unseen_group_gets_population_weights` (condition `mu`), `test_hierarchical_weights_random_walk_offsets_are_centered`, `test_hierarchical_weights_several_group_indices_are_additive`, `test_hierarchical_weights_reject_a_single_model`, and the slow `test_stacking_model_recovers_group_varying_weights_with_nuts`.

### 3.18 `stacking.stacking_factor(name, log_density, log_weights, *, cell_weights)`

`value = jnp.sum(pi * logsumexp(log_density + log_weights, axis=-1))`; `numpyro.factor(name, value)`; return `value`. Pinned by `test_stacking_factor_registers_the_objective`, which reads the factor site's `fn.log_prob(value)` from a trace.

### 3.19 `stacking.stack_forecasts(rng_key, draws, weights, *, per)`

Validate: non-negative, rows sum to one within $10^{-5}$, broadcastable to `(*batch, future, obs, K)` (all `StackingShapeError`); resolve `per="auto"` to `"path"` for a `(K,)` vector and `"cell"` otherwise, and reject `per="path"` with cell-varying weights. Then with `log_w = log(weights)`: for `"path"`, `idx = random.categorical(rng_key, log_w, shape=(S,))` and `jnp.take_along_axis` over the model axis after moving it; for `"cell"`, `idx = random.categorical(rng_key, log_w, shape=(S, *batch, future, obs))` broadcast and gathered per cell. Pinned by `test_stack_forecasts_one_hot_weights_select_that_model`, `test_stack_forecasts_path_mode_keeps_whole_paths`, `test_stack_forecasts_cell_mode_matches_the_mixture_mean` (statistical, 20,000 draws, tolerance 0.1), `test_stack_forecasts_auto_resolution` and `test_stack_forecasts_reject_bad_weights`.

## 4. The test suite

### 4.1 Layout and markers

Two modules, both at `tests/`, both importing the shared helpers from `conftest.py` (`empty_covariates`, `svi_forecast_fn`, `rw_model`, `assert_host_resident`, `assert_numpy_host`, the `count_compilations` and `rng_key` fixtures). `tests/test_density.py` covers everything in `predictive.py` and the `evaluate.py` hook; `tests/test_stacking.py` covers `stacking.py`. Both define

```python
scaffold = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="stacking scaffold: kernel not implemented"
)
```

and apply it per test, never at module level, because the tests of implemented validation (`_unpack_forecast_output`, the missing-site branch of `predict_density`, the beartype boundary checks, the trace-level mechanism check) pass today and must stay unmarked. One test is `@pytest.mark.slow` (the NUTS recovery), which the suite's `slow` marker already declares. No test reads a fitted number from a snapshot; every numeric assertion is against a closed form, an independent reference implementation, a NumPy re-derivation, or a bound.

### 4.2 Models with known parameters

`tests/test_density.py` defines six small models built from the package's blocks, each chosen so that one property is closed form:

| Model | Why it exists |
| --- | --- |
| `mean_model` | constant Normal mean, no future latents: every reduction is exact, chunking and `lax.map` must reproduce `vmap` bit for bit, the Normal closed form holds per draw |
| `rw_reparam_model` | random walk with `LocScaleReparam(0)`: the `drift_future_decentered` auxiliary must be drawn by the seed; with the in-sample drift substituted the h-step marginal is $\text{Normal}(\text{level}_t, \sqrt{h\,\text{drift\_scale}^2 + \sigma^2})$ |
| `poisson_model` | discrete support through a link: the integer-dtype error and the Poisson closed form |
| `mvn_model` | time-correlated `MultivariateNormal` in the layout the surgery module supports: the joint block case, checked against the conditional MVN computed with `numpy.linalg` |
| `series_model` | a series plate: batch dimensions are carried through as `(S, 3, future, 1)` |
| `ar1_ssoe_model` | an AR(1) in error-feedback form with the `ar_mu_future` deterministic: the `ssoe_density` reader against the AR(1) closed form, the chain-rule route against `log_likelihood` on the concatenated series, and the agreement of the two routes at one step |

`tiled(num_samples, **values)` builds a posterior dict of fixed values with a leading sample axis, so the driver is tested with known parameters and the only randomness is the future path.

### 4.3 Test matrix: densities

| Test | Property | Kind |
| --- | --- | --- |
| `test_predict_density_known_normal_matches_closed_form` | per-draw value equals `Normal(bias, sigma).log_prob(truth)`, shape `(S, future, 1)` | identity |
| `test_predict_density_missing_site_raises` | `DensitySiteError` naming `ssoe_density` | contract, passes today |
| `test_predict_density_reads_a_custom_observed_site` | `partial(predict_density, site=)` reads a user predictive | identity |
| `test_mvn_obs_returns_the_joint_block_density` | shape `(S, 1, 1)`, value equals the conditional MVN | identity |
| `test_unsupported_event_shape_raises` | event `(future, 2)` is refused | contract |
| `test_ssoe_density_matches_ar1_closed_form` | AR(1) h-step marginal at $S = 20{,}000$ | statistical, tolerance 0.03 |
| `test_ssoe_density_requires_mu_future_deterministic` | `DensitySiteError` naming `ar_mu_future` | contract |
| `test_random_walk_marginal_matches_analytic_h_step_density` | $\text{Normal}(\text{level}_t, \sqrt{h\,\text{ds}^2 + \sigma^2})$ at $S = 8{,}000$ | statistical, tolerance 0.03 |
| `test_future_latents_are_drawn_from_the_prior_and_in_sample_sites_are_pinned` | key changes the result iff the future path is random | mechanism |
| `test_time_reparam_model_matches_the_plain_model_marginal` | `time_reparam("haar")` leaves the marginal unchanged | statistical |
| `test_rejects_posterior_carrying_future_sites` | `ValueError` listing `drift_future_decentered` | contract |
| `test_rejects_truth_shape_mismatch` | `ValueError` on a wrong `truth` shape | contract |
| `test_discrete_family_requires_integer_truth` | the `predict()` message on float counts | contract |
| `test_poisson_density_matches_closed_form` | `Poisson(rate).log_prob` | identity |
| `test_batch_dims_follow_the_series_plate` | `(S, 3, future, 1)` and per-series closed form | identity |
| `test_chunked_matches_unchunked_without_future_randomness` | equality and at most one compilation in the chunk loop | identity, performance |
| `test_parallel_false_matches_parallel_true_without_future_randomness` | `lax.map` equals `vmap` | identity |
| `test_host_placement_and_numpy_posterior` (x2) | `device="host"` and `"numpy"` placement, NumPy posterior leaves | contract |
| `test_reduce_marginal_is_log_mean_exp_per_cell` | against `np.log(np.mean(np.exp(...)))` | identity |
| `test_reduce_joint_sums_over_time_before_averaging` | against the NumPy block definition, shape `(2, 1, 1)` | identity |
| `test_reductions_coincide_at_horizon_one` | equal at `future == 1` | identity |
| `test_reduce_is_stable_for_tiny_densities` | shift by $-800$ is exact | numerics |
| `test_reduce_rejects_unknown_reduction_at_the_type_boundary` | `TypeCheckError` on a misspelled `Literal` | typing, passes today |
| `test_log_density_ess_bounds_and_extremes` | $S$ for equal densities, 1 for one dominant draw, within $[1, S]$ otherwise | identity |
| `test_reduce_accepts_numpy_input` | NumPy input accepted | contract |
| `test_conditional_log_density_equals_full_window_obs_slice` | equals `log_likelihood(...)["obs"][:, t:]` | identity |
| `test_conditional_marginal_at_horizon_one_matches_forecast_density` | the two routes agree at one step | statistical, tolerance 0.02 |
| `test_conditional_log_density_rejects_time_indexed_latents` | `ValueError` naming `drift_decentered` | contract |
| `test_backtest_log_density_requires_forecast_output` | `ValueError` naming `ForecastOutput` | contract, passes today |
| `test_backtest_forecast_output_without_log_density_matches_bare_draws` | metrics identical, density fields `None` | contract, passes today |
| `test_backtest_stores_reduced_log_density_with_its_label` (x2) | shapes `(future, 1)` and `(1, 1)`, label recorded | contract |
| `test_backtest_log_density_changes_no_other_field` | metrics and predictions unchanged; `transform` rescales metrics but not the density | contract |
| `test_backtest_vectorized_log_density_shape_and_label` | `(num_windows, test_window, 1)` and label | contract |
| `test_backtest_vectorized_log_density_default_is_absent` | `None` by default | contract, passes today |
| `test_marginal_equals_logsumexp_identity_on_the_driver_output` | the reduction formula on real driver output | identity |
| `test_condition_substitute_seed_pattern_pins_only_in_sample_sites` | the mechanism at trace level | mechanism, passes today |

### 4.4 Test matrix: stacking

| Test | Property | Kind |
| --- | --- | --- |
| `test_stacking_weights_on_the_simplex` | non-negative, sum one | contract |
| `test_stacking_weights_dominant_model_takes_all` | above 0.99 | contract |
| `test_stacking_weights_match_scipy_reference` | the ArviZ SLSQP algorithm inline, tolerance $2 \times 10^{-3}$, objective not worse | reference |
| `test_stacking_weights_match_arviz_compare` | `arviz_stats.compare(method="stacking")` on hand-built `ELPDData` | reference, skipped without `arviz_stats` |
| `test_duplicated_models_combined_weight_is_invariant` | duplicates' sum equals the merged weight | identity (the review's correction of "split evenly") |
| `test_dirichlet_alpha_above_one_splits_duplicates_evenly` | the even split under $\alpha = 1.5$ | identity |
| `test_log_score_is_concave_in_the_weights` | Jensen along ten random segments | property |
| `test_stacking_weights_invariant_to_a_common_shift` | shift by $-700$ | numerics |
| `test_zero_cell_weights_remove_cells` | zero `cell_weights` equals dropping rows | identity |
| `test_stacking_weights_vmaps_over_a_leading_horizon_axis` | `vmap` equals the loop | contract |
| `test_stacking_weights_reject_a_single_model` | `StackingShapeError` | contract |
| `test_stacking_weights_reject_a_vector_at_the_type_boundary` | `TypeCheckError` | typing, passes today |
| `test_log_score_sum_is_the_stacking_objective` | sum of per-cell scores | identity |
| `test_log_score_per_unit_means_and_equal_weights_baseline` | per-unit means equal the reshaped cell means | identity |
| `test_log_score_accepts_per_cell_weights` | `(cells, K)` weights | identity |
| `test_pseudo_bma_without_bootstrap_is_softmax_of_totals` | plain pseudo-BMA | identity |
| `test_pseudo_bma_bootstrap_is_on_the_simplex_and_less_extreme` | bootstrap shrinks toward uniform | property |
| `test_window_design_effect_is_one_for_independent_cells` | 0.8 to 1.3 on iid cells | statistical |
| `test_window_design_effect_grows_with_a_shared_window_term` | above 3 with a shared term | statistical |
| `test_mixture_crps_diagonal_is_the_single_model_crps` | $a_k - \tfrac12 B_{kk}$ equals `crps_empirical` | identity |
| `test_mixture_crps_terms_are_symmetric_with_zero_diagonal` | $B$ symmetric, zero diagonal | identity |
| `test_mixture_crps_identity_against_a_proportional_pool` (x3) | the quadratic form equals `crps_empirical` on a proportionally pooled sample | identity (the review's verification) |
| `test_crps_stacking_weights_two_models_match_a_grid_search` | against a 1,001-point grid | reference |
| `test_crps_stacking_objective_is_convex_on_the_simplex` | midpoint convexity on random simplex points | property |
| `test_crps_stacking_weights_select_a_perfect_model` | above 0.98 | contract |
| `test_rectified_features_identities` | right minus left is $x - m$, right plus left is $|x - m|$ | identity |
| `test_rectified_features_reuse_the_training_center` | held-out cells use the training medians | contract |
| `test_hierarchical_weights_sites_and_shapes` | site names, shapes, rows sum to one, deterministic equals `exp(log_w)` | contract |
| `test_hierarchical_weights_zero_spread_is_complete_pooling` | all rows equal under `sigma = 0` | limit |
| `test_hierarchical_weights_unseen_group_gets_population_weights` | group $-1$ gives `softmax([mu, 0])` | limit |
| `test_hierarchical_weights_random_walk_offsets_are_centered` | the offsets average to zero | identity |
| `test_hierarchical_weights_several_group_indices_are_additive` | two group indices, two priors, suffixed sites | contract |
| `test_hierarchical_weights_reject_a_single_model` | `StackingShapeError` | contract |
| `test_stacking_factor_registers_the_objective` | the factor's `log_prob` equals the weighted objective | identity |
| `test_stacking_model_recovers_group_varying_weights_with_nuts` | posterior-mean weights within 0.1 of a three-group truth, unseen group near the population | statistical, slow |
| `test_stack_forecasts_one_hot_weights_select_that_model` | equality with that model's draws | identity |
| `test_stack_forecasts_path_mode_keeps_whole_paths` | every path equals one model's path; both models used | property |
| `test_stack_forecasts_cell_mode_matches_the_mixture_mean` | mean equals $\sum_k w_k \mu_k$ per cell at 20,000 draws | statistical, tolerance 0.1 |
| `test_stack_forecasts_auto_resolution` | `"auto"` is `"path"` for a vector and `"cell"` otherwise | contract |
| `test_stack_forecasts_reject_bad_weights` (x3) | negative, unnormalized, `"path"` with cell weights | contract |
| `test_stacking_cells_layout_is_window_major_then_horizon` | the flattening order, truth, names, reduction | contract |
| `test_stacking_cells_joint_reduction_has_one_cell_per_window` | horizon $-1$, one row per window | contract |
| `test_stacking_cells_collects_draws_when_every_model_kept_them` | `(models, sample, cells)` | contract |
| `test_stacking_cells_reject_mismatched_windows` and four siblings | `StackingShapeError` with the named reason | contract |
| `test_held_out_comparison_carries_a_paired_standard_error_across_windows` | the report format of the notebook: per-window gain over equal weights and its standard error | end to end |

### 4.5 What the tests deliberately do not do

- No fitted-weight snapshots: a changed optimizer tolerance must not break a test, a wrong objective must.
- No assertion that stacking beats a single model on simulated data: Appendix A of the design shows why such a test would be noise.
- No test of the `ssoe_density` reader on the intermittent-demand examples: those models have no scorable forecast (design Section 7.1).
- The vectorized backtest's density is tested for shape and label only; its agreement with the loop is statistical and already covered for the metrics by `test_metrics_statistically_close_to_loop`, and the density path shares the kernels pinned elsewhere.

## 5. Runtime budget

The two modules run in about 40 seconds on four CPU workers with every kernel stubbed (most of that is the SVI fits of the backtest tests and the compile of the AR(1) model). With kernels implemented, the statistical tests add Monte Carlo work: `test_ssoe_density_matches_ar1_closed_form` ($S = 20{,}000$, $H = 4$) and `test_random_walk_marginal_matches_analytic_h_step_density` ($S = 8{,}000$) are each a single vmapped pass and should stay under five seconds; `test_stack_forecasts_cell_mode_matches_the_mixture_mean` draws 20,000 categorical indices, negligible; the NUTS recovery test is the one to mark `slow` (it is) and to add to `.test_durations` with `make store-durations` when it lands.

## 6. From scaffold to PR 1

1. Implement Sections 3.1 to 3.8 in the order listed; after each, remove the `@scaffold` mark from the tests named in that section and make them pass.
2. Add the `_future`-key guard to `forecast()` and `predict_in_sample()` as well, with a test in `tests/test_predictive.py`, and file the MVN `forecast()` shape issue (design Section 10).
3. Update the `forecasting_univariate` notebook: a `ForecastOutput`-returning closure, `backtest(..., log_density="marginal")`, and a `log_score` column in the backtest table next to CRPS.
4. Update `AGENTS.md` (Section 7) and the `skill:` block.
5. Leave `stacking.py` on the branch as is for PR 2 and PR 3; its tests stay xfail until then. If the maintainer prefers not to ship stubs, move `stacking.py` and `tests/test_stacking.py` to a follow-up branch and drop the corresponding `__init__`, `great-docs.yml` and `test_package.py` entries; nothing in PR 1 imports them.

## 7. Conventions to record in `AGENTS.md`

- **Two h-step scores, never compared.** `reduction="marginal"` is the default and the one the sample metrics correspond to; `"joint"` is the chain-rule block score of `lfo_cv`; they coincide at horizon one only.
- **The window is the unit of exchangeability** for every stacking standard error, bootstrap and tempering; quote held-out gains per window with a paired standard error, and against equal weights.
- **`forecast_fn` may return `ForecastOutput`**; it must when `backtest(log_density=...)` is requested, and the closure never receives the test block.
- **A model is stackable** when it is built on `predict()`, or on `ssoe()` with `numpyro.deterministic(f"{name}_mu_future", r.mu_future)` registered when forecasting (additive single-channel models only), or when it registers an observed-style site holding the predictive to score and the user points `predict_density(site=...)` at it.
- **Model axis last** on every stacking matrix and on per-cell weights.
- **No shipped model functions**: hierarchical stacking is two blocks the user composes.
