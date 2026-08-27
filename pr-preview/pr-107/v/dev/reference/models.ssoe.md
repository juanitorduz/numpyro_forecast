## models.ssoe()


Run a single-source-of-error recursion over the full horizon.


Usage

``` python
models.ssoe(
    h,
    name,
    y,
    init_carry,
    step,
    noise_dist,
    xs=None,
)
```


The building block for innovations state-space models (ARMA, exponential smoothing, Croston/TSB levels, censored autoregressions): a deterministic filter whose state is driven by the one-step-ahead *error* `eps_t = y_t - mu_t`. In-sample it runs `step` in a raw `jax.lax.scan` over the observed series `y` (no sample sites inside); when forecasting it draws iid future errors at the site `f"{name}_future"` from `noise_dist` under `plate("time_future", h.future, dim=-2)` and runs a second scan from the final in-sample carry with `y_t = mu_t + eps_t` fed back through `carry_fn`. The guide never sees the future site, because fitting always happens with `future == 0`. Linear-Gaussian members (ARMA, additive exponential smoothing) can be marginalized exactly by a Kalman filter; the error-feedback form is the one that also covers the nonlinear members.

The block registers nothing but the error site. The caller writes the likelihood against `r.mu` and registers `numpyro.deterministic("forecast", r.y_future)` when `h.future > 0` (an unconditional registration is harmless to `~numpyro_forecast.predictive.forecast()`, but lands a size-0 variable in every posterior). Driver contract: `~numpyro_forecast.predictive.predict_in_sample()` and `~numpyro_forecast.convert.to_datatree()` call the model with `data=None` and read `"obs"`, so `y` must come from `covariates` or be computed in the model, never from `h.data`; `~numpyro_forecast.predictive.forecast()` reads `"forecast"`.

**Frozen gates.** Route an update gate (Croston's demand indicator, an availability mask) through an `xs` leaf frozen over the horizon with `~numpyro_forecast.arrays.pad_future()`, and read it from `x_t`, never from `y_t` (over the horizon `y_t = mu_t + eps_t` is nonzero). With the gate off, `carry_fn` is the identity and the forecast is the last level plus iid errors. `~numpyro_forecast.evaluate.backtest()` and `~numpyro_forecast.predictive.forecast()` hand the model *real* future covariate rows, so a gate sliced from the full covariates keeps updating on sampled values and leaks the test window; scenario inputs such as a future availability mask are the only thing to read from those rows.

**Shapes.** Rows are `(*batch, obs)`: a scalar state emits `mu[None]` and starts from `init[None]` (the block refuses a scalar or a wider mean because either would silently broadcast the likelihood into a `(t, t)` log-prob); a tuple carry with scalar leaves reads `eps_t[0]` (the ETS idiom). A panel puts the series on the observation axis: `y` is `(t_obs, series)`, the carry `(series,)`, and a `noise` sampled under `plate("series")` has exactly the batch shape `(series,)` the block needs. Batch dims to the left of time (`(B, t_obs, obs)`) take a `(B, obs)` carry and a `(B, 1, obs)` noise batch. Inputs are jax Arrays (the import hook rejects NumPy). With `obs == 1` a noise batch shape `(future, 1)` is indistinguishable from time and is consumed as such: per-step error scales, if that is what you meant.

**Composition.** Two channels are two calls sharing the same [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon); each opens its own `time_future` plate. Scoping a helper that contains the call is fine (`handlers.scope` prefixes the error site and the plate; use `name="eps"` inside a scoped helper so the site reads `z_eps_future`); register `"obs"` and `"forecast"` outside any scope and build the forecast from the channels' `y_future`.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon)).

`name: str`  
Base name of the error site; the future errors are drawn at `f"{name}_future"`.

`y: Array | None`  
The driving series over the observed window, shape `(*batch, t_obs, obs)` with time at axis `-2` (integer counts are fine as long as the carry, hence the mean, is floating; the error promotes). Sliced from `covariates` or computed in the model; `None` (the value of `h.data` under `data=None`) raises.

`init_carry: Carry`  
Initial carry, any PyTree, already broadcast to the `(*batch, obs)` rows: a scalar level is `init[None]`, a panel level `(series,)`. Every leaf must keep its shape and dtype through `carry_fn`.

`step: SSOEStep[Carry]`  
`(carry, x_t) -> (mu_t, carry_fn)` (see [SSOEStep](models.SSOEStep.md#numpyro_forecast.models.SSOEStep)): `mu_t` is the mean for the current row (shape `(*batch, obs)`, so a scalar state emits `mu[None]`) and `carry_fn(y_t, eps_t)` the next carry. `carry_fn` receives the drawn `eps_t` over the horizon (not a recomputed `y_t - mu_t`, which can differ by an ulp), so close over `mu_t` when the update needs it.

`noise_dist: dist.Distribution`  
Zero-centered per-step error distribution. Its batch shape must be `(obs,)` for a `(t_obs, obs)` series (`()` is fine when `obs == 1`) and `(B, 1, obs)` for a batched `(B, t_obs, obs)` series, so the draw under the time plate is exactly `(*batch, future, obs)` with the dtype of the means; event-shaped distributions are rejected.

`xs: PyTree[Array] | None = None`  
Optional exogenous inputs over the full horizon: a PyTree of arrays with time at axis `-2` and `duration` rows (a single array, a tuple, a dict, …), split at `h.t_obs` and handed to `step` row by row as `x_t`; `None` for autonomous dynamics.


## Returns


`SSOEResult`  
`mu` (in-sample means), `mu_future` and `y_future` (forecast means and sampled values; size-0 time axis while training).


## Raises


`ValueError`  
If `y` is `None`, lacks the time or observation axis, or does not cover exactly `h.t_obs` rows; if an `xs` leaf lacks the axes or does not span `h.duration` rows; if `step` returns a mean without the observation axis or a carry with a different tree structure, shape or dtype; if `step` calls `numpyro.sample`; or if `noise_dist` draws errors of the wrong shape or dtype.


## Examples

ARMA(1,1) with the lambda form of `carry_fn` (`y` is the observed series routed through `covariates`):

``` python
>>> def step(carry, _):
...     y_prev, eps_prev = carry
...     mu_t = mu + phi * y_prev + theta * eps_prev
...     return mu_t, lambda y_t, eps_t: (y_t, eps_t)
>>> r = ssoe(h, "eps", y, (mu[None], jnp.zeros((1,))), step, dist.Normal(0.0, sigma))
>>> numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
>>> if h.future > 0:
...     numpyro.deterministic("forecast", r.y_future)
```

A gated level (Croston, TSB) with the gate frozen over the horizon:

``` python
>>> def step(level, gate_t):
...     def carry_fn(y_t, _):
...         return jnp.where(gate_t, alpha * y_t + (1 - alpha) * level, level)
```

…

``` python
...     return level, carry_fn
>>> gate_full = pad_future(gate, h.future)
>>> r = ssoe(h, "eps", y, init[None], step, dist.Normal(0.0, noise), xs=gate_full)
```
