---
name: numpyro_forecast
description: >
  A JAX/NumPyro port of the ideas in Pyro's forecasting module. Use when writing Python code that uses the numpyro_forecast package.
license: Apache-2.0
compatibility: Requires Python >=3.12.
---

# numpyro_forecast

A JAX/NumPyro port of the ideas in Pyro's forecasting module.

## Installation

```bash
pip install numpyro_forecast
```

## When to use what

| Need | Use |
|------|-----|
| Derive the train/forecast split inside a model | `Horizon.from_data(covariates, data)` |
| Per-step iid latent innovations (random-walk level, local trend) | `innovations(h, name, prior); build the series with jnp.cumsum(drift, axis=-2)` |
| Latent whose per-step distribution depends on the previous state | `markov_series(h, name, init_carry, transition, advance=...)` |
| Single-source-of-error recursion (ARMA, exponential smoothing, Croston, TSB) | `ssoe(h, name, y, init_carry, mean, update, noise_dist)` |
| Register the likelihood over the training window and the forecast site | `predict(h, obs_dist, prediction)` |
| Vector autoregression | `var.var_step(...) as the mean/update pair of ssoe; priors.minnesota_prior for shrinkage` |
| Posterior draws from a fitted SVI guide | `draw_posterior(rng_key, guide, params, num_samples)` |
| Forecast over the horizon | `forecast(rng_key, model, posterior, data, covariates)` |
| In-sample posterior predictive | `predict_in_sample(rng_key, model, posterior, covariates)` |
| ArviZ DataTree with posterior, in-sample predictive, forecasts and observed data | `to_datatree(rng_key, model, posterior, data, covariates)` |
| Rolling or expanding window backtest with scores per window | `backtest(rng_key, data, covariates, model_fn, forecast_fn=...); backtest_vectorized for one vmapped SVI fit` |
| Better SVI or NUTS geometry for a random-walk level | `time_reparam(model, "dct") or time_reparam(model, "haar"), created once and reused` |
| BlackJAX samplers or Pathfinder | `contrib.blackjax.BlackjaxNUTSKernel / BlackjaxMCLMCKernel with MCMC; fit_multipathfinder + multipathfinder_samples` |

## API overview

### Model building blocks

Plain model functions that register the train/forecast sites for you.

- `models.Horizon`
- `models.Transition`
- `models.Advance`
- `models.innovations`
- `models.markov_series`
- `models.ssoe`
- `models.SSOEMean`
- `models.SSOEUpdate`
- `models.SSOEResult`
- `models.predict`
- `models.PlateName`

### Vector autoregression

VAR components that compose with `ssoe` and `markov_series`: conditional mean, step factory, companion matrix, impulse responses.

- `var.var_mean`
- `var.var_step`
- `var.companion_matrix`
- `var.impulse_response`

### Priors

Shrinkage prior moments for coefficient arrays.

- `priors.minnesota_prior`

### Distribution surgery

Time-axis operations on observation distributions, extensible via singledispatch.

- `surgery.shift_loc`
- `surgery.slice_time`
- `surgery.prefix_condition`
- `surgery.register_elementwise`

### Reparameterization

Time-axis reparameterization of in-sample latents (Haar / DCT), after Pyro's `time_reparam`.

- `reparam.time_reparam`

### Producing draws

Drawing posterior samples and generating forecasts and in-sample predictions.

- `predictive.draw_posterior`
- `predictive.forecast`
- `predictive.predict_in_sample`

### Backtesting & evaluation

Rolling-window backtesting and forecast metrics.

- `evaluate.backtest`
- `evaluate.backtest_vectorized`
- `evaluate.BacktestResult`
- `evaluate.VectorizedBacktestResult`
- `evaluate.evaluate_forecast`
- `evaluate.results_to_dataframe`
- `evaluate.eval_crps`
- `evaluate.eval_mae`
- `evaluate.eval_rmse`
- `evaluate.eval_coverage`
- `metrics.crps_empirical`
- `metrics.eval_pinball`
- `metrics.eval_interval_score`
- `metrics.make_mase`

### ArviZ export

Convert posteriors into ArviZ-schema xarray DataTrees for diagnostics and plotting.

- `convert.to_datatree`
- `convert.add_forecast_groups`
- `convert.predictions_to_datatree`

### Extensions (contrib)

Optional backends behind pyproject extras (never imported by default).

- `contrib.blackjax.BlackjaxNUTSKernel`
- `contrib.blackjax.BlackjaxMCLMCKernel`
- `contrib.blackjax.BlackjaxCustomKernel`
- `contrib.blackjax.PathfinderFit`
- `contrib.blackjax.fit_pathfinder`
- `contrib.blackjax.pathfinder_samples`
- `contrib.blackjax.MultiPathfinderFit`
- `contrib.blackjax.fit_multipathfinder`
- `contrib.blackjax.multipathfinder_samples`

### Typing

Public type contracts.

- `typing.ForecastModel`
- `typing.ForecastFn`
- `typing.Guide`
- `typing.InSampleFn`
- `typing.Metric`
- `typing.ModelFactory`

### Autocorrelation

Batched autocorrelation and partial autocorrelation diagnostics.

- `acf.acf`
- `acf.pacf`

### Seasonal features

Fourier design matrices and seasonal tiling.

- `features.fourier_features`
- `features.periodic_repeat`

### Array helpers

Time-axis array shaping for the train/forecast split.

- `arrays.zero_data_like`
- `arrays.concat_future`
- `arrays.pad_future`

### Datasets

Example datasets used in the tutorials.

- `datasets.load_bart_weekly`
- `datasets.load_bart_hierarchical`
- `datasets.load_victoria_electricity`
- `datasets.bart_available`

### Optional dependencies

Lazy imports behind pyproject extras.

- `optional.require`

### Exceptions

Package exception hierarchy raised at validation boundaries.

- `exceptions.NumpyroForecastError`
- `exceptions.BacktestWindowError`
- `exceptions.VectorizedMetricError`
- `exceptions.KernelConfigError`
- `exceptions.CovariateDimsError`
- `exceptions.MVNLayoutError`
- `exceptions.DeviceMemoryError`
- `exceptions.HostMemoryKindError`
- `exceptions.DevicePlatformError`

## Gotchas

1. Array layout is time at axis -2 and the observation dim at -1, batch dims to the left: a univariate series is shape (time, 1), not (time,).
2. There is no horizon argument. The forecast horizon is covariates.shape[-2] - data.shape[-2]: fit with covariates[:t_obs] and forecast with the full-horizon covariates, using the same model function.
3. Every function that consumes randomness takes rng_key as its first positional argument (draw_posterior, forecast, predict_in_sample, to_datatree, backtest).
4. The drivers read the sites "obs" and "forecast" by name. Never put predict (or hand-written obs/forecast sites) inside handlers.scope; scope only the latent building blocks.
5. innovations is called inside plates you open yourself; markov_series rejects an enclosing plate and takes plates=[(name, size)] instead.
6. ssoe registers only the <name>_future error site: write the likelihood against result.mu yourself and register numpyro.deterministic("forecast", result.y_future) when h.future > 0. Its mean and update functions must not call numpyro.sample.
7. innovations, ssoe and predict take distribution instances (dist.Normal(0.0, scale)), not distribution classes or thunks.
8. predict_in_sample and to_datatree call the model with data=None, so anything the model needs at prediction time must travel through covariates, never through h.data.
9. time_reparam returns a wrapped model: create it once and hand that same object to the guide, SVI/MCMC, forecast, predict_in_sample, to_datatree and backtest; wrapping inside a loop recompiles, and nesting two time_reparam calls raises ValueError.
10. draw_posterior is for variational guides only; MCMC users pass mcmc.get_samples() and Pathfinder users multipathfinder_samples(...) to the same drivers.
11. numpyro_forecast.contrib.blackjax is never imported by default and needs the blackjax extra (pip install "numpyro_forecast[blackjax]"); optax optimizers need the optax extra and numpyro.optim.optax_to_numpyro.

## Best practices

- Keep inference plain NumPyro: SVI with an autoguide or MCMC with any kernel; nothing in the package wraps svi.run or mcmc.run.
- Use LocScaleReparam(0) on innovations priors for SVI, and time_reparam(model, "dct") when the latent is a random-walk level.
- On a GPU, call numpyro.set_platform("cuda") before building the model and pass batch_size and device="host" to draw_posterior, forecast and predict_in_sample so the draws never sit on the accelerator at once; draws come back as jax Arrays on the CPU device or NumPy arrays.
- Score forecasts with eval_crps and eval_coverage on held-out data, and backtest over rolling origins rather than trusting a single split.
- Write integers with four or more digits with underscores (1_000, 10_000) and thread PRNG keys explicitly with jax.random.split.

## Resources

- [Full documentation](https://juanitorduz.github.io/numpyro_forecast/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
- [Source code](https://github.com/juanitorduz/numpyro_forecast)
