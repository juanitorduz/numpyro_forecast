---
name: numpyro-forecast
description: >
  A JAX/NumPyro port of Pyro's forecasting module. Use when writing Python code that uses the numpyro_forecast package.
license: Apache-2.0
compatibility: Requires Python >=3.12.
---

# numpyro_forecast

A JAX/NumPyro port of Pyro's forecasting module.

## Installation

```bash
pip install numpyro-forecast
```

## API overview

### Forecasters

High-level interfaces for fitting and forecasting.

- `forecaster.Forecaster`
- `forecaster.HMCForecaster`
- `forecaster.PathfinderForecaster`

### Models

Building forecasting models (object-oriented and functional).

- `forecaster.ForecastingModel`
- `functional.models.forecasting_model`

### Functional core: model primitives

Pure functional primitives for the train/forecast split.

- `functional.models.Horizon`
- `functional.models.time_series`
- `functional.models.markov_time_series`
- `functional.models.predict`
- `functional.models.predict_glm`

### Functional core: fitting

Optimizer/guide/kernel resolution and the SVI and MCMC fit entry points.

- `functional.svi.resolve_optimizer`
- `functional.svi.resolve_guide`
- `functional.svi.fit_svi`
- `functional.svi.SVIFit`
- `functional.mcmc.resolve_kernel`
- `functional.mcmc.fit_mcmc`
- `functional.mcmc.MCMCFit`

### Functional core: posterior and prediction

Drawing posterior samples and generating forecasts and in-sample predictions.

- `functional.posterior.draw_posterior`
- `functional.prediction.forecast`
- `functional.prediction.predict_in_sample`

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

### Utilities

Array helpers and feature builders.

- `util.fourier_features`
- `util.periodic_repeat`
- `util.zero_data_like`
- `util.concat_future`
- `util.shift_loc`
- `util.slice_time`
- `util.prefix_condition`
- `util.register_elementwise`
- `util.require`

### Exceptions

Package exception hierarchy raised at resolution and validation boundaries.

- `exceptions.NumpyroForecastError`
- `exceptions.BacktestWindowError`
- `exceptions.VectorizedGuideError`
- `exceptions.VectorizedMetricError`
- `exceptions.OptimizerResolutionError`
- `exceptions.GuideResolutionError`
- `exceptions.GuideSampleArgsError`
- `exceptions.KernelResolutionError`
- `exceptions.KernelConfigError`
- `exceptions.MVNLayoutError`

### ArviZ export

Convert fits into ArviZ-schema xarray DataTrees for diagnostics and plotting.

- `convert.to_datatree`
- `convert.add_forecast_groups`

### Extensions (contrib)

Optional backends behind pyproject extras (never imported by default).

- `contrib.blackjax.BlackjaxNUTSKernel`
- `contrib.blackjax.BlackjaxMCLMCKernel`
- `contrib.blackjax.BlackjaxCustomKernel`
- `contrib.blackjax.PathfinderFit`
- `contrib.blackjax.fit_pathfinder`

### Datasets

Example datasets used in the tutorials.

- `datasets.load_bart_weekly`
- `datasets.load_bart_hierarchical`
- `datasets.load_victoria_electricity`
- `datasets.bart_available`

## Resources

- [Full documentation](https://juanitorduz.github.io/numpyro_forecast/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
- [Source code](https://github.com/juanitorduz/numpyro_forecast)
