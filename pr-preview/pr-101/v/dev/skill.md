---
name: numpyro_forecast
description: >
  A JAX/NumPyro port of Pyro's forecasting module. Use when writing Python code that uses the numpyro_forecast package.
license: Apache-2.0
compatibility: Requires Python >=3.12.
---

# numpyro_forecast

A JAX/NumPyro port of Pyro's forecasting module.

## Installation

```bash
pip install numpyro_forecast
```

## API overview

### Typing

Public type contracts.

- `typing.ForecastModel`
- `typing.ForecastFn`
- `typing.InSampleFn`

### Model building blocks

Plain model functions that register the train/forecast sites for you.

- `models.Horizon`
- `models.Transition`
- `models.innovations`
- `models.markov_series`
- `models.ssoe`
- `models.SSOEStep`
- `models.SSOEResult`
- `models.predict`

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

### Distribution surgery

Time-axis operations on observation distributions, extensible via singledispatch.

- `surgery.shift_loc`
- `surgery.slice_time`
- `surgery.prefix_condition`
- `surgery.register_elementwise`

### Optional dependencies

Lazy imports behind pyproject extras.

- `optional.require`

### Exceptions

Package exception hierarchy raised at resolution and validation boundaries.

- `exceptions.NumpyroForecastError`
- `exceptions.BacktestWindowError`
- `exceptions.VectorizedGuideError`
- `exceptions.VectorizedMetricError`
- `exceptions.KernelConfigError`
- `exceptions.CovariateDimsError`
- `exceptions.MVNLayoutError`
- `exceptions.DeviceMemoryError`

### ArviZ export

Convert fits into ArviZ-schema xarray DataTrees for diagnostics and plotting.

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
