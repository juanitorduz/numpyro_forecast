# Reference


## Forecasters


High-level interfaces for fitting and forecasting.


[forecaster.Forecaster](forecaster.Forecaster.md#numpyro_forecast.forecaster.Forecaster)  
Fit a forecasting model with stochastic variational inference.

[forecaster.HMCForecaster](forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster)  
Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).


## Models


Building forecasting models (object-oriented and functional).


[forecaster.ForecastingModel](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel)  
Abstract base class for forecasting models.

[functional.forecasting_model()](functional.forecasting_model.md#numpyro_forecast.functional.forecasting_model)  
Build a NumPyro model from a functional model body.


## Functional core


Pure functional primitives for the train/forecast split.


[functional.Horizon](functional.Horizon.md#numpyro_forecast.functional.Horizon)  
The train/forecast split for a single model call.

[functional.time_series()](functional.time_series.md#numpyro_forecast.functional.time_series)  
Sample a time-varying latent over the full horizon.

[functional.predict()](functional.predict.md#numpyro_forecast.functional.predict)  
Register the observation/forecast sites for the model.

[functional.fit_svi()](functional.fit_svi.md#numpyro_forecast.functional.fit_svi)  
Fit a forecasting model with stochastic variational inference.

[functional.draw_posterior()](functional.draw_posterior.md#numpyro_forecast.functional.draw_posterior)  
Draw `num_samples` posterior samples of the latent sites from a fit.

[functional.fit_mcmc()](functional.fit_mcmc.md#numpyro_forecast.functional.fit_mcmc)  
Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).

[functional.forecast()](functional.forecast.md#numpyro_forecast.functional.forecast)  
Sample forecasts for the steps in `[t, duration)` from a posterior.

[functional.SVIFit](functional.SVIFit.md#numpyro_forecast.functional.SVIFit)  
The result of fitting a forecasting model with SVI.

[functional.MCMCFit](functional.MCMCFit.md#numpyro_forecast.functional.MCMCFit)  
The result of fitting a forecasting model with MCMC (NUTS).


## Backtesting & evaluation


Rolling-window backtesting and forecast metrics.


[evaluate.backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)  
Backtest a forecasting model on a moving window of `(train, test)` data.

[evaluate.BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult)  
Per-window result of a `backtest()` run.

[evaluate.evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast)  
Evaluate forecast samples against ground truth for several metrics at once.

[evaluate.eval_crps()](evaluate.eval_crps.md#numpyro_forecast.evaluate.eval_crps)  
Empirical CRPS averaged over all data elements.

[evaluate.eval_mae()](evaluate.eval_mae.md#numpyro_forecast.evaluate.eval_mae)  
Mean absolute error using the forecast sample median as point estimate.

[evaluate.eval_rmse()](evaluate.eval_rmse.md#numpyro_forecast.evaluate.eval_rmse)  
Root mean squared error using the forecast sample mean as point estimate.

[evaluate.eval_coverage()](evaluate.eval_coverage.md#numpyro_forecast.evaluate.eval_coverage)  
Empirical coverage of the central `alpha` prediction interval.

[metrics.crps_empirical()](metrics.crps_empirical.md#numpyro_forecast.metrics.crps_empirical)  
Compute the empirical Continuous Ranked Probability Score (CRPS).


## Utilities


Array helpers and feature builders.


[util.fourier_features()](util.fourier_features.md#numpyro_forecast.util.fourier_features)  
Build a Fourier seasonality design matrix.

[util.periodic_repeat()](util.periodic_repeat.md#numpyro_forecast.util.periodic_repeat)  
Tile a seasonal pattern to cover `duration` time steps.

[util.zero_data_like()](util.zero_data_like.md#numpyro_forecast.util.zero_data_like)  
Return zeros shaped like `data` but extended to the covariate duration.

[util.concat_future()](util.concat_future.md#numpyro_forecast.util.concat_future)  
Concatenate in-sample and forecast-horizon arrays along the time axis.

[util.shift_loc()](util.shift_loc.md#numpyro_forecast.util.shift_loc)  
Re-center a zero-centered noise distribution at `loc`.

[util.slice_time()](util.slice_time.md#numpyro_forecast.util.slice_time)  
Slice an elementwise distribution along the time axis `-2`.

[util.prefix_condition()](util.prefix_condition.md#numpyro_forecast.util.prefix_condition)  
Condition a `(t+f)`-length distribution on a `t`-length data prefix.


## Datasets


Example datasets used in the tutorials.


[datasets.load_bart_weekly()](datasets.load_bart_weekly.md#numpyro_forecast.datasets.load_bart_weekly)  
Load total weekly BART ridership (log scale) for the univariate example.

[datasets.load_bart_hierarchical()](datasets.load_bart_hierarchical.md#numpyro_forecast.datasets.load_bart_hierarchical)  
Load the windowed hierarchical BART panel for the hierarchical example.

[datasets.load_victoria_electricity()](datasets.load_victoria_electricity.md#numpyro_forecast.datasets.load_victoria_electricity)  
Load hourly Victoria (Australia) electricity demand and temperature.

[datasets.bart_available()](datasets.bart_available.md#numpyro_forecast.datasets.bart_available)  
Return whether the BART dataset can be loaded (download succeeds).
