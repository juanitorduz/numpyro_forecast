# Reference


## Forecasters


High-level interfaces for fitting and forecasting.


[forecaster.Forecaster](forecaster.Forecaster.md#numpyro_forecast.forecaster.Forecaster)  
Fit a forecasting model with stochastic variational inference.

[forecaster.HMCForecaster](forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster)  
Fit a forecasting model with MCMC (NUTS by default).


## Models


Building forecasting models (object-oriented and functional).


[forecaster.ForecastingModel](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel)  
Abstract base class for forecasting models.


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
