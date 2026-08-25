# API Reference


## Typing


Public type contracts.


[typing.ForecastModel](typing.ForecastModel.md#numpyro_forecast.typing.ForecastModel)  
A NumPyro forecasting model: a callable `(covariates, data=None) -> None`.


## Functional core: model primitives


Pure functional primitives for the train/forecast split.


[functional.models.Horizon](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon)  
The train/forecast split for a single model call.

[functional.models.Transition](functional.models.Transition.md#numpyro_forecast.functional.models.Transition)  
(carry, x_t) -\> (dist_t, carry_fn) where carry_fn(z_t) builds the next carry

[functional.models.time_series()](functional.models.time_series.md#numpyro_forecast.functional.models.time_series)  
Sample a time-varying latent over the full horizon.

[functional.models.markov_time_series()](functional.models.markov_time_series.md#numpyro_forecast.functional.models.markov_time_series)  
Sample a Markov (state-space) latent over the full horizon.

[functional.models.predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict)  
Register the observation/forecast sites for the model.

[functional.models.predict_glm()](functional.models.predict_glm.md#numpyro_forecast.functional.models.predict_glm)  
Register GLM-style observation/forecast sites from a latent predictor.


## Functional core: posterior and prediction


Drawing posterior samples and generating forecasts and in-sample predictions.


[functional.posterior.draw_posterior()](functional.posterior.draw_posterior.md#numpyro_forecast.functional.posterior.draw_posterior)  
Draw `num_samples` posterior samples of the latent sites from a fitted guide.

[functional.prediction.forecast()](functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast)  
Sample forecasts for the steps in `[t, duration)` from a posterior.

[functional.prediction.predict_in_sample()](functional.prediction.predict_in_sample.md#numpyro_forecast.functional.prediction.predict_in_sample)  
Sample the in-sample posterior predictive of the `obs` site.


## Backtesting & evaluation


Rolling-window backtesting and forecast metrics.


[evaluate.backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)  
Backtest a forecasting model on a moving window of `(train, test)` data.

[evaluate.backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized)  
Rolling-window backtest with all windows fitted in one vmapped SVI run.

[evaluate.BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult)  
Per-window result of a `backtest()` run.

[evaluate.VectorizedBacktestResult](evaluate.VectorizedBacktestResult.md#numpyro_forecast.evaluate.VectorizedBacktestResult)  
Result of a `backtest_vectorized()` run (all windows at once).

[evaluate.evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast)  
Evaluate forecast samples against ground truth for several metrics at once.

[evaluate.results_to_dataframe()](evaluate.results_to_dataframe.md#numpyro_forecast.evaluate.results_to_dataframe)  
Flatten backtest results into a tidy one-row-per-window `DataFrame`.

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

[metrics.eval_pinball()](metrics.eval_pinball.md#numpyro_forecast.metrics.eval_pinball)  
Mean pinball (quantile) loss of the forecast `quantile`.

[metrics.eval_interval_score()](metrics.eval_interval_score.md#numpyro_forecast.metrics.eval_interval_score)  
Mean Winkler interval score for the central `alpha` prediction interval.

[metrics.make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase)  
Build a Mean Absolute Scaled Error metric scaled by `train_data`.


## Autocorrelation


Batched autocorrelation and partial autocorrelation diagnostics.


[acf.acf()](acf.acf.md#numpyro_forecast.acf.acf)  
Compute the empirical autocorrelation function up to `max_lag`.

[acf.pacf()](acf.pacf.md#numpyro_forecast.acf.pacf)  
Compute the empirical partial autocorrelation function up to `max_lag`.


## Seasonal features


Fourier design matrices and seasonal tiling.


[features.fourier_features()](features.fourier_features.md#numpyro_forecast.features.fourier_features)  
Build a Fourier seasonality design matrix.

[features.periodic_repeat()](features.periodic_repeat.md#numpyro_forecast.features.periodic_repeat)  
Tile a seasonal pattern to cover `duration` time steps.


## Array helpers


Time-axis array shaping for the train/forecast split.


[arrays.zero_data_like()](arrays.zero_data_like.md#numpyro_forecast.arrays.zero_data_like)  
Return zeros shaped like `data` but extended to the covariate duration.

[arrays.concat_future()](arrays.concat_future.md#numpyro_forecast.arrays.concat_future)  
Concatenate in-sample and forecast-horizon arrays along the time axis.


## Distribution surgery


Time-axis operations on observation distributions, extensible via singledispatch.


[surgery.shift_loc()](surgery.shift_loc.md#numpyro_forecast.surgery.shift_loc)  
Re-center a zero-centered noise distribution at `loc`.

[surgery.slice_time()](surgery.slice_time.md#numpyro_forecast.surgery.slice_time)  
Slice an elementwise distribution along the time axis `-2`.

[surgery.prefix_condition()](surgery.prefix_condition.md#numpyro_forecast.surgery.prefix_condition)  
Condition a `(t+f)`-length distribution on a `t`-length data prefix.

[surgery.register_elementwise()](surgery.register_elementwise.md#numpyro_forecast.surgery.register_elementwise)  
Declare a distribution family elementwise (usable as a decorator).


## Optional dependencies


Lazy imports behind pyproject extras.


[optional.require()](optional.require.md#numpyro_forecast.optional.require)  
Import an optional dependency, or raise a targeted `ImportError`.


## Exceptions


Package exception hierarchy raised at resolution and validation boundaries.


[exceptions.NumpyroForecastError](exceptions.NumpyroForecastError.md#numpyro_forecast.exceptions.NumpyroForecastError)  
Base class for all deliberate `numpyro_forecast` errors.

[exceptions.BacktestWindowError](exceptions.BacktestWindowError.md#numpyro_forecast.exceptions.BacktestWindowError)  
A backtest window configuration is invalid.

[exceptions.VectorizedGuideError](exceptions.VectorizedGuideError.md#numpyro_forecast.exceptions.VectorizedGuideError)  
The vectorized backtest requires an `AutoGuide` instance.

[exceptions.VectorizedMetricError](exceptions.VectorizedMetricError.md#numpyro_forecast.exceptions.VectorizedMetricError)  
A metric is not vmappable in the vectorized backtest.

[exceptions.KernelConfigError](exceptions.KernelConfigError.md#numpyro_forecast.exceptions.KernelConfigError)  
A `contrib.blackjax` kernel is run unbound or misconfigured.

[exceptions.CovariateDimsError](exceptions.CovariateDimsError.md#numpyro_forecast.exceptions.CovariateDimsError)  
Covariate dimension names are inconsistent or malformed.

[exceptions.MVNLayoutError](exceptions.MVNLayoutError.md#numpyro_forecast.exceptions.MVNLayoutError)  
A `MultivariateNormal` layout is unsupported for time-axis surgery.

[exceptions.DeviceMemoryError](exceptions.DeviceMemoryError.md#numpyro_forecast.exceptions.DeviceMemoryError)  
A memory pool ran out during posterior or predictive sampling.


## ArviZ export


Convert fits into ArviZ-schema xarray DataTrees for diagnostics and plotting.


[convert.to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree)  
Convert an already-drawn posterior into an ArviZ-schema `xarray.DataTree`.

[convert.add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups)  
Attach out-of-sample forecast groups to a copy of `tree`.

[convert.predictions_to_datatree()](convert.predictions_to_datatree.md#numpyro_forecast.convert.predictions_to_datatree)  
Pack prediction draws into a DataTree laid out for per-series `plot_lm` faceting.


## Extensions (contrib)


Optional backends behind pyproject extras (never imported by default).


[contrib.blackjax.BlackjaxNUTSKernel](contrib.blackjax.BlackjaxNUTSKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxNUTSKernel)  
BlackJAX NUTS with Stan-style window adaptation.

[contrib.blackjax.BlackjaxMCLMCKernel](contrib.blackjax.BlackjaxMCLMCKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxMCLMCKernel)  
BlackJAX Microcanonical Langevin Monte Carlo (MCLMC).

[contrib.blackjax.BlackjaxCustomKernel](contrib.blackjax.BlackjaxCustomKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel)  
Adapt an arbitrary BlackJAX sampler via a user-supplied `build_fn`.

[contrib.blackjax.PathfinderFit](contrib.blackjax.PathfinderFit.md#numpyro_forecast.contrib.blackjax.PathfinderFit)  
The result of fitting a forecasting model with BlackJAX Pathfinder.

[contrib.blackjax.fit_pathfinder()](contrib.blackjax.fit_pathfinder.md#numpyro_forecast.contrib.blackjax.fit_pathfinder)  
Fit a forecasting model with BlackJAX Pathfinder variational inference.


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
