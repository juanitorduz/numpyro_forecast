## evaluate.backtest()


Backtest a forecasting model on a moving window of `(train, test)` data.


Usage

``` python
evaluate.backtest(
    rng_key,
    data,
    covariates,
    model_fn,
    *,
    forecaster_fn=Forecaster,
    metrics=None,
    coverage_alpha=_DEFAULT_COVERAGE_ALPHA,
    transform=None,
    train_window=None,
    min_train_window=1,
    test_window=None,
    min_test_window=1,
    stride=1,
    num_samples=100,
    batch_size=None,
    forecaster_options=None
)
```


## Parameters


`rng_key: Array`  
Base PRNG key (used for every window, matching Pyro).

`data: Array`  
Dataset with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` (same duration as `data`).

`model_fn: ModelFactory`  
Factory returning a fresh [ForecastingModel](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel) per window.

`forecaster_fn: ForecasterFactory = Forecaster`    
Factory returning a fitted forecaster (defaults to [Forecaster](forecaster.Forecaster.md#numpyro_forecast.forecaster.Forecaster)).

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; defaults to `DEFAULT_METRICS`.

`coverage_alpha: float = _DEFAULT_COVERAGE_ALPHA`    
Nominal central-interval level for the default `coverage` metric (in `(0, 1)`, defaults to `0.9`); forwarded to [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast).

`transform: Callable[[Array, Array], tuple[Array, Array]] | None = None`  
Optional `(pred, truth) -> (pred, truth)` applied before metrics.

`train_window: int | None = None`  
Training window size; if `None` the window expands from the start.

`min_train_window: int = ``1`  
Minimum training window size when `train_window` is `None`.

`test_window: int | None = None`  
Test window size; if `None` forecasts to the end of the data.

`min_test_window: int = ``1`  
Minimum test window size when `test_window` is `None`.

`stride: int = ``1`  
Step between successive train/test splits.

`num_samples: int = ``100`  
Number of forecast samples per window.

`batch_size: int | None = None`  
Optional forecast-sampling chunk size.

`forecaster_options: Mapping[str, Any] | Callable[…, Mapping[str, Any]] | None = None`  
Options dict passed to `forecaster_fn`, or a callable `(t0, t1, t2) -> dict` returning per-window options.


## Returns


`list[BacktestResult]`  
One result per backtest window.
