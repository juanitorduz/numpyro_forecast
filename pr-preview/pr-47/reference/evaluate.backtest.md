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
    per_window_metrics=None,
    transform=None,
    window_type=None,
    train_window=None,
    min_train_window=1,
    test_window=None,
    min_test_window=1,
    stride=1,
    num_samples=100,
    batch_size=None,
    forecaster_options=None,
    eval_train=False,
    keep_predictions=False,
    reuse_model=True
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
Mapping of metric name to function; defaults to `DEFAULT_METRICS`. Each function takes `(pred, truth)` and returns a scalar array (see `~numpyro_forecast.typing.Metric`); bind any metric-specific parameters with `functools.partial()`, e.g. `{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}`.

`per_window_metrics: Callable[[int, int, int], Mapping[str, Metric]] | None = None`  
Optional `(t0, t1, t2) -> Mapping[str, Metric]` callable producing extra metrics merged over `metrics` for each window. Use it for window-dependent metrics such as a MASE scaled by that window's training data ([numpyro_forecast.metrics.make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase)).

`transform: Callable[[Array, Array], tuple[Array, Array]] | None = None`  
Optional `(pred, truth) -> (pred, truth)` applied before metrics.

`window_type: WindowType | None = None`  
Windowing strategy. If `None` (default) it is inferred from `train_window`: `"expanding"` when `train_window` is `None` and `"rolling"` when it is set, matching the historical behavior. Pass `"expanding"` to always train on all history from `t0 = 0`, or `"rolling"` to hold the training length fixed at `train_window` and slide it forward. `"expanding"` and `train_window` are mutually exclusive, and `"rolling"` requires `train_window` (both validated).

`train_window: int | None = None`  
Training window size; if `None` the window expands from the start. Required for `window_type="rolling"`.

`min_train_window: int = ``1`  
Minimum training window size for the expanding strategy (used when `train_window` is `None`).

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

`eval_train: bool = ``False`  
If `True`, also score the in-sample posterior predictive over each training window with the same `metrics` and store them in `BacktestResult.train_metrics`. Requires a forecaster exposing [predict_in_sample](functional.predict_in_sample.md#numpyro_forecast.functional.predict_in_sample) (the built-in [Forecaster](forecaster.Forecaster.md#numpyro_forecast.forecaster.Forecaster) and [HMCForecaster](forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster) do).

`keep_predictions: bool = ``False`  
If `True`, store each window's out-of-sample forecast samples (after `transform`) on `BacktestResult.prediction`. Defaults to `False` to avoid retaining large Monte Carlo arrays.

`reuse_model: bool = ``True`  
When `True` (default) and the windowing strategy is rolling, the model instance returned by the first `model_fn()` call is reused for every window so forecast/predict kernels can cache across windows. SVI still recompiles per window; for a single fused fit over all windows use [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized). Ignored for expanding windows and when `False`.


## Returns


`list[BacktestResult]`  
One result per backtest window.
