## evaluate.evaluate_forecast()


Evaluate forecast samples against ground truth for several metrics at once.


Usage

``` python
evaluate.evaluate_forecast(
    pred, truth, *, metrics=None, coverage_alpha=_DEFAULT_COVERAGE_ALPHA
)
```


A one-call convenience that applies each metric in `metrics` to the same forecast samples and ground truth. It is the one-shot counterpart to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) and is also used internally by [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) to score each rolling window.


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first, shape `(sample, *batch)`.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values with shape `(*batch)`.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; when `None` defaults to `DEFAULT_METRICS` (`mae`, `rmse`, `crps` and `coverage`).

`coverage_alpha: float = _DEFAULT_COVERAGE_ALPHA`    
Nominal central-interval level for the default `coverage` metric (in `(0, 1)`, defaults to `0.9`). Only used on the default-metrics path; a custom `metrics` mapping controls its own coverage level.


## Returns


`dict[str, float]`  
Each metric name mapped to its value.
