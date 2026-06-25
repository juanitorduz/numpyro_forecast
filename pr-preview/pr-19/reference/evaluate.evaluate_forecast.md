## evaluate.evaluate_forecast()


Evaluate forecast samples against ground truth for several metrics at once.


Usage

``` python
evaluate.evaluate_forecast(
    pred,
    truth,
    *,
    metrics=DEFAULT_METRICS,
)
```


A one-call convenience that applies each metric in `metrics` to the same forecast samples and ground truth. This is the one-shot counterpart to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest), which reports the same metrics for each rolling window.


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first, shape `(sample, *batch)`.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values with shape `(*batch)`.

`metrics: Mapping[str, Metric] = DEFAULT_METRICS`    
Mapping of metric name to function; defaults to `DEFAULT_METRICS` (`mae`, `rmse`, `crps` and `coverage`).


## Returns


`dict[str, float]`  
Each metric name mapped to its value.
