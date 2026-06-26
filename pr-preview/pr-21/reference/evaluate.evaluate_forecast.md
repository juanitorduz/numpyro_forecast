## evaluate.evaluate_forecast()


Evaluate forecast samples against ground truth for several metrics at once.


Usage

``` python
evaluate.evaluate_forecast(
    pred,
    truth,
    *,
    metrics=None,
)
```


A one-call convenience that applies each metric in `metrics` to the same forecast samples and ground truth. It is the one-shot counterpart to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) and is also used internally by [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) to score each rolling window.

Metric-specific parameters live with the metric in the `metrics` mapping, not on this function. To tune a metric, bind its keyword with `functools.partial()`; for example, to score coverage at the 80% level:

``` python
from functools import partial
```

    metrics = {**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}
    evaluate_forecast(pred, truth, metrics=metrics)


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first, shape `(sample, *batch)`.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values with shape `(*batch)`.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; when `None` defaults to `DEFAULT_METRICS` (`mae`, `rmse`, `crps` and `coverage`). Each function takes `(pred, truth)` and returns a float; bind any extra parameters with `functools.partial()` (see above).


## Returns


`dict[str, float]`  
Each metric name mapped to its value.
