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


A one-call convenience that applies each metric in `metrics` to the same forecast samples and ground truth, converting to host floats at the end. It is the one-shot counterpart to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) and is also used internally by [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) to score each rolling window.

Metric-specific parameters live with the metric in the `metrics` mapping, not on this function. To tune a metric, bind its keyword with `functools.partial()`; for example, to score coverage at the 80% level:

``` python
from functools import partial

metrics = {**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}
evaluate_forecast(pred, truth, metrics=metrics)
```


## Parameters


`pred: Real[ArrayLike, ``" sample *batch"]`  
Forecast samples with the sample axis first, shape `(sample, *batch)`. Integer draws (a count model) are accepted.

`truth: Real[ArrayLike, ``" *batch"]`  
Ground-truth values with shape `(*batch)`. Integer values are accepted.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; when `None` defaults to [DEFAULT_METRICS](evaluate.DEFAULT_METRICS.md#numpyro_forecast.evaluate.DEFAULT_METRICS) (`mae`, `rmse`, `crps` and `coverage`). Each function takes `(pred, truth)` and returns a 0-d array (see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric)); bind any extra parameters with `functools.partial()` (see above). Each function receives floating arrays: an integer `pred` or `truth` is cast to the default float dtype once, before the first metric runs. A per-cell function such as [crps_empirical()](metrics.crps_empirical.md#numpyro_forecast.metrics.crps_empirical) is not a metric: use [eval_crps()](evaluate.eval_crps.md#numpyro_forecast.evaluate.eval_crps), its mean over the cells.


## Returns


`dict[str, float]`  
Each metric name mapped to its value. A bool, integer, `float8` or `float4` metric result is converted to `float`, whatever the other metrics are (a `float6` result keeps its dtype: XLA on CPU cannot convert it).


## Raises


`ValueError`  
If `truth.shape` is not `pred.shape[1:]` (see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric)). The check runs here once, so a custom metric in `metrics` never sees a mismatched pair. Also if a metric returns a value that is not 0-d or does not have a bool, integer or floating dtype; the message names the metric and gives the shape and dtype it returned.
