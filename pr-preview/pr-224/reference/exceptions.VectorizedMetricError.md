## exceptions.VectorizedMetricError


A metric is not vmappable in the vectorized backtest.


Usage

``` python
exceptions.VectorizedMetricError(message=None)
```


Raised by [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) when a metric forces a host conversion (e.g. `float(...)` or `numpy` calls) under `vmap`. Metrics must be pure JAX functions returning a scalar array; see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric).
