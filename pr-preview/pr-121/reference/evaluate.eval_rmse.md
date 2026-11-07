## evaluate.eval_rmse()


Root mean squared error using the forecast sample mean as point estimate.


Usage

``` python
evaluate.eval_rmse(
    pred,
    truth,
)
```


A pure JAX scalar kernel (see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric)). `pred` and `truth` are moved to device memory first (`float_view()`), so either (or both) may be host-committed, e.g. draws sampled with `device="host"`. An integer operand (the draws of a count model) is cast to the default float dtype in the same step.


## Parameters


`pred: Real[ArrayLike, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Real[ArrayLike, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).


## Returns


`Array`  
The root mean squared error as a scalar array.


## Raises


`ValueError`  
If `truth.shape` is not `pred.shape[1:]` (see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric)).
