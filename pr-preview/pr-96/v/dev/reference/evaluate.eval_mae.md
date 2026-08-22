## evaluate.eval_mae()


Mean absolute error using the forecast sample median as point estimate.


Usage

``` python
evaluate.eval_mae(
    pred,
    truth,
)
```


A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`). `pred` and `truth` are moved to device memory first (`~numpyro_forecast.functional._offload._device_view()`), so either (or both) may be host-committed, e.g. draws sampled with `device="host"`.


## Parameters


`pred: Float[ArrayLike, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Float[ArrayLike, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).


## Returns


`Array`  
The mean absolute error as a scalar array.
