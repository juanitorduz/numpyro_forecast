## evaluate.eval_mae()


Mean absolute error using the forecast sample median as point estimate.


Usage

``` python
evaluate.eval_mae(
    pred,
    truth,
)
```


A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`).


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).


## Returns


`Array`  
The mean absolute error as a scalar array.
