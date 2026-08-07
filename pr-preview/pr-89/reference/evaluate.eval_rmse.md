## evaluate.eval_rmse()


Root mean squared error using the forecast sample mean as point estimate.


Usage

``` python
evaluate.eval_rmse(
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
The root mean squared error as a scalar array.
