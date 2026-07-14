## evaluate.eval_crps()


Empirical CRPS averaged over all data elements.


Usage

``` python
evaluate.eval_crps(
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
The mean empirical CRPS as a scalar array.
