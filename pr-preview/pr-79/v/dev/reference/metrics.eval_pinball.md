## metrics.eval_pinball()


Mean pinball (quantile) loss of the forecast `quantile`.


Usage

``` python
metrics.eval_pinball(
    pred,
    truth,
    *,
    quantile=0.5,
)
```


The pinball loss for the forecast \\\hat q\\ of quantile \\\tau\\ is \\\max(\tau (y - \hat q), (\tau - 1)(y - \hat q))\\, averaged over all data elements. At `quantile=0.5` it is half the mean absolute error. A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`); `quantile` is static so each level specializes its own branch.


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).

`quantile: float = ``0.5`  
Target quantile in `(0, 1)`.


## Returns


`Array`  
The mean pinball loss as a scalar array.


## Raises


`ValueError`  
If `quantile` is not strictly inside `(0, 1)`.
