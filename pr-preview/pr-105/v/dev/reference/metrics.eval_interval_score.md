## metrics.eval_interval_score()


Mean Winkler interval score for the central `alpha` prediction interval.


Usage

``` python
metrics.eval_interval_score(
    pred,
    truth,
    *,
    alpha=0.9,
)
```


For the central `alpha` interval :math:`[l, u]` (the :math:`(1-\alpha)/2` and :math:`1-(1-\alpha)/2` quantiles), the interval score is :math:`(u - l) + \tfrac{2}{1-\alpha}\big[(l - y)\mathbf 1_{y<l} + (y - u)\mathbf 1_{y>u}\big]`, averaged over all data elements. It rewards narrow intervals and penalizes ground truth falling outside them; lower is better. A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`). `pred` and `truth` are moved to device memory first (`~numpyro_forecast._offload._device_view()`), so either (or both) may be host-committed, e.g. draws sampled with `device="host"`.


## Parameters


`pred: Float[ArrayLike, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Float[ArrayLike, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).

`alpha: float = ``0.9`  
Nominal interval level in `(0, 1)`.


## Returns


`Array`  
The mean interval score as a scalar array.


## Raises


`ValueError`  
If `alpha` is not strictly inside `(0, 1)`.
