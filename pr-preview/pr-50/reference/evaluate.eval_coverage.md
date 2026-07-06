## evaluate.eval_coverage()


Empirical coverage of the central `alpha` prediction interval.


Usage

``` python
evaluate.eval_coverage(
    pred,
    truth,
    *,
    alpha=_DEFAULT_COVERAGE_ALPHA,
)
```


The central `alpha` interval is bounded by the `(1 - alpha) / 2` and `1 - (1 - alpha) / 2` quantiles of the forecast samples; the metric is the fraction of ground-truth values that fall inside it. A well-calibrated forecast has coverage close to `alpha`. A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`); bind a non-default level with `functools.partial(eval_coverage, alpha=...)`.


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).

`alpha: float = _DEFAULT_COVERAGE_ALPHA`    
Nominal interval level in `(0, 1)`; defaults to `0.9`.


## Returns


`Array`  
The fraction of ground truth inside the central `alpha` interval, as a scalar array.


## Raises


`ValueError`  
If `alpha` is not strictly inside `(0, 1)`.
