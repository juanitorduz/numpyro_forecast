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


The central `alpha` interval is bounded by the `(1 - alpha) / 2` and `1 - (1 - alpha) / 2` quantiles of the forecast samples; the metric is the fraction of ground-truth values that fall inside it. A well-calibrated forecast has coverage close to `alpha`.


## Parameters


`pred: Array`  
Forecast samples with the sample axis first.

`truth: Array`  
Ground-truth values (matching `pred` without the sample axis).

`alpha: float = _DEFAULT_COVERAGE_ALPHA`    
Nominal interval level in `(0, 1)`; when omitted, uses the module default `_DEFAULT_COVERAGE_ALPHA`.


## Returns


`float`  
The fraction of ground-truth values inside the central `alpha` interval.
