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


For the central `alpha` interval \\\[l, u\]\\ (the \\(1-\alpha)/2\\ and \\1-(1-\alpha)/2\\ quantiles), the interval score is \\(u - l) + \tfrac{2}{1-\alpha}\big\[(l - y)\mathbf 1\_{y\<l} + (y - u)\mathbf 1\_{y\>u}\big\]\\, averaged over all data elements. It rewards narrow intervals and penalizes ground truth falling outside them; lower is better.


## Parameters


`pred: Array`  
Forecast samples with the sample axis first.

`truth: Array`  
Ground-truth values (matching `pred` without the sample axis).

`alpha: float = ``0.9`  
Nominal interval level in `(0, 1)`.


## Returns


`float`  
The mean interval score.


## Raises


`ValueError`  
If `alpha` is not strictly inside `(0, 1)`.
