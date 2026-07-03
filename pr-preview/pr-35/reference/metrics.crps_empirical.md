## metrics.crps_empirical()


Compute the empirical Continuous Ranked Probability Score (CRPS).


Usage

``` python
metrics.crps_empirical(
    pred,
    truth,
)
```


The CRPS generalises the mean absolute error to probabilistic forecasts and is computed elementwise as

\\ \mathrm{CRPS}(F, y) = \mathbb{E}\|X - y\| - \tfrac{1}{2}\\\mathbb{E}\|X - X'\|, \\

where \\X, X'\\ are independent draws from the forecast distribution \\F\\. The expectations are estimated from the forecast `sample` axis using the sorted-sample \\O(n \log n)\\ identity.


## Parameters


`pred: Float[Array, ``" sample *batch"]`  
Forecast samples with the sample axis first, shape `(sample, *batch)`.

`truth: Float[Array, ``" *batch"]`  
Ground-truth values with shape `(*batch)` (broadcastable to `pred`).


## Returns


`Float[Array, ``"*batch"]`  
Elementwise CRPS, one value per `batch` location.


## References

Tilmann Gneiting, Adrian E. Raftery (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation". *Journal of the American Statistical Association*.
