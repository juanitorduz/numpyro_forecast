## acf.pacf()


Compute the empirical partial autocorrelation function up to `max_lag`.


Usage

``` python
acf.pacf(
    y,
    max_lag,
)
```


The lag-k partial autocorrelation is the correlation between y_t and y\_{t-k} after removing the linear effect of the intermediate observations y\_{t-1}, \ldots, y\_{t-k+1}; it equals the last coefficient \phi\_{kk} of the best linear predictor of order k. The coefficients are obtained from the empirical autocorrelations (see [acf()](acf.acf.md#numpyro_forecast.acf.acf)) with the Durbin-Levinson recursion

 \phi\_{kk} = \frac{\hat{\rho}\_k - \sum\_{j=1}^{k-1} \phi\_{k-1,j}\\ \hat{\rho}\_{k-j}} {1 - \sum\_{j=1}^{k-1} \phi\_{k-1,j}\\ \hat{\rho}\_j}. 

The recursion runs along the last axis and broadcasts over any leading batch axes.


## Parameters


`y: Float[ArrayLike, ``" *batch time"]`  
Time series with time on the last axis, shape `(*batch, time)`. NumPy arrays are accepted directly.

`max_lag: int`  
Largest lag to evaluate; must satisfy `1 <= max_lag < time`.


## Returns


`Float[Array, ``"*batch lags"]`  
Partial autocorrelations for lags `0, 1, ..., max_lag` (`max_lag + 1` values; lag `0` is identically `1.0`).


## Raises


`ValueError`  
If `max_lag` is not in `[1, time)`.


## Notes

A constant series has zero variance, so the result is meaningless (see [acf()](acf.acf.md#numpyro_forecast.acf.acf)); a near-deterministic series can likewise overflow the recursion (the denominator above approaches zero) and produce infinities or NaNs.


## References

James Durbin (1960). "The Fitting of Time-Series Models". *Revue de l'Institut International de Statistique*.
