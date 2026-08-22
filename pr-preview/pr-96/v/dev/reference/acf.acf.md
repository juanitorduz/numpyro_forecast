## acf.acf()


Compute the empirical autocorrelation function up to `max_lag`.


Usage

``` python
acf.acf(
    y,
    max_lag,
)
```


The lag-:math:`k` autocorrelation of a series :math:`y_1, \ldots, y_T` with sample mean :math:`\bar{y}` is estimated with the biased normalization

.. math::

    \hat{\rho}_k =
    \frac{\sum_{t=k+1}^{T} (y_t - \bar{y})(y_{t-k} - \bar{y})}
         {\sum_{t=1}^{T} (y_t - \bar{y})^2},

computed for all lags at once via the FFT of the centered series (the Wiener-Khinchin identity), so the cost is :math:`O(T \log T)` instead of :math:`O(T \, k_{\max})`. The computation runs along the last axis and broadcasts over any leading batch axes.


## Parameters


`y: Float[ArrayLike, ``" *batch time"]`  
Time series with time on the last axis, shape `(*batch, time)`. NumPy arrays are accepted directly.

`max_lag: int`  
Largest lag to evaluate; must satisfy `1 <= max_lag < time`.


## Returns


`Float[Array, ``"*batch lags"]`  
Autocorrelations for lags `0, 1, ..., max_lag` (`max_lag + 1` values; lag `0` is identically `1.0`).


## Raises


`ValueError`  
If `max_lag` is not in `[1, time)`.


## Notes

A constant series has zero variance, so the result is meaningless: all NaN in eager execution (a `0 / 0`), while under `jax.jit()` the rounding of the mean can instead yield arbitrary finite values.


## References

Adapted from `numpyro.diagnostics.autocorrelation()`, itself adapted from the Stan implementation.
