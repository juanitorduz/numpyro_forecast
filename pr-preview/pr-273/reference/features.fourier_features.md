## features.fourier_features()


Build a Fourier seasonality design matrix.


Usage

``` python
features.fourier_features(
    duration,
    period,
    num_terms,
)
```


Safe to call inside a model function (for example with `h.duration` of a [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon)), also under `svi.run` or the jitted [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast). The matrix is computed in float64 and cached per argument tuple, then returned in the current default float dtype (`float64` with x64 enabled, `float32` otherwise).


## Parameters


`duration: IntScalar`  
Number of time steps, a non-negative Python or NumPy integer (not a `float`).

`period: RealScalar`  
Seasonal period (in time steps): a positive, finite Python or NumPy integer or float, for example `7` for daily data with a weekly season. A NumPy float is used at its exact binary value, so `np.float32(0.1)` is the period `0.10000000149011612`, not `0.1`, and the two spellings are two memo entries.

`num_terms: IntScalar`  
Number of harmonics, a positive Python or NumPy integer; the output has `2 * num_terms` columns (sine then cosine).


## Returns


`Float[Array, ``"duration two_num_terms"]`  
The design matrix of shape `(duration, 2 * num_terms)`.


## Raises


`ValueError`  
If `duration` is negative, `period` is not positive and finite (`0`, `inf` or `nan` would cache an all-`inf` or all-`nan` matrix without an error), or `num_terms` is less than `1`.
