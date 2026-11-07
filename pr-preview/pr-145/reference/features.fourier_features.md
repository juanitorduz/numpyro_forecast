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


`duration: int`  
Number of time steps.

`period: float`  
Seasonal period (in time steps).

`num_terms: int`  
Number of harmonics; the output has `2 * num_terms` columns (sine then cosine).


## Returns


`Float[Array, ``"duration two_num_terms"]`  
The design matrix of shape `(duration, 2 * num_terms)`.
