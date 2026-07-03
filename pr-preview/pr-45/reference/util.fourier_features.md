## util.fourier_features()


Build a Fourier seasonality design matrix.


Usage

``` python
util.fourier_features(
    duration,
    period,
    num_terms,
)
```


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
