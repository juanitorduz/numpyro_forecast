## exceptions.MVNLayoutError


A `MultivariateNormal` layout is unsupported for time-axis surgery.


Usage

``` python
exceptions.MVNLayoutError(message=None)
```


Raised by `~numpyro_forecast.util.shift_loc()`, `~numpyro_forecast.util.slice_time()`, and `~numpyro_forecast.util.prefix_condition()` on MVN noise whose `loc`/`covariance_matrix` shapes do not match the supported time-leading layout.
