## exceptions.MVNLayoutError


A `MultivariateNormal` layout is unsupported for time-axis surgery.


Usage

``` python
exceptions.MVNLayoutError(message=None)
```


Raised by [shift_loc()](surgery.shift_loc.md#numpyro_forecast.surgery.shift_loc), [slice_time()](surgery.slice_time.md#numpyro_forecast.surgery.slice_time), and [prefix_condition()](surgery.prefix_condition.md#numpyro_forecast.surgery.prefix_condition) on MVN noise whose `loc`/`covariance_matrix` shapes do not match the supported time-leading layout.
