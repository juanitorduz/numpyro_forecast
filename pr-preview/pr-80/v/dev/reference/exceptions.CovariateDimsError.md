## exceptions.CovariateDimsError


Covariate dimension names are inconsistent or malformed.


Usage

``` python
exceptions.CovariateDimsError(message=None)
```


Raised by `~numpyro_forecast.convert.to_datatree()` and `~numpyro_forecast.convert.add_forecast_groups()` when `covariate_dims` does not name every covariates axis, or when the names passed to (or inherited by) `~numpyro_forecast.convert.add_forecast_groups()` disagree with the dimension names already stored on the tree's `constant_data` covariates.
