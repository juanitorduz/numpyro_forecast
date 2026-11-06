## exceptions.CovariateDimsError


Covariate dimension names are inconsistent or malformed.


Usage

``` python
exceptions.CovariateDimsError(message=None)
```


Raised by [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and [add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups) when `covariate_dims` does not name every covariates axis, or when the names passed to (or inherited by) [add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups) disagree with the dimension names already stored on the tree's `constant_data` covariates.
