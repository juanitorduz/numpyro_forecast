## convert.add_forecast()


Attach out-of-sample forecast groups to a copy of `tree`.


Usage

``` python
convert.add_forecast(
    tree, forecast_samples, covariates_future, *, time_coord=None
)
```


Adds a `predictions` group (the forecast `obs` draws) and a `predictions_constant_data` group (the future covariates). The forecast `time` coordinate continues the in-sample one: integer continuation by default, or explicit values via `time_coord`.


## Parameters


`tree: xarray.DataTree`  
A tree from [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) (its `observed_data` time coordinate is continued).

`forecast_samples: Array`  
Forecast draws shaped `(num_samples, future, obs)` from `~numpyro_forecast.functional.forecast()`.

`covariates_future: Array`  
Future covariates shaped `(future, covariate_dim)`.

`time_coord: Sequence[Any] | None = None`  
Optional explicit forecast time coordinate; defaults to integer continuation of the in-sample time.


## Returns


`xarray.DataTree`  
A new tree with the `predictions` and `predictions_constant_data` groups added.
