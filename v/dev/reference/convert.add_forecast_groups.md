## convert.add_forecast_groups()


Attach out-of-sample forecast groups to a copy of `tree`.


Usage

``` python
convert.add_forecast_groups(
    tree,
    forecast_samples,
    covariates_future,
    *,
    time_coord=None,
    covariate_dims=None
)
```


Adds a `predictions` group (the forecast `obs` draws) and a `predictions_constant_data` group (the future covariates). The forecast `time` coordinate continues the in-sample one: integer continuation by default, or explicit values via `time_coord`. This is the step-by-step route for draws you produced yourself; [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) attaches the same groups automatically when its `covariates` extend beyond `data`.


## Parameters


`tree: xarray.DataTree`  
A tree from [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) (its `observed_data` time coordinate is continued).

`forecast_samples: Array | np.ndarray`  
Forecast draws shaped `(num_samples, future, obs)` from `~numpyro_forecast.functional.prediction.forecast()`.

`covariates_future: Array`  
Future covariates shaped `(future, covariate_dim)`, or any layout with time at axis `-2` when `covariate_dims` names the axes.

`time_coord: Sequence[Any] | None = None`  
Optional explicit forecast time coordinate; defaults to integer continuation of the in-sample time. Required when the in-sample time coordinate is non-integer (e.g. datetime64): auto-continuing would have to guess the frequency, so explicit values are demanded instead.

`covariate_dims: Sequence[str] | None = None`  
Optional dimension names for `covariates_future`, one per axis. When omitted, the names are inherited from the tree's `constant_data["covariates"]` variable (falling back to `("time", "covariate_dim")` if the tree carries no stored covariates), so the forecast covariates always share the in-sample axis names. When given explicitly, the names must match the stored ones. See [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree).


## Returns


`xarray.DataTree`  
A new tree with the `predictions` and `predictions_constant_data` groups added.


## Raises


`ValueError`  
If `time_coord` is given but its length differs from the forecast horizon, or if it is omitted while the in-sample time coordinate is non-integer.

`CovariateDimsError`  
If the resolved `covariate_dims` (explicit or inherited) do not name every `covariates_future` axis, or if explicit names disagree with the dimension names already stored on the tree's `constant_data["covariates"]`.
