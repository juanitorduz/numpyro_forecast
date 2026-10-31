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


Adds a `predictions` group (the forecast `obs` draws) and a `predictions_constant_data` group (the future covariates). The forecast `time` coordinate continues the in-sample one: integer continuation by default, or explicit values via `time_coord`. This is the step-by-step route for draws you produced yourself; [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) attaches the same groups automatically when its `covariates` extend beyond `data`. Both routes give the forecast groups the `(chain, draw)` layout of the posterior and the labels of the in-sample groups on every non-time axis, so xarray aligns the forecast with the rest of the tree draw by draw.


## Parameters


`tree: xarray.DataTree`  
A tree from [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree). The forecast continues its `observed_data` time coordinate, takes the `(chain, draw)` layout and labels of its `posterior` group, and copies the labels of the non-time axes of `observed_data["obs"]` and `constant_data["covariates"]` to the same axes of `predictions` and `predictions_constant_data`.

`forecast_samples: Num[ArrayLike, ``" sample future obs"]`  
Forecast draws shaped `(num_samples, future, obs)` from [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast), one per draw of the tree's `posterior` group as it is now, in the order of `tree["posterior"].stack(sample=("chain", "draw"))` (for a tree straight from [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree), the order of the flattened posterior it was built from). They are split into the `(chain, draw)` layout of that group; a group without a `chain` dimension (for example after `tree.isel(chain=0)`) counts as one chain. A tree with no `posterior` group uses its `posterior_predictive` group instead, and a tree with neither gets one pseudo-chain.

`covariates_future: Array`  
Future covariates shaped `(future, covariate_dim)`, or any layout with time at axis `-2` when `covariate_dims` names the axes, with one time step per forecast step.

`time_coord: Sequence[Any] | None = None`  
Optional explicit forecast time coordinate; defaults to integer continuation of the in-sample time. Required when the in-sample time coordinate is non-integer (e.g. datetime64): auto-continuing would have to guess the frequency, so explicit values are demanded instead.

`covariate_dims: Sequence[str] | None = None`  
Optional dimension names for `covariates_future`, one per axis. When omitted, the names are inherited from the tree's `constant_data["covariates"]` variable (falling back to `("time", "covariate_dim")` if the tree carries no stored covariates), so the forecast covariates always share the in-sample axis names. When given explicitly, the names must match the stored ones. See [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree).


## Returns


`xarray.DataTree`  
A new tree with the `predictions` and `predictions_constant_data` groups added.


## Raises


`ValueError`  
If `forecast_samples` does not have one draw per draw of the tree's posterior (to attach fewer draws, subset the tree first, for example with `tree.isel(draw=...)`), if `covariates_future` does not have one time step per forecast step, if a non-time axis of `forecast_samples` or `covariates_future` differs in size from the labels of its dimension on the tree's in-sample groups, if `time_coord` is given but its length differs from the forecast horizon, or if it is omitted while the in-sample time coordinate is non-integer.

`CovariateDimsError`  
If the resolved `covariate_dims` (explicit or inherited) do not name every `covariates_future` axis, or if explicit names disagree with the dimension names already stored on the tree's `constant_data["covariates"]`.
