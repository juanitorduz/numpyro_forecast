## names.Group


DataTree groups that [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) writes.


Usage

``` python
names.Group()
```


[add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups) adds the two forecast groups to an existing tree.


## Attributes


`POSTERIOR: Final`  
`"posterior"`, the posterior draws.

`POSTERIOR_PREDICTIVE: Final`  
`"posterior_predictive"`, the in-sample predictive draws.

`OBSERVED_DATA: Final`  
`"observed_data"`, the training data.

`CONSTANT_DATA: Final`  
`"constant_data"`, the in-sample covariates.

`PREDICTIONS: Final`  
`"predictions"`, the forecast draws (when the covariates extend past the data).

`PREDICTIONS_CONSTANT_DATA: Final`  
`"predictions_constant_data"`, the forecast covariates.
