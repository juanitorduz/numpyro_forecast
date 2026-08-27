## arrays.zero_data_like()


Return zeros shaped like `data` but extended to the covariate duration.


Usage

``` python
arrays.zero_data_like(
    data,
    covariates,
)
```


Mirrors Pyro's [zero_data](models.Horizon.md#numpyro_forecast.models.Horizon.zero_data): it exposes the shape/dtype of the data over the full forecast horizon without leaking observed values into the model. The functional API exposes the equivalent value as [numpyro_forecast.models.Horizon.zero_data](models.Horizon.md#numpyro_forecast.models.Horizon.zero_data).


## Parameters


`data: Array`  
Observed data with time at axis `-2`, shape `(*batch, t, obs)`.

`covariates: Array`  
Covariates with time at axis `-2`, shape `(*batch, duration, cov)`.


## Returns


`Array`  
Zeros of shape `(*batch, duration, obs)`.
