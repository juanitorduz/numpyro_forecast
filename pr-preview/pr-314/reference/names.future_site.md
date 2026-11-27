## names.future_site()


Return the name of the forecast-horizon site of the in-sample site `name`.


Usage

``` python
names.future_site(name)
```


[innovations()](models.innovations.md#numpyro_forecast.models.innovations), [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series) and [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) sample the horizon steps of `name` at this site, and [predict()](models.predict.md#numpyro_forecast.models.predict) samples the forecast suffix at `future_site(Site.OBS)`. The drivers reject a posterior key with values that ends in [FUTURE_SUFFIX](names.FUTURE_SUFFIX.md#numpyro_forecast.names.FUTURE_SUFFIX) (issue \#161).


## Parameters


`name: str`  
Name of the in-sample site.


## Returns


`str`  
`name` followed by [FUTURE_SUFFIX](names.FUTURE_SUFFIX.md#numpyro_forecast.names.FUTURE_SUFFIX).


## Examples

``` python
>>> from numpyro_forecast.names import future_site
>>> future_site("drift")
```

'drift_future'
