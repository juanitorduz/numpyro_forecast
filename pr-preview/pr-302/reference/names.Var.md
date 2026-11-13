## names.Var


Variables of the trees [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) writes.


Usage

``` python
names.Var()
```


[add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups) reads both back to inherit the stored dimension names and labels.


## Attributes


`OBS: Final`  
`"obs"`, the observed, predicted and forecast values; it has the value of `Site.OBS` because the tree names the variable after the site.

`COVARIATES: Final`  
`"covariates"`, the in-sample and forecast covariates of the constant-data groups.
