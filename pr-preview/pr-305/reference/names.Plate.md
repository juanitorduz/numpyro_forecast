## names.Plate


Names of the plates the building blocks open.


Usage

``` python
names.Plate()
```


## Attributes


`TIME: Final`  
`"time"`, the in-sample plate of [innovations()](models.innovations.md#numpyro_forecast.models.innovations); [time_reparam()](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) targets the sites under it by this name.

`TIME_FUTURE: Final`  
`"time_future"`, the forecast-horizon plate of [innovations()](models.innovations.md#numpyro_forecast.models.innovations) and [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe).
