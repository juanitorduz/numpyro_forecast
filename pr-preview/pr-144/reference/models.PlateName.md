## models.PlateName


Names of the plates the building blocks open, as `str` members.


Usage

``` python
models.PlateName()
```


[TIME](models.PlateName.md#numpyro_forecast.models.PlateName.TIME) is the in-sample plate of [innovations()](models.innovations.md#numpyro_forecast.models.innovations); [TIME_FUTURE](models.PlateName.md#numpyro_forecast.models.PlateName.TIME_FUTURE) is the forecast-horizon plate of [innovations()](models.innovations.md#numpyro_forecast.models.innovations) and [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe). [time_reparam()](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) targets the sites under [TIME](models.PlateName.md#numpyro_forecast.models.PlateName.TIME) by name, so both modules read this enum rather than spelling the literal.


## Attributes

| Name | Description |
|----|----|
| [TIME](#TIME) |  |
| [TIME_FUTURE](#TIME_FUTURE) |  |

------------------------------------------------------------------------


#### TIME


`TIME=``"time"`


------------------------------------------------------------------------


#### TIME_FUTURE


`TIME_FUTURE=``"time_future"`
