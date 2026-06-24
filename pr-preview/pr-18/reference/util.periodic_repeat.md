## util.periodic_repeat()


Tile a seasonal pattern to cover [duration](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.duration) time steps.


Usage

``` python
util.periodic_repeat(
    x,
    duration,
    *,
    axis=-1,
)
```


## Parameters


`x: Array`  
Seasonal pattern; the repeated axis has length equal to the period.

`duration: int`  
Target length along `axis`.

`axis: int = `<span class="dv">`-1`  
</span>  
Axis to repeat along (defaults to `-1`).


## Returns


`Array`  
`x` periodically repeated to length [duration](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.duration) along `axis`.
