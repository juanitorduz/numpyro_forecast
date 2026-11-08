## names.Site


NumPyro site names the building blocks register and the drivers read.


Usage

``` python
names.Site()
```


## Attributes


`OBS: Final`  
`"obs"`, the observation site. [predict()](models.predict.md#numpyro_forecast.models.predict) and [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) register it; [predict_in_sample()](predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) read it, and the drivers drop a posterior key with this name (issue \#260).

`FORECAST: Final`  
`"forecast"`, the forecast deterministic site over the horizon. [predict()](models.predict.md#numpyro_forecast.models.predict) and [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) register it; [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) and [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) read it, and a posterior key with this name and values is rejected (issue \#161).
