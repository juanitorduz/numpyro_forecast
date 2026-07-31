## functional.models.forecasting_model()


Build a NumPyro model from a functional model body.


Usage

``` python
functional.models.forecasting_model(model_fn)
```


The functional analogue of subclassing `~numpyro_forecast.forecaster.ForecastingModel`. `model_fn` is a pure function `(Horizon, covariates) -> None` that calls [time_series()](functional.models.time_series.md#numpyro_forecast.functional.models.time_series) and [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict); this wraps it into the standard NumPyro model callable `(covariates, data=None)`, deriving the [Horizon](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon) from the shapes.


## Parameters


`model_fn: Callable[[Horizon, Array], None]`  
The model body. It receives the per-call [Horizon](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon) (use `h.zero_data` for the Pyro-style [zero_data](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon.zero_data)) and the covariates with time at axis `-2`.


## Returns


`ForecastModel`  
A callable `(covariates, data=None) -> None` usable with `SVI`, `MCMC`, `Predictive`, `~numpyro_forecast.functional.svi.fit_svi()`, `~numpyro_forecast.functional.mcmc.fit_mcmc()`, and the OOP forecaster classes.
