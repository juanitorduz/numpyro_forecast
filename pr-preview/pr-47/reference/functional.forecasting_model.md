## functional.forecasting_model()


Build a NumPyro model from a functional model body.


Usage

``` python
functional.forecasting_model(model_fn)
```


The functional analogue of subclassing `~numpyro_forecast.forecaster.ForecastingModel`. `model_fn` is a pure function `(Horizon, covariates) -> None` that calls [time_series()](functional.time_series.md#numpyro_forecast.functional.time_series) and [predict()](functional.predict.md#numpyro_forecast.functional.predict); this wraps it into the standard NumPyro model callable `(covariates, data=None)`, deriving the [Horizon](functional.Horizon.md#numpyro_forecast.functional.Horizon) from the shapes.


## Parameters


`model_fn: Callable[[Horizon, Array], None]`  
The model body. It receives the per-call [Horizon](functional.Horizon.md#numpyro_forecast.functional.Horizon) (use `h.zero_data` for the Pyro-style [zero_data](functional.Horizon.md#numpyro_forecast.functional.Horizon.zero_data)) and the covariates with time at axis `-2`.


## Returns


`ForecastModel`  
A callable `(covariates, data=None) -> None` usable with `SVI`, `MCMC`, `Predictive`, [fit_svi()](functional.fit_svi.md#numpyro_forecast.functional.fit_svi), [fit_mcmc()](functional.fit_mcmc.md#numpyro_forecast.functional.fit_mcmc), and the OOP forecaster classes.
