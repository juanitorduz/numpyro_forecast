## functional.forecast()


Sample forecasts for the steps in `[t, duration)` from a posterior.


Usage

``` python
functional.forecast(
    rng_key, model, posterior, data, covariates, *, batch_size=None
)
```


Runs `Predictive` with full-horizon `covariates` and the in-sample `data`: the in-sample latent sites are drawn from `posterior` while the `_future` suffix is drawn from the prior, and the `"forecast"` site is returned. The number of forecast samples equals the leading (sample) axis of `posterior` (see [draw_posterior()](functional.draw_posterior.md#numpyro_forecast.functional.draw_posterior)).


## Parameters


`rng_key: Array`  
PRNG key.

`model: ForecastModel`  
The forecasting model callable (the same one that produced `posterior`).

`posterior: dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading.

`data: Array`  
Observed data with time at axis `-2` and length `t`.

`covariates: Array`  
Covariates with time at axis `-2` and length `duration > t`.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).


## Returns


`Float[Array, ``" sample *batch future obs"]`  
Forecast samples over the `future = duration - t` horizon.


## Raises


`ValueError`  
If `covariates` does not extend beyond `data` along the time axis.
