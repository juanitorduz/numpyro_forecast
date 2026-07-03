## functional.predict_in_sample()


Sample the in-sample posterior predictive of the `obs` site.


Usage

``` python
functional.predict_in_sample(
    rng_key, model, posterior, covariates, *, batch_size=None
)
```


Runs `Predictive` with the in-sample `covariates` and the supplied posterior latent draws. Unlike [forecast()](functional.forecast.md#numpyro_forecast.functional.forecast) there is no forecast horizon: `covariates` span only the observed window, so the model's `obs` site is sampled at every step. The number of predictive samples equals the leading (sample) axis of `posterior` (see [draw_posterior()](functional.draw_posterior.md#numpyro_forecast.functional.draw_posterior)).


## Parameters


`rng_key: Array`  
PRNG key.

`model: ForecastModel`  
The forecasting model callable (the same one that produced `posterior`).

`posterior: dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading.

`covariates: Array`  
Covariates with time at axis `-2` spanning the observed window. Its time length must match the data the `posterior` was fit on, since the in-sample latent sites are sized to that window.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).


## Returns


`Float[Array, ``" sample *batch time obs"]`  
In-sample posterior-predictive draws of the `obs` site.
