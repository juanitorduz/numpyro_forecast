## functional.draw_posterior()


Draw `num_samples` posterior samples of the latent sites from a fit.


Usage

``` python
functional.draw_posterior(
    rng_key,
    fit,
    num_samples,
)
```


Dispatches on the fit type (e.g. [SVIFit](functional.SVIFit.md#numpyro_forecast.functional.SVIFit), [MCMCFit](functional.MCMCFit.md#numpyro_forecast.functional.MCMCFit)). The returned dict has the sample axis leading and is ready to pass to [forecast()](functional.forecast.md#numpyro_forecast.functional.forecast) or NumPyro's `Predictive`.


## Parameters


`rng_key: Array`  
PRNG key.

`fit: object`  
A fit result produced by [fit_svi()](functional.fit_svi.md#numpyro_forecast.functional.fit_svi) or [fit_mcmc()](functional.fit_mcmc.md#numpyro_forecast.functional.fit_mcmc).

`num_samples: int`  
Number of posterior draws.


## Returns


`dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading.


## Raises


`NotImplementedError`  
If `fit` is of an unsupported type.
