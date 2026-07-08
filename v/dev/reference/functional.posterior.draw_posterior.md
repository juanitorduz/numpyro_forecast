## functional.posterior.draw_posterior()


Draw `num_samples` posterior samples of the latent sites from a fit.


Usage

``` python
functional.posterior.draw_posterior(
    rng_key,
    fit,
    num_samples,
)
```


Dispatches on the fit type (e.g. `~numpyro_forecast.functional.svi.SVIFit`, `~numpyro_forecast.functional.mcmc.MCMCFit`). The returned dict has the sample axis leading and is ready to pass to `~numpyro_forecast.functional.prediction.forecast()` or NumPyro's `Predictive`.


## Parameters


`rng_key: Array`  
PRNG key.

`fit: object`  
A fit result produced by `~numpyro_forecast.functional.svi.fit_svi()` or `~numpyro_forecast.functional.mcmc.fit_mcmc()`.

`num_samples: int`  
Number of posterior draws.


## Returns


`dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading.


## Raises


`NotImplementedError`  
If `fit` is of an unsupported type.

`GuideSampleArgsError`  
If the fit holds a hand-written guide but was constructed without its in-sample covariates/data.


## Notes

For an `~numpyro_forecast.functional.mcmc.MCMCFit`, when `num_samples` does not exceed the number of draws in the chain the draws are thinned on an evenly spaced grid (no duplicates); only when more samples are requested than the chain holds are they resampled with replacement. For an `~numpyro_forecast.functional.svi.SVIFit` the draws are sampled afresh from the fitted guide.
