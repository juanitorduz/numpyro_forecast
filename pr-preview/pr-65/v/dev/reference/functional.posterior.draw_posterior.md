## functional.posterior.draw_posterior()


Draw `num_samples` posterior samples of the latent sites from a fit.


Usage

``` python
functional.posterior.draw_posterior(
    rng_key, fit, num_samples, *, batch_size=None, device=None
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

`batch_size: int | None = None`  
Optional chunk size for the drawing itself. Sampling a variational posterior materializes every latent and deterministic site for all draws at once, which on a wide panel is the largest allocation of the whole workflow. With `batch_size` set (strictly below `num_samples`), the draws are sampled in chunks of exactly this many samples, each chunk is moved per `device` before the next is drawn, and the final chunk's overdraw is discarded, so accelerator memory is bounded by one chunk. Chunking changes the PRNG stream layout: draws are reproducible per `(rng_key, batch_size)`. Ignored for `~numpyro_forecast.functional.mcmc.MCMCFit` (see Notes).

`device: jax.Device | str | None = None`  
Where each chunk of draws is moved as soon as it is drawn. `"host"` copies to host memory with `jax.device_get()` and returns NumPy leaves; it needs no CPU backend, so it works even when `numpyro.set_platform("cuda")` (or `jax_platforms`) leaves only an accelerator backend initialized. A `jax.Device` or platform name like `"cpu"` commits the draws to that device instead (`"cpu"` falls back to `"host"` with a `UserWarning` when the CPU backend is not initialized). `None` keeps everything on the default device. `device` never changes the draw values.


## Returns


`dict[str, Array | np.ndarray]`  
Posterior samples of the latent sites, sample axis leading (NumPy leaves when `device` resolves to `"host"`).


## Raises


`NotImplementedError`  
If `fit` is of an unsupported type.

`GuideSampleArgsError`  
If the fit holds a hand-written guide but was constructed without its in-sample covariates/data.

`ValueError`  
If `batch_size` is not positive.


## Notes

For an `~numpyro_forecast.functional.mcmc.MCMCFit`, when `num_samples` does not exceed the number of draws in the chain the draws are thinned on an evenly spaced grid (no duplicates); only when more samples are requested than the chain holds are they resampled with replacement. Because that selection is deterministic and the chain draws are already materialized, `batch_size` is ignored for MCMC fits (there is no drawing peak to bound and chunked selection would duplicate draws); `device` still applies. For an `~numpyro_forecast.functional.svi.SVIFit` (or a Pathfinder fit) the draws are sampled afresh from the fitted guide, and chunks drawn from independent subkeys remain valid i.i.d. posterior samples.
