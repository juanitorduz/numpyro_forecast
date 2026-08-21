## functional.posterior.draw_posterior()


Draw `num_samples` posterior samples of the latent sites from a fitted guide.


Usage

``` python
functional.posterior.draw_posterior(
    rng_key, guide, params, num_samples, *, batch_size=None, device=None
)
```


The returned dict has the sample axis leading and is ready to pass to `~numpyro_forecast.functional.prediction.forecast()` or NumPyro's `Predictive`. An `AutoDelta` guide is a MAP point estimate: it is drawn once and tiled to `num_samples` (`_ensure_sample_axis_for_delta()`), since it carries no posterior spread of its own. Every other `AutoGuide` is sampled through a jitted, per-guide-cached `sample_posterior` (`_jitted_sample_posterior()`).


## Parameters


`rng_key: Array`  
PRNG key.

`guide: AutoGuide`  
The fitted variational guide, e.g. the `AutoGuide` instance passed to `SVI`.

`params: dict[str, Array]`  
The learned variational parameters, e.g. the trained parameters from `svi.run`'s result.

`num_samples: int`  
Number of posterior draws.

`batch_size: int | None = None`  
Optional chunk size for the drawing itself. Sampling a variational posterior materializes every latent and deterministic site for all draws at once, which on a wide panel is the largest allocation of the whole workflow. With `batch_size` set (strictly below `num_samples`), the draws are sampled in chunks of exactly this many samples, each chunk is moved per `device` before the next is drawn, and the final chunk's overdraw is discarded, so accelerator memory is bounded by one chunk. Chunking changes the PRNG stream layout: draws are reproducible per `(rng_key, batch_size)`.

`device: jax.Device | str | None = None`  
Where each chunk of draws is moved as soon as it is drawn. `"host"` copies to host memory with `jax.device_get()` and returns NumPy leaves; it needs no CPU backend, so it works even when `numpyro.set_platform("cuda")` (or `jax_platforms`) leaves only an accelerator backend initialized. A `jax.Device` or platform name like `"cpu"` commits the draws to that device instead (`"cpu"` falls back to `"host"` with a `UserWarning` when the CPU backend is not initialized). `None` keeps everything on the default device. `device` never changes the draw values.


## Returns


`dict[str, Array | np.ndarray]`  
Posterior samples of the latent sites, sample axis leading (NumPy leaves when `device` resolves to `"host"`).


## Raises


`ValueError`  
If `num_samples` or `batch_size` is not positive.


## Notes

For an MCMC fit, use its samples directly (`mcmc.get_samples()`); this function draws afresh from a variational guide, and chunks drawn from independent subkeys remain valid i.i.d. posterior samples.
