## contrib.blackjax.pathfinder_samples()


Draw `num_samples` posterior samples from a fitted Pathfinder approximation.


Usage

``` python
contrib.blackjax.pathfinder_samples(
    rng_key, fit, num_samples, *, batch_size=None, device=None
)
```


The returned dict has the sample axis leading and is ready to pass to [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) or NumPyro's `Predictive`, exactly like [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior). Chunking and device offload are delegated to the shared `_draw_chunked()` driver, so the same memory-bounding contract applies.

PRNG: within each chunk (the whole draw, when unchunked), the chunk key is split into a model-initialization stream and a Pathfinder-sampling stream (the init param draws are unused, but the streams are kept isolated).


## Parameters


`rng_key: Array`  
PRNG key.

`fit: PathfinderFit`  
A fit from [fit_pathfinder()](contrib.blackjax.fit_pathfinder.md#numpyro_forecast.contrib.blackjax.fit_pathfinder).

`num_samples: int`  
Number of posterior draws.

`batch_size: int | None = None`  
Optional chunk size for the drawing itself; see [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) (the same memory/reproducibility contract applies).

`device: jax.Device | str | None = None`  
Where each chunk of draws is moved as soon as it is drawn; see [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior), including for the NumPy path taken when the JAX CPU backend is not initialized and for the rules on mixing a host-committed result with other arrays.


## Returns


`dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading (with `device="host"`: leaves committed to the CPU device, or NumPy arrays when no CPU backend is initialized).


## Raises


`ValueError`  
If `num_samples` or `batch_size` is not positive.

`HostMemoryKindError`  
If `device="pinned_host"` is requested on a device that exposes no host memory kind (see `_host_memory_kind()`).

`DevicePlatformError`  
If `device` names a platform whose backend is not initialized (see `_resolve_device()`).


## Warns


`UserWarning`  
If `device="cpu"` is requested and the JAX CPU backend is not initialized, so the draws take the NumPy path of `"host"` instead.
