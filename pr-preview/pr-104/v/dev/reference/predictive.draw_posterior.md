## predictive.draw_posterior()


Draw `num_samples` posterior samples of the latent sites from a fitted guide.


Usage

``` python
predictive.draw_posterior(
    rng_key, guide, params, num_samples, *, batch_size=None, device=None
)
```


The returned dict has the sample axis leading and is ready to pass to [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) or NumPyro's `Predictive`. An `AutoDelta` guide is a MAP point estimate: it is drawn once and tiled to `num_samples` (`_ensure_sample_axis_for_delta()`), since it carries no posterior spread of its own. Every other `AutoGuide` is sampled through a jitted, per-guide-cached `sample_posterior` (`_jitted_sample_posterior()`).


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
Where each chunk of draws is moved as soon as it is drawn. `"host"` keeps every leaf in pageable host memory, so nothing of the result occupies accelerator memory: with the JAX CPU backend initialized it commits each leaf to `jax.devices("cpu")[0]` and returns committed `jax.Array` leaves (`np.asarray` on one is a zero-copy view); without it (for example after `numpyro.set_platform("cuda")`, or a `JAX_PLATFORMS` preset) it copies each chunk with `jax.device_get()` and returns NumPy arrays, since a CUDA client offers no pageable `jax.Array` container. It therefore needs no CPU backend and never pins memory. `"numpy"` forces the NumPy path; `"pinned_host"` commits to the accelerator's pinned host memory kind instead, a pool capped by `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` (64 GB by default on CUDA), so prefer `"host"` for large panels. A `jax.Device` or platform name like `"cpu"` commits the draws to that device (`"cpu"` warns and takes the NumPy path when the CPU backend is missing). `None` keeps everything on the default device. `device` never changes the draw values. A host-committed jax result is not a drop-in replacement for a device array in your own `jnp` code: mixed with an uncommitted array an op runs on the CPU and returns a CPU-committed array, mixed with an accelerator-committed array it raises, and a pinned array raises on any mix. Feed a host posterior (in either container) straight into [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample()](predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample), or `~numpyro_forecast.convert.to_datatree()` (all of which already accept it), or convert explicitly first with `np.asarray(x)` (stays on host) or `jax.device_put(x, device)` (moves it to an accelerator).


## Returns


`dict[str, Array | np.ndarray]`  
Posterior samples of the latent sites, sample axis leading. With `device="host"` the leaves are committed to the CPU device, or NumPy arrays when no CPU backend is initialized.


## Raises


`ValueError`  
If `num_samples` or `batch_size` is not positive.

`RuntimeError`  
If `device="pinned_host"` is requested on a device that exposes no host memory kind (see `~numpyro_forecast._offload._host_memory_kind()`).


## Warns


`UserWarning`  
If `device="cpu"` is requested and the JAX CPU backend is not initialized, so the draws take the NumPy path of `"host"` instead.


## Notes

For an MCMC fit, use its samples directly (`mcmc.get_samples()`); this function draws afresh from a variational guide, and chunks drawn from independent subkeys remain valid i.i.d. posterior samples.
