## exceptions.DeviceMemoryError


A memory pool ran out during posterior or predictive sampling.


Usage

``` python
exceptions.DeviceMemoryError(message=None)
```


Raised by `~numpyro_forecast.predictive.draw_posterior()` and the predictive drivers (`~numpyro_forecast.predictive.forecast()`, `~numpyro_forecast.predictive.predict_in_sample()`, and everything built on them, e.g. `~numpyro_forecast.convert.to_datatree()`) when XLA reports `RESOURCE_EXHAUSTED`. For an accelerator OOM the message embeds the device's memory budget and the lever: the per-chunk footprint scales linearly with `batch_size` times the panel width, so lower (or set) `batch_size`, free large device arrays still referenced elsewhere, and keep results off the accelerator with `device="host"`. For a pinned host pool OOM (`Out of host memory`, reached only through an explicit `device="pinned_host"` or the caller's own pinned arrays) it names the pool's cap (`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`) and points at `device="host"`, which lands results in pageable host memory, instead. The original XLA error is chained as `__cause__`.
