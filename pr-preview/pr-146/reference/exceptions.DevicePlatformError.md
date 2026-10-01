## exceptions.DevicePlatformError


A `device` platform name has no initialized JAX backend.


Usage

``` python
exceptions.DevicePlatformError(message=None)
```


Raised by `_resolve_device()` (reached from every function with a `device` parameter) when `device` names a platform, e.g. `"tpu"`, whose backend is not initialized in this process. The message lists the available platforms and how to initialize the missing one via `jax_platforms`. `"host"`, `"cpu"`, `"numpy"` and `"pinned_host"` never raise this: they degrade to the NumPy path when the CPU backend is missing (see [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior)).
