## exceptions.HostMemoryKindError


A device exposes no host memory kind for `device="pinned_host"`.


Usage

``` python
exceptions.HostMemoryKindError(message=None)
```


Raised by `_host_memory_kind()`, reached from [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior), the predictive drivers and everything built on them, when an explicit `device="pinned_host"` targets a device whose addressable memories include neither `"pinned_host"` nor `"unpinned_host"`. The message lists the kinds the device does expose; pass a `jax.Device` or a platform name (for example `device="cpu"`), or `device="host"`, instead.
