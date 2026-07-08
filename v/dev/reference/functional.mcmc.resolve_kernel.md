## functional.mcmc.resolve_kernel()


Normalize a kernel specification.


Usage

``` python
functional.mcmc.resolve_kernel(
    kernel,
    model,
    kernel_kwargs,
)
```


`None` -\> `NUTS(model, **kernel_kwargs)` (kwargs tune the default kernel without naming it); a kernel class -\> `kernel(model, **kernel_kwargs)`; a kernel instance -\> returned unchanged, and combining an instance with non-empty `kernel_kwargs` raises `ValueError` (ambiguous). Anything else -\> `TypeError`.


## Parameters


`kernel: KernelLike`  
The kernel specification (see `~numpyro_forecast.typing.KernelLike`).

`model: ForecastModel`  
The model the kernel is built against.

`kernel_kwargs: Mapping[str, Any] | None`  
Extra keyword arguments forwarded to the kernel constructor (ignored, and rejected, for an already-constructed instance).


## Returns


`MCMCKernel`  
The resolved kernel.


## Raises


`KernelConfigError`  
If a kernel instance is combined with non-empty `kernel_kwargs`.

`KernelResolutionError`  
If `kernel` is neither `None`, an `MCMCKernel` subclass, nor an `MCMCKernel` instance.
