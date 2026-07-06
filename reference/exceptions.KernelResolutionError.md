## exceptions.KernelResolutionError


A kernel specification could not be resolved.


Usage

``` python
exceptions.KernelResolutionError(message=None)
```


Raised by `~numpyro_forecast.functional.resolve_kernel()` for a type that is neither `None`, an `MCMCKernel` subclass, nor an `MCMCKernel` instance.
