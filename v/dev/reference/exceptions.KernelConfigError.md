## exceptions.KernelConfigError


A kernel is combined with an invalid configuration.


Usage

``` python
exceptions.KernelConfigError(message=None)
```


Raised by `~numpyro_forecast.functional.mcmc.resolve_kernel()` when `kernel_kwargs` accompany an already-constructed instance, and by `~numpyro_forecast.functional.mcmc.fit_mcmc()` for run-config constraints (ensemble samplers need multiple vectorized chains; BlackJAX kernels need sequential chains).
