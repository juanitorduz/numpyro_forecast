## forecaster.HMCForecaster


Fit a forecasting model with MCMC (NUTS by default).


Usage

``` python
forecaster.HMCForecaster()
```


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model to fit (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`kernel: KernelLike = None`  
Kernel specification resolved by `~numpyro_forecast.functional.mcmc.resolve_kernel()`: `None` (`NUTS`), an `MCMCKernel` instance, or an `MCMCKernel` subclass.

`kernel_kwargs: Mapping[str, Any] | None = None`  
Extra keyword arguments for the kernel constructor (only with `None` or a kernel class).

`num_warmup: int = ``1000`  
Number of warmup steps.

`num_samples: int = ``1000`  
Number of posterior samples.

`num_chains: int = ``1`  
Number of MCMC chains.

`chain_method: str = ``"sequential"`  
NumPyro chain method (`"sequential"`/`"parallel"`/`"vectorized"`).

`progress_bar: bool = ``False`  
Whether to display the MCMC progress bar.
