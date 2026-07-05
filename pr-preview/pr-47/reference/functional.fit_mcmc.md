## functional.fit_mcmc()


Fit a forecasting model with MCMC.


Usage

``` python
functional.fit_mcmc(
    rng_key,
    model,
    data,
    covariates,
    *,
    kernel=None,
    kernel_kwargs=None,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    chain_method="sequential",
    progress_bar=False
)
```


PRNG: consumed entirely by `~numpyro.infer.MCMC`.


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`kernel: KernelLike = None`  
Kernel specification resolved by [resolve_kernel()](functional.resolve_kernel.md#numpyro_forecast.functional.resolve_kernel): `None` (`NUTS`), an `MCMCKernel` instance, or an `MCMCKernel` subclass.

`kernel_kwargs: Mapping[str, Any] | None = None`  
Extra keyword arguments for the kernel constructor (only with `None` or a kernel class; rejected with an instance).

`num_warmup: int = ``1000`  
Number of warmup steps.

`num_samples: int = ``1000`  
Number of posterior samples.

`num_chains: int = ``1`  
Number of MCMC chains (stored on the returned [MCMCFit](functional.MCMCFit.md#numpyro_forecast.functional.MCMCFit)).

`chain_method: str = ``"sequential"`  
NumPyro chain method (`"sequential"`/`"parallel"`/`"vectorized"`).

`progress_bar: bool = ``False`  
Whether to display the MCMC progress bar.


## Returns


`MCMCFit`  
The posterior samples (flattened) and `num_chains`.


## Raises


`ValueError`  
If `data` and `covariates` have different durations.

`KernelConfigError`  
If a run-config constraint is violated (see `_validate_kernel_run_config()`).
