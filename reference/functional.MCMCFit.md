## functional.MCMCFit


The result of fitting a forecasting model with MCMC.


Usage

``` python
functional.MCMCFit(
    samples,
    num_chains=1,
)
```


## Parameter Attributes


`samples: dict[str, Array]`  

`num_chains: int = ``1`  


## Attributes


`samples: dict[str, Array]`  
The posterior samples of the latent sites, sample axis leading and stored flattened (`group_by_chain=False`).

`num_chains: int`  
The number of chains the fit was run with (frozen; default `1`), so chain structure survives into `~numpyro_forecast.convert.to_datatree()`.
