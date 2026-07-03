## functional.fit_svi()


Fit a forecasting model with stochastic variational inference.


Usage

``` python
functional.fit_svi(
    rng_key,
    model,
    data,
    covariates,
    *,
    guide=None,
    optim=None,
    num_steps=1001,
    num_particles=1,
    progress_bar=False
)
```


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`guide: AutoGuide | None = None`  
Variational guide; defaults to `AutoNormal(model)`.

`optim: _NumPyroOptim | None = None`  
NumPyro optimizer; defaults to `Adam(0.01)`.

`num_steps: int = ``1001`  
Number of SVI steps.

`num_particles: int = ``1`  
Number of ELBO particles.

`progress_bar: bool = ``False`  
Whether to display the SVI progress bar.


## Returns


`SVIFit`  
The fitted guide, variational parameters, and loss history.


## Raises


`ValueError`  
If `data` and `covariates` have different durations.
