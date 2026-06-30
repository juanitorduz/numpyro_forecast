## forecaster.Forecaster


Fit a forecasting model with stochastic variational inference.


Usage

``` python
forecaster.Forecaster()
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
