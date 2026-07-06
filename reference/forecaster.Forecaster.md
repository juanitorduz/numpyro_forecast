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

`guide: GuideLike = None`  
Guide specification resolved by `~numpyro_forecast.functional.resolve_guide()`: `None` (`AutoNormal`), an `AutoGuide` instance, an `AutoGuide` subclass or `functools.partial` factory of one, or a hand-written guide function.

`optim: OptimizerLike = None`  
Optimizer specification resolved by `~numpyro_forecast.functional.resolve_optimizer()`: `None` (`Adam(0.01)`), a positive scalar learning rate, an `optax.GradientTransformation`, or a `_NumPyroOptim`.

`num_steps: int = ``1001`  
Number of SVI steps.

`num_particles: int = ``1`  
Number of ELBO particles.

`progress_bar: bool = ``False`  
Whether to display the SVI progress bar.

`stable_update: bool = ``False`  
Whether SVI skips non-finite parameter updates.
