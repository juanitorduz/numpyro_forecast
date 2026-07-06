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
    progress_bar=False,
    stable_update=False
)
```


PRNG: consumes `rng_key` once for the SVI run; nothing is retained.


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`guide: GuideLike = None`  
Guide specification resolved by [resolve_guide()](functional.resolve_guide.md#numpyro_forecast.functional.resolve_guide): `None` (`AutoNormal`), an `AutoGuide` instance, an `AutoGuide` subclass or `functools.partial` factory of one, or a hand-written guide function.

`optim: OptimizerLike = None`  
Optimizer specification resolved by [resolve_optimizer()](functional.resolve_optimizer.md#numpyro_forecast.functional.resolve_optimizer): `None` (`Adam(0.01)`), a positive scalar learning rate, an `optax.GradientTransformation`, or a `_NumPyroOptim`. For example, a cosine-decayed, gradient-clipped Adam:

``` python
import optax
```

    schedule = optax.cosine_decay_schedule(1e-2, decay_steps=1_000)
    optim = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))

`num_steps: int = ``1001`  
Number of SVI steps.

`num_particles: int = ``1`  
Number of ELBO particles.

`progress_bar: bool = ``False`  
Whether to display the SVI progress bar.

`stable_update: bool = ``False`  
Whether SVI skips parameter updates whose new value is non-finite (NumPyro's `stable_update`).


## Returns


`SVIFit`  
The fitted guide, variational parameters, loss history, and the in-sample `data`/`covariates` (kept by identity, not copied).


## Raises


`ValueError`  
If `data` and `covariates` have different durations.
