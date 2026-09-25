## contrib.blackjax.fit_pathfinder()


Fit a forecasting model with BlackJAX Pathfinder variational inference.


Usage

``` python
contrib.blackjax.fit_pathfinder(
    rng_key,
    model,
    data,
    covariates,
    *,
    num_elbo_samples=200,
    ftol=1e-05,
    maxiter=30,
    maxcor=10,
    maxls=1000,
    gtol=1e-08,
    init_params=None
)
```


PRNG: `rng_key` is split into a model-initialization stream and a Pathfinder-approximation stream.


## Parameters


`rng_key: Array`  
PRNG key for initialization and the Pathfinder run.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`num_elbo_samples: int = ``200`  
Number of Monte Carlo samples used to estimate the ELBO along the L-BFGS optimization path.

`ftol: float = ``1e-05`  
L-BFGS relative function-value tolerance (convergence criterion).

`maxiter: int = ``30`  
Maximum number of L-BFGS iterations (blackjax's default is `30`). Models with per-step time latents typically need far more: the default suits tens of parameters, while a few hundred parameters (e.g. a 400-step random walk) converge around `1_000`.

`maxcor: int = ``10`  
L-BFGS history size; caps the rank of the low-rank-plus-diagonal covariance correction at roughly `2 * maxcor`. High-dimensional posteriors need it raised well above the default of `10`.

`maxls: int = ``1000`  
Maximum number of line-search steps per L-BFGS iteration.

`gtol: float = ``1e-08`  
L-BFGS gradient-norm convergence tolerance.

`init_params: dict[str, Array] | None = None`  
Optional initial unconstrained position, overriding NumPyro's own initialization draw (`param_info.z`); `initialize_model` still runs regardless, since it also builds the potential/postprocess functions.


## Returns


`PathfinderFit`  
The fitted variational approximation.


## See Also

[fit_multipathfinder()](contrib.blackjax.fit_multipathfinder.md#numpyro_forecast.contrib.blackjax.fit_multipathfinder) The recommended entry point  
runs several L-BFGS paths in parallel and combines them (see [multipathfinder_samples()](contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples)), instead of relying on this single path's local normal approximation.


## Notes

Runs with `_stable_bfgs_sample()` patched into blackjax (via `_ensure_stable_bfgs_sample()`): upstream's `bfgs_sample` underflows its log-determinant beyond a few hundred parameters, which floors every path ELBO at `-inf` and makes the returned state garbage.
