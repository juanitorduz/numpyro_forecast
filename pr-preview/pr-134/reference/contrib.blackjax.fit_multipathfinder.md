## contrib.blackjax.fit_multipathfinder()


Fit a forecasting model with multi-path BlackJAX Pathfinder and PSIS resampling.


Usage

``` python
contrib.blackjax.fit_multipathfinder(
    rng_key,
    model,
    data,
    covariates,
    *,
    num_paths=4,
    num_elbo_samples=200,
    maxiter=30,
    maxcor=10,
    maxls=1000,
    gtol=1e-08,
    ftol=1e-05,
    initial_positions=None
)
```


Runs `num_paths` independent single-path Pathfinder approximations in parallel (vmapped L-BFGS runs, by default each from its own fresh `init_to_uniform` starting point) and scores the pooled per-path draws with Pareto-smoothed importance sampling (PSIS), so the returned fit is not tied to a single mode. This is the recommended entry point over [fit_pathfinder()](contrib.blackjax.fit_pathfinder.md#numpyro_forecast.contrib.blackjax.fit_pathfinder): a single L-BFGS path can settle in one mode and its local normal approximation may not reflect the rest of a multimodal or otherwise hard posterior. The PSIS weights computed here are a fit-time diagnostic; the draws themselves come from [multipathfinder_samples()](contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples), which resamples fresh per-path draws and reads `pareto_k` to pick between PSIS and ELBO-weighted path sampling.

PRNG: `rng_key` is split into a model-initialization stream and a multipath-approximation stream; the initialization stream is further split into one subkey per path so that (when `initial_positions` is not supplied) every path starts from its own independent `init_to_uniform` draw, exactly the diverse starting points multipath Pathfinder wants.


## Parameters


`rng_key: Array`  
PRNG key for initialization and the multipath Pathfinder run.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`num_paths: int = ``4`  
Number of independent L-BFGS paths to run in parallel (vmapped).

`num_elbo_samples: int = ``200`  
Number of Monte Carlo samples drawn per path to estimate its ELBO and to build the pooled sample used for PSIS resampling.

`maxiter: int = ``30`  
Maximum number of L-BFGS iterations per path.

`maxcor: int = ``10`  
L-BFGS history size; caps the rank of the low-rank-plus-diagonal covariance correction at roughly `2 * maxcor`. High-dimensional posteriors need it raised well above the default of `10`.

`maxls: int = ``1000`  
Maximum number of line-search steps per L-BFGS iteration.

`gtol: float = ``1e-08`  
L-BFGS gradient-norm convergence tolerance.

`ftol: float = ``1e-05`  
L-BFGS relative function-value convergence tolerance.

`initial_positions: dict[str, Array] | None = None`  
Optional starting positions, one per path, overriding the default per-path `init_to_uniform` draws. Every leaf must already carry a leading axis of size `num_paths` (validated; a mismatch raises `ValueError`).


## Returns


`MultiPathfinderFit`  
The fitted multipath approximation together with its per-path ELBOs and the fit-time PSIS log weights/`pareto_k` diagnostic over the pooled draws.


## Raises


`ValueError`  
If `num_paths` is not positive, or `initial_positions` is supplied without a leading axis of size `num_paths` on every leaf.


## Warns


`UserWarning`  
If one or more per-path ELBOs are non-finite (raise `maxiter`/`maxcor` or inspect `fit.elbos`), or if the PSIS `pareto_k` diagnostic exceeds `0.7`, in which case `multipathfinder_samples(..., resample="auto")` falls back to ELBO-weighted path sampling (increase `num_paths`/`maxiter`/`maxcor`, or fall back to MCMC). Neither condition raises: the fit is always returned, carrying everything needed to inspect and diagnose it.


## Notes

Runs with `_stable_bfgs_sample()` patched into blackjax (via `_ensure_stable_bfgs_sample()`, called before `blackjax.vi.multipathfinder.multi_approximate`): `multi_approximate` calls `approximate`/`sample` imported from `blackjax.vi.pathfinder`, whose module globals hold `bfgs_sample`, so patching that module (as [fit_pathfinder()](contrib.blackjax.fit_pathfinder.md#numpyro_forecast.contrib.blackjax.fit_pathfinder) already does) covers the multipath route too.
