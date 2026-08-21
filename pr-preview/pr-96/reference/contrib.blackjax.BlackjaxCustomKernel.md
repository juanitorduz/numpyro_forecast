## contrib.blackjax.BlackjaxCustomKernel


Adapt an arbitrary BlackJAX sampler via a user-supplied `build_fn`.


Usage

``` python
contrib.blackjax.BlackjaxCustomKernel(
    model=None,
    *,
    build_fn,
)
```


The escape hatch for kernels without a dedicated wrapper. `build_fn` receives `(rng_key, logdensity_fn, position, num_warmup)` and must return `(inner_state, step_fn)`, where `inner_state` exposes `.position` over the same sites as the model and `step_fn` has signature `(rng_key, inner_state) -> (inner_state, info)`. The base class validates the returned state's key set and raises `TypeError` if it is malformed.


## Run Configuration

Pass this kernel to `~numpyro.infer.MCMC` with **`chain_method="sequential"` only**: the instance holds its step/postprocess functions as plain attributes (`self._step_fn`, `self._postprocess_fn`), and `vmap`/`pmap` chain parallelism (`"vectorized"`/`"parallel"`) traces this instance, capturing those attributes as tracers instead of running the closed-over blackjax step. Also pass **`num_warmup=0`**: whatever adaptation `build_fn` performs runs once inside `~_BlackjaxKernel.init()`, so any NumPyro-driven warmup steps on top of that are simply discarded work, not additional tuning.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`build_fn: BlackjaxBuildFn`  
The blackjax build callable described above.
