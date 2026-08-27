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

Pass this kernel to `~numpyro.infer.MCMC` with **`chain_method="sequential"` only**. With more than one chain, `chain_method="vectorized"` hands `~_BlackjaxKernel.init()` a stacked `rng_key` of shape `(num_chains, 2)` rather than vmapping a per-chain call, and the `jax.random.split` inside `init` rejects a non-scalar key with a `ValueError` (pinned by the test suite). `chain_method="parallel"` is likewise unsupported: it is not exercised by this package and shares the same single-key assumption. Also pass **`num_warmup=0`**: whatever adaptation `build_fn` performs runs once inside `~_BlackjaxKernel.init()`, so any NumPyro-driven warmup steps on top of that are simply discarded work, not additional tuning.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`build_fn: BlackjaxBuildFn`  
The blackjax build callable described above.
