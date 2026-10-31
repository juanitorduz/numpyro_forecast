## contrib.blackjax.BlackjaxCustomKernel


Adapt an arbitrary BlackJAX sampler via a user-supplied `build_fn` and `step_fn`.


Usage

``` python
contrib.blackjax.BlackjaxCustomKernel(
    model=None,
    *,
    build_fn,
    step_fn,
)
```


The escape hatch for kernels without a dedicated wrapper. `build_fn` receives `(rng_key, logdensity_fn, position, num_warmup)`, runs the adaptation, and must return `(inner_state, params)`, where `inner_state` exposes `.position` over the same sites as the model and `params` is a pytree of numeric arrays with the tuned parameters (use `None` for a sampler without tuning). `step_fn` receives `(rng_key, inner_state, logdensity_fn, params)` and must return `(inner_state, info)`. The base class validates the returned state's key set and the `params` leaves, and raises `TypeError` if either is malformed. The sampling loop traces `params`, so a static setting that blackjax needs as a Python `int` (for example `max_num_doublings` of NUTS) goes in `step_fn`, not in `params`: `TypeError` reports a Python `int` or `bool` leaf.

`step_fn` gets the log density and the tuned parameters as arguments in every step, and must not close over the `logdensity_fn` or the tuned values of a `build_fn` call: NumPyro compiles the sampling loop once and reuses it for all chains (and re-runs with `jit_model_args=True`), so a closure would keep the values of the first call. For BlackJAX NUTS with window adaptation:

``` python
import blackjax


def build_fn(rng_key, logdensity_fn, position, num_warmup):
    adapt = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    result, _ = adapt.run(rng_key, position, num_steps=500)
    return result.state, result.parameters


def step_fn(rng_key, inner_state, logdensity_fn, params):
    return blackjax.nuts(logdensity_fn, **params).step(rng_key, inner_state)


kernel = BlackjaxCustomKernel(model, build_fn=build_fn, step_fn=step_fn)
```


## Run Configuration

Pass this kernel to `numpyro.infer.MCMC` with **`chain_method="sequential"` only**. With more than one chain, `chain_method="vectorized"` hands `_BlackjaxKernel.init()` a stacked `rng_key` of shape `(num_chains, 2)` rather than vmapping a per-chain call, and the `jax.random.split` inside `init` rejects a non-scalar key with a `ValueError` (pinned by the test suite). `chain_method="parallel"` is likewise unsupported: it is not exercised by this package and shares the same single-key assumption. Each chain of a sequential run calls `_BlackjaxKernel.init()` and so `build_fn`. The `params` it returns are part of the sampler state and `_BlackjaxKernel.sample()` builds the log density from the model arguments of the current run, so the chains are independent runs and a second `mcmc.run` on new data (also with `jit_model_args=True`) samples the new posterior. A run that continues from `mcmc.post_warmup_state` keeps the blackjax state of the earlier data, so leave `post_warmup_state` unset for a run on new data. Also pass **`num_warmup=0`**: whatever adaptation `build_fn` performs runs once inside `_BlackjaxKernel.init()`, so any NumPyro-driven warmup steps on top of that are simply discarded work, not additional tuning.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`build_fn: BlackjaxBuildFn`  
The blackjax build callable described above.

`step_fn: BlackjaxStepFn`  
The blackjax step callable described above.
