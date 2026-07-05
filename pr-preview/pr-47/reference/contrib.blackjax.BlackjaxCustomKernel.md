## contrib.blackjax.BlackjaxCustomKernel


Adapt an arbitrary BlackJAX sampler via a user-supplied `build_fn`.


Usage

``` python
contrib.blackjax.BlackjaxCustomKernel()
```


The escape hatch for kernels without a dedicated wrapper. `build_fn` receives `(rng_key, logdensity_fn, position, num_warmup)` and must return `(inner_state, step_fn)`, where `inner_state` exposes `.position` over the same sites as the model and `step_fn` has signature `(rng_key, inner_state) -> (inner_state, info)`. The base class validates the returned state's key set and raises `TypeError` if it is malformed.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`build_fn: BuildFn`  
The blackjax build callable described above.
