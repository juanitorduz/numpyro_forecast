## typing.BlackjaxBuildFn


A blackjax sampler build function `(rng_key, logdensity_fn, position, num_warmup)`.


`typing.BlackjaxBuildFn=Callable[…, object]`


Returns `(inner_state, step_fn)`. `rng_key` is first, matching the package's rng-key-first convention. Consumed by [BlackjaxCustomKernel](contrib.blackjax.BlackjaxCustomKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel).
