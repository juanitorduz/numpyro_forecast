## typing.BlackjaxBuildFn


A blackjax sampler build function `(rng_key, logdensity_fn, position, num_warmup)`.


`typing.BlackjaxBuildFn=Callable[…, object]`


Runs the adaptation and returns `(inner_state, params)`, with `params` a pytree of numeric arrays holding the tuned parameters (static Python `int` settings belong in the [BlackjaxStepFn](typing.BlackjaxStepFn.md#numpyro_forecast.typing.BlackjaxStepFn)). `rng_key` is first, matching the package's rng-key-first convention. Consumed by [BlackjaxCustomKernel](contrib.blackjax.BlackjaxCustomKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel).
