## typing.BlackjaxStepFn


A blackjax step function `(rng_key, inner_state, logdensity_fn, params)`.


`typing.BlackjaxStepFn=Callable[…, object]`


Advances `inner_state` by one step and returns `(inner_state, info)`. It receives the log density of the current run and the `params` of a [BlackjaxBuildFn](typing.BlackjaxBuildFn.md#numpyro_forecast.typing.BlackjaxBuildFn) as arguments and must not close over them. Consumed by [BlackjaxCustomKernel](contrib.blackjax.BlackjaxCustomKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel).
