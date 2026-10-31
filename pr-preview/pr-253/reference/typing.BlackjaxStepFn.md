## typing.BlackjaxStepFn


A blackjax step function `(rng_key, inner_state, logdensity_fn, params)`.


`typing.BlackjaxStepFn=Callable[…, object]`


Advances `inner_state` by one step and returns `(inner_state, info)`. It receives the log density of the current run and the `params` of a [BlackjaxBuildFn](typing.BlackjaxBuildFn.md#numpyro_forecast.typing.BlackjaxBuildFn) as arguments and must use those: a closure over the log density or the tuned values of a build call keeps the values of the first call. A closure over static configuration (for example `max_num_doublings`) is fine. Consumed by [BlackjaxCustomKernel](contrib.blackjax.BlackjaxCustomKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel).
