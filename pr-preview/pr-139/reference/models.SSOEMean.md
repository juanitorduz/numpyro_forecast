## models.SSOEMean


`(carry, x_t) -> mu_t`: the one-step-ahead mean of the current row, shape


`type`` models.SSOEMean[Carry] = Callable[[Carry, PyTree[Array] | None], Float[Array, ``" *batch obs"]]`


`(*batch, obs)` (a scalar state emits `mu[None]`). [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) owns the error site: `mean` must not call `numpyro.sample` (that is [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series)).

`Carry` is the user's carry type (any PyTree), bound per [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) call; `x_t` is one row of the `xs` PyTree (`None` when `xs` is `None`).
