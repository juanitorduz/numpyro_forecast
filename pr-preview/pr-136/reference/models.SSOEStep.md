## models.SSOEStep


`(carry, x_t) -> (mu_t, carry_fn)` where `mu_t` is the one-step-ahead mean


`type`` models.SSOEStep[Carry] = Callable[`\
`    [Carry, PyTree[Array] | None],`\
`    tuple[Float[Array, `<span class="st">`" *batch obs"], Callable[[Array, Array], Carry]],`\
`]`</span>


of the current row (shape `(*batch, obs)`) and `carry_fn(y_t, eps_t)` builds the next carry from the row's value and error. [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) owns the error site: `step` must not call `numpyro.sample` (that is [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series)).

`Carry` is the user's carry type (any PyTree), bound per [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) call; `x_t` is one row of the `xs` PyTree (`None` when `xs` is `None`).
