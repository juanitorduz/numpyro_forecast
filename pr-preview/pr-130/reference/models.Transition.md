## models.Transition


`(carry, x_t) -> (dist_t, carry_fn)` where `carry_fn(z_t)` builds the next


`type`` models.Transition[Carry] = Callable[`\
`    [Carry, PyTree[Array] | None], tuple[dist.Distribution, Callable[[Array], Carry]]`\
`]`


carry from the *sampled* latent. The wrapper owns the sample statement.

`Carry` is the user's carry type (any PyTree), bound per [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series) call; `x_t` is one row of the `xs` PyTree (`None` for autonomous dynamics).
