## models.Transition


`(carry, x_t) -> dist_t`: the distribution of the next latent given the carry.


`type`` models.Transition[Carry] = Callable[[Carry, PyTree[Array] | None], dist.Distribution]`


The wrapper owns the sample statement; see [Advance](models.Advance.md#numpyro_forecast.models.Advance) for the carry update.

`Carry` is the user's carry type (any PyTree), bound per [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series) call; `x_t` is one row of the `xs` PyTree (`None` for autonomous dynamics).
