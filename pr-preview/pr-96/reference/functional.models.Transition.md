## functional.models.Transition


(carry, x_t) -\> (dist_t, carry_fn) where carry_fn(z_t) builds the next carry


`functional.models.Transition=Callable[[Any, Array | None], tuple[dist.Distribution, Callable[[Array], Any]]]`


from the *sampled* latent. The wrapper owns the sample statement.

`carry` is an arbitrary PyTree (hence `Any`): typing it as `object` would, by function-parameter contravariance, reject every concretely-typed transition a user might write (e.g. `carry: Array`).
