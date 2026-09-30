## models.Advance


`(carry, z_t, x_t) -> carry`: the next carry from the current carry, the


`type`` models.Advance[Carry] = Callable[[Carry, Array, PyTree[Array] | None], Carry]`


*sampled* latent `z_t` and the exogenous row. Omit it when the carry is the latent itself (the AR(1) case); a lag window keeps the last `p` samples.
