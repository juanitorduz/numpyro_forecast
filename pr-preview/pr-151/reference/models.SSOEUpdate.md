## models.SSOEUpdate


`(carry, y_t, eps_t, x_t) -> carry`: the next carry from the row's value and


`type`` models.SSOEUpdate[Carry] = Callable[[Carry, Array, Array, PyTree[Array] | None], Carry]`


error. In-sample `eps_t = y_t - mu_t`; over the horizon `eps_t` is the drawn error and `y_t = mu_t + eps_t`. An update that needs the mean calls `mean(carry, x_t)` again (bit-identical, computed once by XLA) rather than reconstructing it as `y_t - eps_t`, which can differ by an ulp. Must preserve the carry's tree structure, shapes and dtypes. Like [mean](typing.Array.md#numpyro_forecast.typing.Array.mean), it must not call `numpyro.sample`.
