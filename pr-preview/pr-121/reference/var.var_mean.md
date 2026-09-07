## var.var_mean()


Compute the VAR conditional mean of the next row from a lag window.


Usage

``` python
var.var_mean(
    phi,
    lags,
    intercept=None,
)
```


 \mu_t = c + \sum\_{l=1}^{p} \Phi_l \\ y\_{t-l}, 

with `phi[..., l - 1, :, :]` holding \Phi_l and `lags[..., -l, :]` holding y\_{t-l} (most recent row last).


## Parameters


`phi: Float[Array, ``" *#batch lags obs obs"]`  
Coefficient tensor `(*batch, lags, obs, obs)`; see the module docstring for the index convention.

`lags: Float[Array, ``" *#batch lags obs"]`  
Lag window `(*batch, lags, obs)` in natural time order.

`intercept: Float[Array, ``" *#batch obs"] | None`` = None`  
Optional intercept c of shape `(*batch, obs)`; `None` for a zero-mean recursion (the impulse response case).


## Returns


`Float[Array, ``"*batch obs"]`  
The conditional mean row, broadcast over the batch axes of the inputs.


## Examples

A latent VAR under [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series): the transition returns the next-row distribution and the window update, with `phi` and the Cholesky factor `scale_tril` sampled outside:

``` python
def transition(carry, _):
    dist_t = dist.MultivariateNormal(var_mean(phi, carry), scale_tril=scale_tril)
    return dist_t, lambda z: jnp.concatenate([carry[..., 1:, :], z[..., None, :]], axis=-2)


z = markov_series(h, "z", init_carry=jnp.zeros((n_lags, n_obs)), transition=transition)
```
