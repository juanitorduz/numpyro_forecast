## contrib.dynestyx.StateSpaceResult


Latent state draws of one [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) call, `None` outside their mode.


Usage

``` python
contrib.dynestyx.StateSpaceResult()
```


The block registers the `"forecast"` and `"obs"` sites itself; the states are returned so that a model can register its latent level (for example `numpyro.deterministic("level", result.x_in_sample)` when `h.data is None`) under any handler that serves the mode.


## Attributes


`x_in_sample: Float[Array, ``" time state"] | None`  
One draw of the in-sample states, shape `(t_obs, state)`, when the model is called without data: the draw behind the `"obs"` site (per-step smoothing marginals under a `Smoother`, the posterior path under a `LatentPathBuilder`). `None` otherwise.

`x_future: Float[Array, ``" future state"] | None`  
The Simulator's state rollout over the horizon, shape `(future, state)`, when forecasting; `None` otherwise.
