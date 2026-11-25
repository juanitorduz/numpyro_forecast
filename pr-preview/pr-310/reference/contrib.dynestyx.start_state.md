## contrib.dynestyx.start_state()


Read the filtered state at the end of the earlier window inside a model.


Usage

``` python
contrib.dynestyx.start_state(
    name,
    state_dim,
)
```


The function registers the two sample sites `f"{name}_start_mean"` (event shape `(state_dim,)`) and `f"{name}_start_cov"` (event shape `(state_dim, state_dim)`) and returns their values. They are sample sites, not deterministic sites, because the drivers substitute sample sites only. The values come from the posterior of the driver call, where [end_state()](contrib.dynestyx.end_state.md#numpyro_forecast.contrib.dynestyx.end_state) put them. The returned pair is the mean and the covariance of the filtering distribution at the last step T of the earlier window, p(x_T \mid y\_{1:T}, \theta), one per posterior draw \theta.


## Parameters


`name: str`  
The site prefix of the [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) call, as given to [end_state()](contrib.dynestyx.end_state.md#numpyro_forecast.contrib.dynestyx.end_state).

`state_dim: int`  
The dimension of the state vector.


## Returns


`tuple[Array, Array]`  
The mean, shape `(state_dim,)`, and the covariance, shape `(state_dim, state_dim)`, of the filtered state.


## Raises


`ValueError`  
If the posterior does not hold the two sites; if their values do not have the state dimension `state_dim`.


## Notes

- dynestyx puts the `initial_condition` at the first observation time, which is one step after the end of the earlier window. The model must move the state one transition forward before it builds the initial condition: `A @ mean` and `A @ cov @ A.T + Q` for a linear Gaussian evolution with matrix `A` and noise covariance `Q`. A nonlinear evolution needs a linearization.
- An evolution that reads the control needs the control of the last step of the earlier window for that transition. The covariates of the resumed window do not hold it.
- The time grid of [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) starts at 0 again in the resumed window, so the dynamics must not read the absolute time.
- The function is for prediction only. Use the resumed model function with [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample()](predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and a fixed posterior. Do not fit it with `MCMC` or `SVI`, where the two sites would be latent variables.
- A `LatentPathBuilder` and a `Smoother` record no filtered state. A particle filter records the mean and the covariance of its weighted particles under the same two site names, so [end_state()](contrib.dynestyx.end_state.md#numpyro_forecast.contrib.dynestyx.end_state) returns them without an error. The resumed state is then a Gaussian approximation of the particles and is not tested.
- For a linear Gaussian model the resumed form and the whole-window form agree up to floating point. With an extended Kalman filter they agree up to the linearization of one extra transition.


## See Also

[end_state()](contrib.dynestyx.end_state.md#numpyro_forecast.contrib.dynestyx.end_state)  
The function that builds the two sites from the samples of a filter run.

[state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space)  
The building block that runs the filter.


## Examples

The local level model of [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) in resumed form. Stage 1 fits the model on the first window and keeps the filtered state at its end. Stage 2 forecasts from the new observations only:

``` python
def resumed_local_level(covariates, data=None):
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    q = numpyro.sample("q", dist.HalfNormal(1.0))
    r = numpyro.sample("r", dist.HalfNormal(1.0))
    mean, cov = start_state("f", 1)
    dynamics = DynamicalModel(
        # one transition forward, with A = I and Q = q**2 I
        initial_condition=dist.MultivariateNormal(mean, cov + q**2 * jnp.eye(1)),
        state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=q**2 * jnp.eye(1)),
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=r**2 * jnp.eye(1)),
    )
    state_space(h, "f", y, dynamics)


# stage 1: fit local_level on the first window, keep the parameters and the end state
kalman = KFConfig(filter_source="cuthbert")
with Filter(filter_config=kalman):
    mcmc.run(key_fit, series[:t_obs], series[:t_obs])

samples = mcmc.get_samples()
parameters = {k: v for k, v in samples.items() if not k.startswith("f_")}
posterior = parameters | end_state(samples, "f")

# stage 2: filter the new observations only, then forecast
new, covariates = series[t_obs:t_new], series[t_obs : t_new + horizon]
with DiscreteTimeSimulator(n_simulations=1), Filter(filter_config=kalman):
    draws = forecast(key_forecast, resumed_local_level, posterior, new, covariates)
```
