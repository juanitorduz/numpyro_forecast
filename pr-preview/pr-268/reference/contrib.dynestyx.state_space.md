## contrib.dynestyx.state_space()


Condition, forecast or predict in-sample a dynestyx model over the horizon.


Usage

``` python
contrib.dynestyx.state_space(
    h,
    name,
    y,
    dynamics,
    *,
    controls=None,
)
```


One `dsx.sample` call on the integer time grid `0, ..., h.duration - 1` (`obs_times` are the first `h.t_obs` steps, `predict_times` the rest, `ctrl_times` the whole grid), interpreted by the dynestyx handlers active around the model call. What the block registers depends on the mode of the call, exactly as the other building blocks decide it from `h`:

- `h.data` given and `h.future == 0` (training): nothing. The active `Filter`, `Smoother` or `LatentPathBuilder` adds its likelihood factor, which is what `MCMC` and `SVI` fit.
- `h.data` given and `h.future > 0` (forecasting): the site `"forecast"`, shape `(future, obs)`, read from the rollout of a `DiscreteTimeSimulator(n_simulations=1)` placed outside the `Filter`, `Smoother` or `LatentPathBuilder` (the explicit simulator is the recommended form; dynestyx 0.8 adds one itself when none is active). The rollout starts from the conditioned state at the last observation: the filtering or smoothing distribution at the parameters of each posterior draw (the two coincide there), or the sampled final state of the path under a `LatentPathBuilder`, which therefore inherits the mixing of the path. The block asks the simulator for `predict_times` starting at the last observation and drops that first row, because dynestyx places the rollout's initial state at the first predict time without a transition.
- `h.data is None` (the in-sample predictive of [predict_in_sample()](predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree)): the site `"obs"`, shape `(t_obs, obs)`, one observation draw per step from `dynamics.observation_model(x_t, u_t, t)` at a draw of the in-sample states: step-wise from the smoothing marginals under a `Smoother`, the posterior path itself under a `LatentPathBuilder`. This is not a prior predictive: the block still conditions on `y` from the covariates, so `Predictive(model, num_samples=...)(rng_key, covariates)` returns draws conditioned on the observed window at prior parameters. Simulate a prior predictive with `dsx.sample` and `predict_times` under a `Simulator`, without `obs_values`.


## Parameters


`h: Horizon`  
The current call's horizon.

`name: str`  
Site prefix handed to `dsx.sample` (dynestyx registers its own sites as `f"{name}_..."`).

`y: Float[Array, ``" time obs"]`  
The observed window, shape `(h.t_obs, obs)`, read from the covariates by the caller (`covariates[..., : h.t_obs, :1]` for a series stored in column 0) so that it is present when the drivers call the model with `data=None`. NaN entries are missing observations and must be in `y` (the likelihood never reads `h.data`). The `LatentPathBuilder` and the Kalman and ensemble Kalman filters and smoothers of the cuthbert backend (`KFConfig`, `EnKFConfig`, `KFSmootherConfig`, `EnRTSSmootherConfig` with `filter_source="cuthbert"`, and the default `Filter()`, an ensemble Kalman filter) skip them. The other configurations reject them, including the cd_dynamax backend (the default of `KFConfig`) and the default `Smoother()`. The particle filter returns a NaN log likelihood (its warning disappears inside the jitted drivers), and dynestyx does not check the HMM filters. Under a `LatentPathBuilder`, a partly missing row of an observation family the builder cannot marginalize adds a latent `f"{name}_missing_obs_values"` site.

`dynamics: Any`  
A `dynestyx.DynamicalModel` with a vector state (one row of the path per observed step). Its `observation_model(x, u, t)` must return a NumPyro distribution for one time step (dynestyx's `LinearGaussianObservation` or any callable), which the in-sample predictive samples.

`controls: Float[Array, ``" duration control"] | None`` = None`  
Optional control inputs over the whole horizon, shape `(h.duration, control)`; passed as `ctrl_values` with `ctrl_times` equal to the full grid, and sliced to the observed window for the in-sample predictive.


## Returns


`StateSpaceResult`  
`x_future` while forecasting and `x_in_sample` when called without data, `None` otherwise.


## Raises


`ValueError`  
If no `Filter`, `Smoother` or `LatentPathBuilder` is active; if `y` or `controls` do not span the horizon; if the forecast runs with `n_simulations != 1`; if the in-sample predictive runs under a `Filter`, receives no per-step marginals, or gets a state path that is not `(t_obs, state)`; if a dynestyx site is missing (for example under `numpyro.handlers.scope`). A fresh `LatentPathBuilder` inside a jitted driver raises dynestyx's own `ValueError` about the missingness pattern.

`RuntimeError`  
If the in-sample predictive runs without a seed handler.


## Notes

- The `Smoother` in-sample draws are per-step marginals p(x_t \mid y\_{1:T}), independent across steps: exact for per-step bands and metrics, not a joint path. The `LatentPathBuilder` draws are joint. The smoother's marginals arrive as a Python list, so building the draws costs trace time that grows with the length of the window.
- A `Filter` and a `Smoother` that run the same forward filter add the same marginal likelihood, so the draws of a `Filter` fit can be exported under that `Smoother` without a refit: the Kalman filter with the Kalman (Rauch-Tung-Striebel) smoother, and an approximate filter with the smoother of the same configuration and seed. With an approximate filter, the posterior is only as good as that filter's likelihood estimate.
- Under a `LatentPathBuilder` the drivers need the posterior draws of `f"{name}_state_path_params"`, which only a fit under a builder has. A posterior from a `Filter` or `Smoother` fit has no path, and the builder then draws the path from its prior without an error: export such fits under a `Smoother`.
- The `LatentPathBuilder` around a driver call must be the instance that fitted the model: it caches the observation missingness layout per site name and shape from the concrete observations of the fit, and the jitted drivers hand it traced arrays. A later fit with the same site name and shape and a different missingness pattern overwrites that layout, and the drivers of the earlier model then fail at run time with dynestyx's error about the layout; use one builder per fitted model when the patterns differ. A fresh builder needs its first model run eager.
- `mcmc.get_samples()` also holds the deterministic sites dynestyx records (`f"{name}_filtered_states_mean"` and similar). The drivers ignore deterministic sites, but [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) stores them in the posterior group; drop them for a smaller tree, and keep `f"{name}_state_path_params"` under a `LatentPathBuilder`.


## Examples

A local level model, fitted with the Kalman filter and exported with the smoother around the driver call:

``` python
def local_level(covariates, data=None):
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]  # the series travels in the covariates
    q = numpyro.sample("q", dist.HalfNormal(1.0))
    r = numpyro.sample("r", dist.HalfNormal(1.0))
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=q**2 * jnp.eye(1)),
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=r**2 * jnp.eye(1)),
    )
    state_space(h, "f", y, dynamics)


with Filter(filter_config=KFConfig(filter_source="cuthbert")):
    mcmc.run(rng_key, series[:t_obs], series[:t_obs])

smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert"))
with DiscreteTimeSimulator(n_simulations=1), smoother:
    tree = to_datatree(rng_key, local_level, mcmc.get_samples(), series[:t_obs], series)
```
