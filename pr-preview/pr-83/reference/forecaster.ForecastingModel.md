## forecaster.ForecastingModel


Abstract base class for forecasting models.


Usage

``` python
forecaster.ForecastingModel()
```


Subclasses implement [model()](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.model), which must call [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict) exactly once. The instance itself is the (pure) NumPyro model function with signature `model_instance(covariates, data=None)`: the forecast horizon is inferred from the shapes (`future = covariates.shape[-2] - data.shape[-2]`).

This is the object-oriented façade over the functional API: [time_series()](functional.models.time_series.md#numpyro_forecast.functional.models.time_series) and [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict) delegate to the free functions in `numpyro_forecast.functional`, passing the current `~numpyro_forecast.functional.models.Horizon`.


## Attributes

| Name | Description |
|----|----|
| [duration](#duration) | Total horizon length `t + future` (in time steps). |
| [future](#future) | Number of forecast time steps `f` (`0` while training). |
| [t_obs](#t_obs) | Number of observed (in-sample) time steps `t`. |

------------------------------------------------------------------------


#### duration


Total horizon length `t + future` (in time steps).


`duration: int`


------------------------------------------------------------------------


#### future


Number of forecast time steps `f` (`0` while training).


`future: int`


------------------------------------------------------------------------


#### t_obs


Number of observed (in-sample) time steps `t`.


`t_obs: int`


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Run the model as a NumPyro model function. |
| [markov_time_series()](#markov_time_series) | Sample a Markov (state-space) latent over the full horizon. |
| [model()](#model) | Define the generative model and call [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict) exactly once. |
| [predict()](#predict) | Register the observation/forecast sites for the model. |
| [predict_glm()](#predict_glm) | Register GLM-style observation/forecast sites from a latent predictor. |
| [time_series()](#time_series) | Sample a time-varying latent over the full horizon. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Run the model as a NumPyro model function.


Usage

``` python
__call__(covariates, data=None)
```


##### Parameters


`covariates: Array`  
Covariates with time at axis `-2` spanning the full horizon.

`data: Array | None = None`  
Observed data with time at axis `-2` (`None` for prior sampling).


------------------------------------------------------------------------


#### markov_time_series()


Sample a Markov (state-space) latent over the full horizon.


Usage

``` python
markov_time_series(
    name, init_carry, transition, xs=None, *, plates=(), reparam_config=None
)
```


Thin wrapper over [numpyro_forecast.functional.models.markov_time_series()](functional.models.markov_time_series.md#numpyro_forecast.functional.models.markov_time_series) that threads this model's train/forecast horizon. In-sample steps run in a `scan` under site `name`; the forecast horizon runs in a second scan under `f"{name}_future"` seeded by the final in-sample carry, so the guide never sees the future site.


##### Parameters


`name: str`  
Sample-site name for the in-sample scan; the forecast scan uses `f"{name}_future"`.

`init_carry: Any`  
Initial carry PyTree fed to the first transition step.

`transition: Transition`  
Callable `(carry, x_t) -> (dist_t, carry_fn)`: `dist_t` is the per-step observation distribution (its per-step shape must carry the trailing observation dimension) and `carry_fn(z_t)` builds the next carry from the sampled latent `z_t`. The wrapper owns the `sample` statement, so the Markov structure cannot be broken by resampling.

`xs: Array | None = None`  
Optional exogenous inputs spanning the full horizon with time at axis `-2`; split and moved into scan layout internally. `None` for autonomous dynamics.

`plates: Sequence[tuple[str, int]] = ()`\  
`(name, size)` pairs opened inside the scan body around the sample statement (the only placement NumPyro accepts around a scan).

`reparam_config: Mapping[str, Reparam] | None = None`  
Optional site-name to `~numpyro.infer.reparam.Reparam` mapping applied inside the scan body.


##### Returns


`Array`  
The latent over the full horizon in package layout `(*plate_batch, duration, obs)` (time at axis `-2`).


------------------------------------------------------------------------


#### model()


Define the generative model and call [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict) exactly once.


Usage

``` python
model(zero_data, covariates)
```


##### Parameters


`zero_data: Array | None`  
Zeros shaped like the data extended to the covariate duration (shape/dtype only; `None` during pure prior sampling).

`covariates: Array`  
Covariates with time at axis `-2` and shape `(*batch, duration, cov)`.


------------------------------------------------------------------------


#### predict()


Register the observation/forecast sites for the model.


Usage

``` python
predict(noise_dist, prediction)
```


Thin wrapper over [numpyro_forecast.functional.models.predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict).


##### Parameters


`noise_dist: dist.Distribution`  
Zero-centered observation noise (e.g. `Normal(0, sigma)`).

`prediction: Array`  
Deterministic mean with time at axis `-2`, shape `(*batch, duration, obs)`.


------------------------------------------------------------------------


#### predict_glm()


Register GLM-style observation/forecast sites from a latent predictor.


Usage

``` python
predict_glm(obs_dist_fn, latent)
```


Thin wrapper over [numpyro_forecast.functional.models.predict_glm()](functional.models.predict_glm.md#numpyro_forecast.functional.models.predict_glm).


##### Parameters


`obs_dist_fn: Callable[[Array], dist.Distribution]`  
Link mapping the full-horizon `latent` predictor to the observation distribution (e.g. `lambda eta: Poisson(jnp.exp(eta))`).

`latent: Array`  
The deterministic latent predictor over the full horizon, time at axis `-2`.


------------------------------------------------------------------------


#### time_series()


Sample a time-varying latent over the full horizon.


Usage

``` python
time_series(name, dist_fn, *, reparam=None)
```


Thin wrapper over [numpyro_forecast.functional.models.time_series()](functional.models.time_series.md#numpyro_forecast.functional.models.time_series).


##### Parameters


`name: str`  
Base sample-site name for the in-sample latent.

`dist_fn: Callable[[], dist.Distribution]`  
Zero-argument callable returning the per-step prior distribution.

`reparam: Reparam | None = None`  
Optional reparameterization (e.g. `LocScaleReparam`) applied to both the in-sample and forecast sites.


##### Returns


`Array`  
The latent over the full horizon with time at axis `-2`.
