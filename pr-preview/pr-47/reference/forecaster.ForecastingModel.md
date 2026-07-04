## forecaster.ForecastingModel


Abstract base class for forecasting models.


Usage

``` python
forecaster.ForecastingModel()
```


Subclasses implement [model()](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.model), which must call [predict()](functional.predict.md#numpyro_forecast.functional.predict) exactly once. The instance itself is the (pure) NumPyro model function with signature `model_instance(covariates, data=None)`: the forecast horizon is inferred from the shapes (`future = covariates.shape[-2] - data.shape[-2]`).

This is the object-oriented façade over the functional API: [time_series()](functional.time_series.md#numpyro_forecast.functional.time_series) and [predict()](functional.predict.md#numpyro_forecast.functional.predict) delegate to the free functions in `numpyro_forecast.functional`, passing the current `~numpyro_forecast.functional.Horizon`.


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
| [model()](#model) | Define the generative model and call [predict()](functional.predict.md#numpyro_forecast.functional.predict) exactly once. |
| [predict()](#predict) | Register the observation/forecast sites for the model. |
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


#### model()


Define the generative model and call [predict()](functional.predict.md#numpyro_forecast.functional.predict) exactly once.


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


Thin wrapper over [numpyro_forecast.functional.predict()](functional.predict.md#numpyro_forecast.functional.predict).


##### Parameters


`noise_dist: dist.Distribution`  
Zero-centered observation noise (e.g. `Normal(0, sigma)`).

`prediction: Array`  
Deterministic mean with time at axis `-2`, shape `(*batch, duration, obs)`.


------------------------------------------------------------------------


#### time_series()


Sample a time-varying latent over the full horizon.


Usage

``` python
time_series(name, dist_fn, *, reparam=None)
```


Thin wrapper over [numpyro_forecast.functional.time_series()](functional.time_series.md#numpyro_forecast.functional.time_series).


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
