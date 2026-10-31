## models.predict()


Register the observation and forecast sites for the model.


Usage

``` python
models.predict(
    h,
    obs_dist,
    prediction,
)
```


`prediction` is the deterministic predictor over the full horizon (time at axis `-2`, shape `(*batch, duration, obs)`). `obs_dist` takes one of two forms. A `numpyro.distributions.Distribution` is observation noise with location `0` (e.g. `dist.StudentT(nu, 0.0, sigma)`) whose location is shifted onto the predictor with [shift_loc()](surgery.shift_loc.md#numpyro_forecast.surgery.shift_loc), which also owns the multivariate-normal layout: noise correlated over time, `dist.MultivariateNormal(jnp.zeros(h.duration), cov)`, gets the event shape `(time, 1)`, so `"obs"` scores each series jointly and the forecast keeps the observation axis. The predictor is then the location of the observation distribution: the mean for `Normal`, `StudentT` (`df > 1`), `Laplace` and `MultivariateNormal` noise, but the mode for `Gumbel` and `AsymmetricLaplace` noise, whose mean is offset from it (for `AsymmetricLaplace` unless `asymmetry = 1`), and for `Cauchy` noise, which has no mean (see [shift_loc()](surgery.shift_loc.md#numpyro_forecast.surgery.shift_loc)). A callable is a link mapping the predictor to the observation distribution directly (the GLM form, e.g. `lambda eta: dist.Poisson(jnp.exp(eta))`). Either way, while training the observation site `"obs"` is observed; while forecasting the in-sample prefix is observed and the forecast suffix is sampled at `"obs_future"` and exposed as the `"forecast"` deterministic site that [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) reads. The observation distribution must support time-axis surgery ([slice_time()](surgery.slice_time.md#numpyro_forecast.surgery.slice_time) / [prefix_condition()](surgery.prefix_condition.md#numpyro_forecast.surgery.prefix_condition)), i.e. an elementwise family or a registered one. This also holds for a distribution wrapped in `Independent` (e.g. by `.to_event(1)`).


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon)).

`obs_dist: dist.Distribution | Callable[[Array], dist.Distribution]`  
Noise distribution with location `0` (its location is shifted onto `prediction`) or a link callable from the full-horizon predictor to the observation distribution.

`prediction: Array`  
The deterministic predictor over the full horizon, shape `(*batch, duration, obs)`: `h.duration` steps at axis `-2` and the observation axis at `-1`, so `(duration, 1)` for a univariate series and `(duration, n)` for a panel of `n` series (or `(duration, 1)` inside a size-`n` plate at `dim=-1`). A 0-d or per-series predictor also works when a parameter of `obs_dist` carries the time axis, for example a constant mean with a `(duration, 1)` noise scale.


## Raises


`TypeError`  
If `obs_dist` is a callable that does not return a distribution (for example a link that returns the predictor itself).

`RuntimeError`  
If forecasting (`future > 0`) but no observed data is available.

`ValueError`  
If `prediction` has shape `(duration,)` (for example `covariates @ beta`, which needs `[..., None]`); if the observation distribution has fewer than two axes; if `h.data` is given and the observation distribution would score it on a grid larger than `h.data` (for example a `(duration,)` term added to a `(duration, 1)` predictor, a `(duration,)` noise scale or a second `[..., None]`); or if the observation distribution has discrete support but `h.data` is not integer-dtyped (the usual mistake for count models). The shape checks cannot tell a `(duration,)` term from a per-series one when `h.data` has `h.duration` series (a square panel in training).
