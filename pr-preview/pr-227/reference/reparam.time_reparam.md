## reparam.time_reparam()


Reparameterize every in-sample time latent of `model` along the time axis.


Usage

``` python
reparam.time_reparam(
    model,
    transform,
)
```


Port of the [time_reparam](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) argument of Pyro's `Forecaster` and `HMCForecaster` ([forecaster.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/forecast/forecaster.py)). The returned model is `numpyro.handlers.reparam(model, config)` with a config that targets every non-observed continuous sample site under the `time` plate opened by [innovations()](models.innovations.md#numpyro_forecast.models.innovations), exactly as Pyro's `time_reparam_haar` targets every site inside its `time` plate. Each targeted site `name` becomes a `deterministic` site of the same shape, computed from a new sample site `f"{name}_haar"` or `f"{name}_dct"` whose event shape owns the time axis and every plate axis to its right (plate axes to the left of time stay batch axes). The transform is orthonormal, so the model's log density is unchanged; only the geometry inference sees changes.


## Parameters


`model: ForecastModel`  
A forecasting model `(covariates, data=None) -> None`.

`transform: TimeTransform`  
`"haar"` for `numpyro.distributions.transforms.HaarTransform` (a multi-resolution average/difference basis, suited to blocky or multi-scale dynamics) or `"dct"` for `numpyro.distributions.transforms.DiscreteCosineTransform` (a cosine frequency basis, close to the Karhunen-Loeve basis of first-order Markov processes). Note that Pyro's own string mapping is swapped: its `"haar"` runs a DCT and its `"dct"` runs a Haar transform.


## Returns


`ForecastModel`  
The wrapped model. It is a `numpyro.handlers.reparam` handler and is the single object to hand to the guide, `SVI` / `MCMC` / blackjax, [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample()](predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample), [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and the `model_fn` of [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest).


## Raises


`ValueError`  
If `model` is already the result of [time_reparam()](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam). Nesting is not supported: the inner handler would turn the site into a deterministic before the outer one sees it, so the outer transform would silently be a no-op.


## Notes

- Create the wrapped model once and reuse it: the drivers jit-compile with the model as a static argument, so wrapping again inside a loop recompiles.
- Apply it innermost. `scope(time_reparam(model), prefix="a")` yields `a/drift_haar`; `time_reparam(scope(model, "a"))` yields `a/a/drift_haar` because the auxiliary sample passes through `scope` a second time (generic NumPyro reparam-under-scope behavior).
- It composes with the per-block `reparam=` hook: after `innovations(..., reparam=LocScaleReparam(0))` the site under the plate is `drift_decentered`, so the auxiliary site is `drift_decentered_haar` and both `drift_decentered` and `drift` become deterministic. This matches Pyro, whose config applies to every site in the plate.
- Untouched sites: the `_future` suffix sites (they stay prior-drawn under `time_future`, so [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) is unaffected), the scan sites of [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series), the error sites of [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe), observed sites and discrete sites.
- Posterior dictionaries from [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) and `mcmc.get_samples()` contain both `drift` (deterministic) and `drift_haar`; `Predictive` substitutes only the latter and recomputes the former. `init_to_value` must therefore target `drift_haar`.
- Measured on random-walk level models: mean-field `AutoNormal` reaches a better ELBO for both transforms (DCT slightly ahead of Haar), but an optimizer schedule tuned for the original coordinates does not transfer as is. The rotated posterior is better conditioned, so it tolerates and may need a larger learning rate to converge within the same step budget; a schedule that is too small stalls with part of the intercept still held by the level. NUTS trajectories become cheaper (fewer leapfrog steps per iteration) while the effective sample size per draw is model dependent.


## Examples

``` python
model_dct = time_reparam(seasonal_model, "dct")
guide = AutoNormal(model_dct)
svi = SVI(model_dct, guide, Adam(0.01), Trace_ELBO())
svi_result = svi.run(key_fit, 1_500, covariates[:t_obs], data, progress_bar=False)
posterior = draw_posterior(key_post, guide, svi_result.params, num_samples=100)
samples = forecast(key_pred, model_dct, posterior, data, covariates)
```
