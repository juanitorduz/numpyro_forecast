## reparam.time_reparam()


Reparameterize every in-sample time latent of `model` along the time axis.


Usage

``` python
reparam.time_reparam(
    model,
    transform,
)
```


Port of the [time_reparam](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) argument of Pyro's `Forecaster` and `HMCForecaster` ([forecaster.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/forecast/forecaster.py)). The returned model is `numpyro.handlers.reparam(model, config)` with a config that targets every non-observed continuous sample site under the `time` plate opened by [innovations()](models.innovations.md#numpyro_forecast.models.innovations), exactly as Pyro's `time_reparam_haar` targets every site inside its `time` plate. Each targeted site `name` becomes a `deterministic` site of the same shape, computed from a new sample site `f"{name}_haar"` or `f"{name}_dct"` whose event shape owns the time axis and every plate axis to its right (plate axes to the left of time stay batch axes). The transform is orthonormal and acts on the unconstrained value of each site, so the posterior and the unconstrained potential energy that NUTS and `AutoNormal` target are the same function of the rotated coordinates; only the geometry inference sees changes. For a site with a constrained support (positive or bounded), the `numpyro.infer.util.log_density` values and the `AutoDelta` MAP estimates are not those of the plain model: see the Notes.


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
If `model` already contains the result of [time_reparam()](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam). Nesting is not supported: the inner handler would turn the site into a deterministic before the outer one sees it, so the outer transform would silently be a no-op. The check runs when wrapping and looks through any NumPyro handlers around `model` (`scope`, `seed`, …). A nesting behind a plain function or a `functools.partial()` raises when the model is first traced (inside `SVI`, `MCMC`, [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast), …), if the outer transform finds no in-sample time latent of its own. A composite model whose body calls a [time_reparam()](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) component next to its own [innovations](models.innovations.md#numpyro_forecast.models.innovations) time latents is valid: the outer transform applies to those latents. A composite whose own time latents are never targeted ([markov_series](models.markov_series.md#numpyro_forecast.models.markov_series) or [ssoe](models.ssoe.md#numpyro_forecast.models.ssoe) sites, observed sites) also raises, because the outer transform does nothing there either.

Also if an in-sample time latent is under a subsampled plate when the model is traced: a plate opened with `subsample_size` smaller than its size, or a full model plate that the guide subsamples through `create_plates`. See the Notes.


## Notes

- Hand the wrapped model, not the original, to the guide, the sampler and the drivers: only the wrapped model has the auxiliary site that the posterior holds. Creating it once is a convenience, not a requirement: the drivers key nothing on the model object, so a rewrapped model gives the same draws.
- Apply it innermost. `scope(time_reparam(model), prefix="a")` yields `a/drift_haar`; `time_reparam(scope(model, "a"))` yields `a/a/drift_haar` because the auxiliary sample passes through `scope` a second time (generic NumPyro reparam-under-scope behavior).
- It composes with the per-block `reparam=` hook: after `innovations(..., reparam=LocScaleReparam(0))` the site under the plate is `drift_decentered`, so the auxiliary site is `drift_decentered_haar` and both `drift_decentered` and `drift` become deterministic. This matches Pyro, whose config applies to every site in the plate.
- Untouched sites: the `_future` suffix sites (they stay prior-drawn under `time_future`, so [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) is unaffected), the scan sites of [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series), the error sites of [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe), observed sites and discrete sites.
- Constrained supports. The rotation `T` acts on the unconstrained value `u = biject_to(support).inv(x)` of a targeted site `x`, so the auxiliary site holds `T(u)`. The potential energy in the unconstrained space is the same function of the rotated coordinates for every support: NUTS and the guides that work in that space (`AutoNormal`, for example) target the same density, only in the rotated basis. A mean-field guide such as `AutoNormal` is not rotation invariant, so its fit and its ELBO can change; that is the point of the feature. For a site with real support, `u` is `x` and the value of `numpyro.infer.util.log_density` is unchanged too. For a site with a constrained support, the value of `numpyro.infer.util.log_density` of the wrapped model at the auxiliary value `T(u)` is the value of the plain model at `x` plus the log absolute Jacobian determinant of `biject_to(support)` at `u`: `sum(log(x))` for a `LogNormal` site and `sum(log(x) + log(1 - x))` for a `Beta` site. Values of `numpyro.infer.util.log_density` and `AutoDelta` MAP estimates are therefore not comparable with those of the plain model (NumPyro's `AutoDelta` does MAP in constrained space, so the plain model's estimate is the mode in `x`; the wrapped model's auxiliary site is real-valued, so its estimate is the mode in `u`, which is what `AutoLaplaceApproximation` gives for the plain model). To keep them comparable, sample the innovation on the real line, for example `innovations(h, "log_growth", dist.Normal(0.0, 0.5))`, and apply `jnp.exp` to it (to its cumulative sum for a multiplicative random walk); the MAP is then the mode in the log for both models. Pyro's [time_reparam](reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) and NumPyro's `UnitJacobianReparam` behave the same way.
- Time latents under a subsampled plate are not supported. The auxiliary site is sampled outside every enclosing plate, so a subsampled plate would apply no `size / subsample_size` scale to it and an `AutoGuide` would size its parameters to the minibatch. A time latent under such a plate therefore raises `ValueError`, also when the guide subsamples the plate through `create_plates`. Time latents outside the subsampled plate (for example one level shared by all series) and a plate with `subsample_size` equal to its size are accepted.
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
