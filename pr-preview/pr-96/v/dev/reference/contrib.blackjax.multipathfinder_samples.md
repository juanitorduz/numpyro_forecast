## contrib.blackjax.multipathfinder_samples()


Draw `num_samples` posterior samples from a fitted multipath Pathfinder fit.


Usage

``` python
contrib.blackjax.multipathfinder_samples(
    rng_key,
    fit,
    num_samples,
    *,
    resample="auto",
    batch_size=None,
    device=None
)
```


Every call draws fresh unconstrained samples from each path's fitted normal approximation (`num_samples` per path, via `blackjax.vi.pathfinder.sample` vmapped over the paths) and then combines the paths into a single set of `num_samples` draws. Nothing is recycled from the small pool stored at fit time, so asking for more draws than that pool held costs nothing in duplication. The output contract is identical to `~numpyro_forecast.functional.posterior.draw_posterior()` and [pathfinder_samples()](contrib.blackjax.pathfinder_samples.md#numpyro_forecast.contrib.blackjax.pathfinder_samples): `~numpyro_forecast.functional.prediction.forecast()` and NumPyro's `Predictive` consume it unchanged. Chunking and device offload are delegated to the shared `~numpyro_forecast.functional._offload._draw_chunked()` driver, so the same memory-bounding contract applies.


## Resampling Strategies

`resample="psis"` Pool the `num_paths * num_samples` fresh draws, score them under the model (`logp`) and under their own path's approximation (`logq`), Pareto-smooth the importance ratios, and resample `num_samples` of them with replacement. This is the textbook multipath Pathfinder estimator and the most faithful one *when the importance weights behave*. `resample="elbo"` Pick a whole path per draw, with probability `softmax(fit.elbos)` over the paths whose ELBO is finite (non-finite paths get zero weight, and an all-non-finite fit degrades to a uniform choice), then take one fresh draw from that path. With ELBO gaps of hundreds of nats this collapses onto the single best path, which is the right answer when the paths disagree strongly: no single draw is ever duplicated. `resample="auto"` (default) `"psis"` when `fit.pareto_k <= 0.7`, `"elbo"` otherwise.

The gate matters because importance sampling degenerates in high dimensions: on a posterior with hundreds of parameters the ratio `logp - logq` is dominated by a handful of draws, `pareto_k` climbs far above `0.7`, and PSIS resampling collapses the answer onto those few draws. ELBO-weighted path sampling never concentrates like that, because it reweights *paths* (a handful of well-separated numbers) rather than individual draws.

PRNG: within each chunk (the whole draw, when unchunked), the chunk key is split into a model-initialization stream (used only to rebuild the constraining transform), a per-path sampling stream, and a resampling-index stream. The split is the same in every mode, so `resample="auto"` produces bitwise the same draws as the explicit mode it resolves to.


## Parameters


`rng_key: Array`  
PRNG key.

`fit: MultiPathfinderFit`  
A fit from [fit_multipathfinder()](contrib.blackjax.fit_multipathfinder.md#numpyro_forecast.contrib.blackjax.fit_multipathfinder).

`num_samples: int`  
Number of posterior draws.

`resample: Literal[``"auto", `<span class="st">`"psis"``, ``"elbo"``]`</span>` = ``"auto"`  
How to combine the per-path draws: `"auto"` (default), `"psis"`, or `"elbo"`; see "Resampling strategies" above.

`batch_size: int | None = None`  
Optional chunk size for the drawing itself; see `~numpyro_forecast.functional.posterior.draw_posterior()` (the same memory/reproducibility contract applies). Note that each chunk draws `num_paths * batch_size` samples internally.

`device: jax.Device | str | None = None`  
Where each chunk of draws is moved as soon as it is drawn; see `~numpyro_forecast.functional.posterior.draw_posterior()`.


## Returns


`dict[str, Array | np.ndarray]`  
Posterior samples of the latent sites, sample axis leading (NumPy leaves when `device` resolves to `"host"`).


## Raises


`ValueError`  
If `num_samples` or `batch_size` is not positive, or `resample` is not one of `"auto"`, `"psis"`, `"elbo"`.


## Warns


`UserWarning`  
In PSIS mode, if the `pareto_k` recomputed on this call's fresh draws exceeds `0.7`. This is a separate diagnostic from `fit.pareto_k`, which is computed once over the pool stored at fit time.
