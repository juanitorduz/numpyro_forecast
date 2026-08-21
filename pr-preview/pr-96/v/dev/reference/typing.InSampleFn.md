## typing.InSampleFn


A closure that fits a model on a training window and scores its in-sample fit.


`typing.InSampleFn=Callable[…, ``"Array | np.ndarray"]`


Called by `~numpyro_forecast.evaluate.backtest()` (only when `eval_train=True`) positionally as `in_sample_fn(rng_key, model, train_data, train_covariates, num_samples, *, batch_size=None)`. Must return in-sample posterior-predictive samples with the sample axis first, shape `(num_samples, *batch, t1 - t0, obs)`. The same `batch_size`/on-device requirements as [ForecastFn](typing.ForecastFn.md#numpyro_forecast.typing.ForecastFn) apply.
