## typing.ForecastFn


A closure that fits a model on a training window and forecasts its test horizon.


`typing.ForecastFn=Callable[…, ``"Array | np.ndarray"]`


Called by `~numpyro_forecast.evaluate.backtest()` positionally as `forecast_fn(rng_key, model, train_data, train_covariates, full_covariates, num_samples, *, batch_size=None)`, where `full_covariates` spans the *full* window (train followed by test, i.e. `covariates[..., t0:t2, :]`). Must return forecast samples with the sample axis first, shape `(num_samples, *batch, t2 - t1, obs)`. `batch_size` is forwarded unchanged from [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) so a chunked closure can bound its own device memory; a closure that offloads work internally must return the draws back on-device (the metrics scoring them are jitted). Typed loosely (a bare `Callable`, like `Metric`) because per-backend fit options differ; the exact shapes are pinned above rather than in the type itself.
