## typing.ForecastFn


A closure that fits a model on a training window and forecasts its test horizon.


`typing.ForecastFn=Callable[…, Array]`


Called by `~numpyro_forecast.evaluate.backtest()` positionally as `forecast_fn(rng_key, model, train_data, train_covariates, full_covariates, num_samples, *, batch_size=None)`, where `full_covariates` spans the *full* window (train followed by test, i.e. `covariates[..., t0:t2, :]`). Must return forecast samples with the sample axis first, shape `(num_samples, *batch, t2 - t1, obs)`. `batch_size` is forwarded unchanged from [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) so a chunked closure can bound its own device memory; a closure may return draws committed to host memory (e.g. via `device="host"`, to cap peak accelerator usage): every metric in `~numpyro_forecast.evaluate.DEFAULT_METRICS` accepts a host-committed `pred` or `truth` (or both), in any mix and regardless of `batch_size`, moving a host-committed operand to device memory first where needed. Returning draws already on-device still avoids the extra host-to-device hop for the metrics scored every window. Typed loosely (a bare `Callable`, like `Metric`) because per-backend fit options differ; the exact shapes are pinned above rather than in the type itself.
