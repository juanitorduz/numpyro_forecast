## typing.Metric


A metric maps `(pred, truth)` forecast samples and ground truth to a scalar array.


`typing.Metric=Callable[[Array, Array], Array]`


`pred` has the sample axis first, shape `(sample, *batch)`; `truth` has shape `(*batch)`; the result is a 0-d array. Metrics must be pure JAX functions (jit- and vmap-compatible): host floats appear only at result boundaries ([evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast), [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult)), never inside metrics, so [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) can vmap any metric over the window axis. Parametrize by closure: `functools.partial` for keywords (e.g. `partial(eval_coverage, alpha=0.5)`) or a factory like [make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase).
