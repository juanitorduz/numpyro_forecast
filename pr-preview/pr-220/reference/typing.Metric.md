## typing.Metric


A metric maps `(pred, truth)` forecast samples and ground truth to a scalar array.


`typing.Metric=Callable[[Array, Array], Array]`


`pred` has the sample axis first, shape `(sample, *batch)`; `truth` has shape `(*batch)`; the result is a 0-d array. Metrics must be pure JAX functions (jit- and vmap-compatible): host floats appear only at result boundaries ([evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast), [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult)), never inside metrics, so [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) can vmap any metric over the window axis. Parametrize by closure: `functools.partial` for keywords (e.g. `partial(eval_coverage, alpha=0.5)`) or a factory like [make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase).

The built-in metrics and [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast) check that `truth.shape` is `pred.shape[1:]` and raise `ValueError` otherwise. With the runtime type hook on (the default), the jaxtyping annotations reject a mismatched array first and raise `jaxtyping.TypeCheckError`. A Python or NumPy scalar `truth` passes the annotations, so the explicit check raises `ValueError` for it. Either way a mismatched `truth` never broadcasts into a wrong value.
