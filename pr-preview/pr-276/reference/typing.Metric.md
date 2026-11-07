## typing.Metric


A metric maps `(pred, truth)` forecast samples and ground truth to a scalar array.


`typing.Metric=Callable[[Array, Array], Array]`


`pred` has the sample axis first, shape `(sample, *batch)`; `truth` has shape `(*batch)`; the result is a 0-d array. Metrics must be pure JAX functions (jit- and vmap-compatible): host floats appear only at result boundaries ([evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast), [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult)), never inside metrics, so [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) can vmap any metric over the window axis. Parametrize by closure: `functools.partial` for keywords (e.g. `partial(eval_coverage, alpha=0.5)`) or a factory like [make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase).

The built-in metrics and [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast) check that `truth.shape` is `pred.shape[1:]` and raise `ValueError` otherwise. With the runtime type hook on (the default), the jaxtyping annotations reject a mismatched array first and raise `jaxtyping.TypeCheckError`. A Python or NumPy scalar `truth` passes the annotations, so the explicit check raises `ValueError` for it. Either way a mismatched `truth` never broadcasts into a wrong value.

`pred` and `truth` may have any real dtype. A count model (a discrete observation family in [predict()](models.predict.md#numpyro_forecast.models.predict)) needs integer data, so its forecast draws are integers. The built-in metrics cast an integer operand to the default float dtype (`float32`, or `float64` with x64 enabled) before they compute, and keep a floating operand in its own dtype. `float32` holds every integer up to 2^{24} exactly; larger counts round unless x64 is enabled. [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast) (and with it [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)) and [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) cast once before they call the metrics, so every metric receives floating arrays and can compute `jnp.abs(pred - truth)` without integer wraparound; a custom metric that needs the integer draws must read them from the kept predictions (`keep_predictions=True`) or cast back inside the metric. The built-in metrics reject boolean and complex arrays.
