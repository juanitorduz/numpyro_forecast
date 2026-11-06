## typing.Metric


A metric maps `(pred, truth)` forecast samples and ground truth to a scalar array.


`typing.Metric=Callable[[Array, Array], Array]`


`pred` has the sample axis first, shape `(sample, *batch)`; `truth` has shape `(*batch)`; the result is a 0-d array. Metrics must be pure JAX functions (jit- and vmap-compatible): host floats appear only at result boundaries ([evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast), [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult)), never inside metrics, so [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) can vmap any metric over the window axis. Parametrize by closure: `functools.partial` for keywords (e.g. `partial(eval_coverage, alpha=0.5)`) or a factory like [make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase).

Every driver checks each result on its own, so the outcome for one metric never depends on the other metrics in the mapping: a 0-d bool or integer result is converted to float, and any other shape or dtype (for example the per-cell values of [crps_empirical()](metrics.crps_empirical.md#numpyro_forecast.metrics.crps_empirical), whose metric counterpart is [eval_crps()](evaluate.eval_crps.md#numpyro_forecast.evaluate.eval_crps)) raises `ValueError` with the name, shape and dtype of the metric. [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast) and [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) check the value they score; [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) checks the value of one window inside its `vmap`, and traces the scoring on the window shapes before the fit, so the error comes before any window is fitted.

The built-in metrics and [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast) check that `truth.shape` is `pred.shape[1:]` and raise `ValueError` otherwise. With the runtime type hook on (the default), the jaxtyping annotations reject a mismatched array first and raise `jaxtyping.TypeCheckError`. A Python or NumPy scalar `truth` passes the annotations, so the explicit check raises `ValueError` for it. Either way a mismatched `truth` never broadcasts into a wrong value.
