## evaluate.backtest()


Backtest a forecasting model on a moving window of `(train, test)` data.


Usage

``` python
evaluate.backtest(
    rng_key,
    data,
    covariates,
    model_fn,
    *,
    forecast_fn,
    in_sample_fn=None,
    metrics=None,
    per_window_metrics=None,
    transform=None,
    window_type=None,
    train_window=None,
    min_train_window=1,
    test_window=None,
    min_test_window=1,
    stride=1,
    num_samples=100,
    batch_size=None,
    eval_train=False,
    keep_predictions=False,
    reuse_model=True
)
```


Fitting and forecasting are delegated entirely to user-supplied closures rather than an OOP forecaster: `forecast_fn` fits `model` on the training window and forecasts the test horizon, and the optional `in_sample_fn` fits and scores the in-sample fit. Both closures own their own inference backend (SVI, MCMC, or anything else), so [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) itself has no dependency on how a model is fit.

`forecast_fn` has the call signature (see `~numpyro_forecast.typing.ForecastFn`)::

    forecast_fn(
        rng_key, model, train_data, train_covariates, full_covariates,
        num_samples, *, batch_size=None,
    ) -> draws  # shape (num_samples, *batch, t2 - t1, obs)

where `full_covariates` spans the *full* window, `covariates[..., t0:t2, :]` (train followed by test), matching what the model needs to run the forecast horizon. The optional `in_sample_fn` has the call signature (see `~numpyro_forecast.typing.InSampleFn`)::

    in_sample_fn(
        rng_key, model, train_data, train_covariates, num_samples, *, batch_size=None,
    ) -> draws  # shape (num_samples, *batch, t1 - t0, obs)

`batch_size` is forwarded unchanged into both closures so a chunked implementation can bound its own device memory. A closure that offloads work internally (e.g. moves draws to host memory to cap peak accelerator usage) must return the draws back on-device before returning: the metrics computed by [evaluate_forecast()](evaluate.evaluate_forecast.md#numpyro_forecast.evaluate.evaluate_forecast) are jitted and expect array inputs.

A minimal `forecast_fn` built on plain NumPyro (`AutoNormal` + `SVI.run` + `Predictive`)::

    import numpyro
    from jax import random
    from numpyro.infer import SVI, Predictive, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal


    def forecast_fn(
        rng_key,
        model,
        train_data,
        train_covariates,
        full_covariates,
        num_samples,
        *,
        batch_size=None,
    ):
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        key_fit, key_post, key_pred = random.split(rng_key, 3)
        state = svi.run(key_fit, 1_000, train_covariates, train_data, progress_bar=False)
        posterior = guide.sample_posterior(key_post, state.params, sample_shape=(num_samples,))
        predictive = Predictive(model, posterior_samples=posterior, return_sites=["forecast"])
        return predictive(key_pred, full_covariates, train_data)["forecast"]


## Parameters


`rng_key: Array`  
Base PRNG key (used for every window, matching Pyro).

`data: Array`  
Dataset with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` (same duration as `data`).

`model_fn: ModelFactory`  
Factory returning a fresh `~numpyro_forecast.typing.ForecastModel` per window.

`forecast_fn: ForecastFn`  
Closure that fits `model` on the training window and forecasts the test horizon (see `~numpyro_forecast.typing.ForecastFn` and the contract above).

`in_sample_fn: InSampleFn | None = None`  
Optional closure that fits `model` on the training window and scores its in-sample fit (see `~numpyro_forecast.typing.InSampleFn` and the contract above). Required when `eval_train=True`.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; defaults to `DEFAULT_METRICS`. Each function takes `(pred, truth)` and returns a scalar array (see `~numpyro_forecast.typing.Metric`); bind any metric-specific parameters with `functools.partial()`, e.g. `{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}`.

`per_window_metrics: Callable[[int, int, int], Mapping[str, Metric]] | None = None`  
Optional `(t0, t1, t2) -> Mapping[str, Metric]` callable producing extra metrics merged over `metrics` for each window. Use it for window-dependent metrics such as a MASE scaled by that window's training data ([numpyro_forecast.metrics.make_mase()](metrics.make_mase.md#numpyro_forecast.metrics.make_mase)).

`transform: Callable[[Array, Array], tuple[Array, Array]] | None = None`  
Optional `(pred, truth) -> (pred, truth)` applied before metrics.

`window_type: WindowType | None = None`  
Windowing strategy. If `None` (default) it is inferred from `train_window`: `"expanding"` when `train_window` is `None` and `"rolling"` when it is set, matching the historical behavior. Pass `"expanding"` to always train on all history from `t0 = 0`, or `"rolling"` to hold the training length fixed at `train_window` and slide it forward. `"expanding"` and `train_window` are mutually exclusive, and `"rolling"` requires `train_window` (both validated).

`train_window: int | None = None`  
Training window size; if `None` the window expands from the start. Required for `window_type="rolling"`.

`min_train_window: int = ``1`  
Minimum training window size for the expanding strategy (used when `train_window` is `None`).

`test_window: int | None = None`  
Test window size; if `None` forecasts to the end of the data.

`min_test_window: int = ``1`  
Minimum test window size when `test_window` is `None`.

`stride: int = ``1`  
Step between successive train/test splits.

`num_samples: int = ``100`  
Number of forecast samples per window.

`batch_size: int | None = None`  
Optional chunk size forwarded to `forecast_fn` and `in_sample_fn` (see the contract above).

`eval_train: bool = ``False`  
If `True`, also score the in-sample posterior predictive over each training window with the same `metrics` and store them in `BacktestResult.train_metrics`. Requires `in_sample_fn`.

`keep_predictions: bool = ``False`  
If `True`, store each window's out-of-sample forecast samples (after `transform`) on `BacktestResult.prediction`. Defaults to `False` to avoid retaining large Monte Carlo arrays.

`reuse_model: bool = ``True`  
When `True` (default) and the windowing strategy is rolling, the model instance returned by the first `model_fn()` call is reused for every window so forecast/predict kernels can cache across windows. SVI still recompiles per window; for a single fused fit over all windows use [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized). Ignored for expanding windows and when `False`.


## Returns


`list[BacktestResult]`  
One result per backtest window.


## Raises


`ValueError`  
If `data` and `covariates` durations differ, or if `eval_train=True` but `in_sample_fn` is `None`.
