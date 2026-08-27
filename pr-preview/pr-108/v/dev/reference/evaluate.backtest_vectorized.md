## evaluate.backtest_vectorized()


Rolling-window backtest with all windows fitted in one vmapped SVI run.


Usage

``` python
evaluate.backtest_vectorized(
    rng_key,
    data,
    covariates,
    model_fn,
    *,
    train_window,
    test_window,
    guide,
    optim=None,
    stride=1,
    num_steps=1001,
    num_samples=100,
    metrics=None,
    keep_predictions=False
)
```


Estimator-equivalent to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) with rolling windows; it differs only in PRNG stream layout and float reduction order, so the equivalence is statistical, not bitwise. Model, guide, and SVI compile once regardless of the number of windows, giving order-of-magnitude wall-clock wins for tens of windows on small models.

PRNG: `fold_in(rng_key, -1)` seeds a discarded eager warm-up init; `fold_in(rng_key, i)` is the window-`i` parent, split into SVI-init, posterior-draw, and forecast subkeys (the parent itself is never consumed; see `_window_key_streams()`).


## Parameters


`rng_key: Array`  
Base PRNG key.

`data: Array`  
Dataset with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` (same duration as `data`).

`model_fn: ModelFactory`  
Factory returning a fresh model; called exactly once (per-window model variation is unsupported here, use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)). Must return the same model object `guide` was built on (see `guide` below).

`train_window: int`  
Fixed training-window length (`>= 1`).

`test_window: int`  
Fixed test-window length (`>= 1`).

`guide: AutoGuide | Callable[…, None]`  
An `AutoGuide` instance built on the same model object `model_fn` returns (hand-written guides are not vmappable, use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) instead):

``` python
model = make_model()
backtest_vectorized(..., model_fn=lambda: model, guide=AutoNormal(model))
```

`optim: _NumPyroOptim | None = None`  
NumPyro optimizer; `None` uses `numpyro.optim.Adam(0.01)`. For an optax optimizer, wrap it with `numpyro.optim.optax_to_numpyro` first.

`stride: int = ``1`  
Step between successive windows (`>= 1`).

`num_steps: int = ``1001`  
Number of SVI steps per window.

`num_samples: int = ``100`  
Number of forecast samples drawn per window.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; defaults to `DEFAULT_METRICS`. Each metric is vmapped over the window axis, so any pure-JAX mapping, including partial-bound variants such as `{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.5)}`, is scored inside the single fused computation. Host-side metrics are unsupported here; use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest), or `keep_predictions=True` and score on the host.

`keep_predictions: bool = ``False`  
If `True`, retain the stacked forecast samples on the result.


## Returns


`VectorizedBacktestResult`  
The stacked per-window losses, metrics, and window indices.


## Raises


`ValueError`  
If `data` and `covariates` durations differ.

`BacktestWindowError`  
If `train_window`, `test_window`, or `stride` is `< 1`, or there is no room for a single window.

`VectorizedGuideError`  
If `guide` is not an `AutoGuide` instance.

`VectorizedMetricError`  
If a metric forces a host conversion under `vmap` (it is not a pure JAX function).
