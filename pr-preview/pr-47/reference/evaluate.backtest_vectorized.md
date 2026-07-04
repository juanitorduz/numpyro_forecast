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
    stride=1,
    num_steps=1001,
    optim=None,
    guide=None,
    num_samples=100,
    metrics=None,
    keep_predictions=False
)
```


Estimator-equivalent to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) with rolling windows; it differs only in PRNG stream layout and float reduction order, so the equivalence is statistical, not bitwise. Model, guide, and SVI compile once regardless of the number of windows (invariant I3), giving order-of-magnitude wall-clock wins for tens of windows on small models.

PRNG: `fold_in(rng_key, -1)` seeds a discarded eager warm-up init; `fold_in(rng_key, i)` seeds window `i`; each window key is split for the posterior draw and the forecast.


## Parameters


`rng_key: Array`  
Base PRNG key.

`data: Array`  
Dataset with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` (same duration as `data`).

`model_fn: ModelFactory`  
Factory returning a fresh model; called exactly once (per-window model variation is unsupported here, use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)).

`train_window: int`  
Fixed training-window length (`>= 1`).

`test_window: int`  
Fixed test-window length (`>= 1`).

`stride: int = ``1`  
Step between successive windows (`>= 1`).

`num_steps: int = ``1001`  
Number of SVI steps per window.

`optim: OptimizerLike = None`  
Optimizer specification resolved by `~numpyro_forecast.functional.resolve_optimizer()`.

`guide: GuideLike = None`  
Guide specification; must resolve to an `AutoGuide` (hand-written guides are not vmappable, use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)).

`num_samples: int = ``100`  
Number of forecast samples drawn per window.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; defaults to `DEFAULT_METRICS`.

`keep_predictions: bool = ``False`  
If `True`, retain the stacked forecast samples on the result.


## Returns


`VectorizedBacktestResult`  
The stacked per-window losses, metrics, and window indices.


## Raises


`ValueError`  
If `data` and `covariates` durations differ, `train_window` or `test_window` or `stride` is `< 1`, the resolved guide is not an `AutoGuide`, or there is no room for a single window.
