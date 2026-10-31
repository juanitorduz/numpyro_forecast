## evaluate.backtest_vectorized()


Rolling-window backtest with all windows fitted in one vmapped SVI run.


Usage

``` python
evaluate.backtest_vectorized(
    rng_key,
    model_fn,
    data,
    covariates,
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


Estimator-equivalent to [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) with rolling windows and a fresh guide fitted on each window; it differs only in PRNG stream layout and float reduction order, so the equivalence is statistical, not bitwise. Model, guide, and SVI compile once regardless of the number of windows, giving order-of-magnitude wall-clock wins for tens of windows on small models.

Each window fits its own copy of `guide`. An autoguide sets itself up at its first init and keeps state that depends on the data of that init: the support transform of every latent site, the deterministic sites, and for `AutoLaplaceApproximation` the loss whose Hessian gives the covariance. The copy of each window sets up on that window's training data, so all of this state comes from the window, as with a fresh guide per window in [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest).

PRNG: `fold_in(rng_key, i)` is the window-`i` parent, split into SVI-init, posterior-draw, and forecast subkeys (the parent itself is never consumed; see `_window_key_streams()`). The SVI-init subkey keys the setup of the window's guide copy, so a random `init_loc_fn` (such as the default `init_to_uniform`) starts each window from its own point, and so does a callable initializer in a hand-written guide, such as `numpyro.param("loc", lambda key: random.normal(key))`. It also seeds the Monte Carlo ELBO estimate of every SVI step of the window (`svi.init` keeps a key split from it in the SVI state).


## Parameters


`rng_key: Array`  
Base PRNG key.

`model_fn: ModelFactory`  
Factory returning a fresh model; called exactly once (per-window model variation is unsupported here, use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest)). Must return the same model object `guide` was built on (see `guide` below).

`data: Array`  
Dataset with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` (same duration as `data`).

`train_window: int`  
Fixed training-window length (`>= 1`).

`test_window: int`  
Fixed test-window length (`>= 1`).

`guide: Guide`  
Any guide for the model object `model_fn` returns (see [Guide](typing.Guide.md#numpyro_forecast.typing.Guide)): an autoguide instance built on that same object, sampled through its `sample_posterior`, or a hand-written guide function with the model's signature, sampled through `numpyro.infer.Predictive(guide, params=...)` on each window's training data:

``` python
model = make_model()
backtest_vectorized(..., model_fn=lambda: model, guide=AutoNormal(model))
backtest_vectorized(..., model_fn=lambda: model, guide=my_guide_fn)
```

An autoguide must be a fresh instance that was never run or fitted: each window fits its own copy of it, and the instance you pass is left unchanged, so it can be passed again to a later call. Its `init_loc_fn` runs inside the vmapped fit, so it must be traceable by JAX (no NumPy conversion of the values it receives). The copy per window applies only when `guide` itself is an autoguide: an autoguide behind a handler (`handlers.scale(AutoNormal(model), ...)`) or called from a guide function is shared by all windows and set up once, on window 0, so its supports and Laplace covariance come from window 0.

`optim: _NumPyroOptim | None = None`  
NumPyro optimizer; `None` uses `numpyro.optim.Adam(0.01)`. For an optax optimizer, wrap it with `numpyro.optim.optax_to_numpyro` first.

`stride: int = ``1`  
Step between successive windows (`>= 1`).

`num_steps: int = ``1001`  
Number of SVI steps per window.

`num_samples: int = ``100`  
Number of forecast samples drawn per window. Must be positive, and at least 2 when `metrics` contains [eval_crps()](evaluate.eval_crps.md#numpyro_forecast.evaluate.eval_crps) (as [DEFAULT_METRICS](evaluate.DEFAULT_METRICS.md#numpyro_forecast.evaluate.DEFAULT_METRICS) does) or a `functools.partial()` of it. Both rules are checked before the fit. A CRPS inside another wrapper (a `lambda` or a `jax.jit` function) is not checked up front: it fails only after the fit.

`metrics: Mapping[str, Metric] | None = None`  
Mapping of metric name to function; defaults to [DEFAULT_METRICS](evaluate.DEFAULT_METRICS.md#numpyro_forecast.evaluate.DEFAULT_METRICS). Each metric is vmapped over the window axis, so any pure-JAX mapping, including partial-bound variants such as `{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.5)}`, is scored inside the single fused computation. Host-side metrics are unsupported here; use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest), or `keep_predictions=True` and score on the host.

`keep_predictions: bool = ``False`  
If `True`, retain the stacked forecast samples on the result.


## Returns


`VectorizedBacktestResult`  
The stacked per-window losses, metrics, and window indices.


## Raises


`ValueError`  
If `data` or `covariates` has fewer than two axes (time must be at axis `-2`), if their durations differ, if `num_samples` is too small (see `num_samples`), if `guide` is an autoguide that is already set up (see `guide`), or if `metrics` is not empty and the model's `"forecast"` site does not have the shape of `data` over the test window.

`BacktestWindowError`  
If `train_window`, `test_window`, or `stride` is `< 1`, or there is no room for a single window.

`RuntimeError`  
If the SVI loss of a window is not finite at the first step (no valid start on that window), after the fit and before the forecasts. When an eager init of the first such window gives NumPyro's own error (for example no valid initial parameters), that error is raised instead; an eager init of window 0 before the fit raises it for window 0.

`VectorizedMetricError`  
If a metric is not a pure JAX function under `vmap`: it forces a host conversion (`float(...)`, `int(...)`, `range(...)`, Python list indexing or a NumPy call on a traced value) or builds a value-dependent shape (a boolean mask index such as `x[x > 0]`).
