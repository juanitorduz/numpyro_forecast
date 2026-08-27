# NumPyro Forecast

[![PyPI version](https://img.shields.io/pypi/v/numpyro_forecast.svg)](https://pypi.org/project/numpyro_forecast/) [![ci](https://github.com/juanitorduz/numpyro_forecast/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/juanitorduz/numpyro_forecast/actions/workflows/ci.yml?query=branch%3Amain) [![docs](https://github.com/juanitorduz/numpyro_forecast/actions/workflows/docs.yml/badge.svg?branch=main)](https://juanitorduz.github.io/numpyro_forecast/) [![codecov](https://codecov.io/gh/juanitorduz/numpyro_forecast/branch/main/graph/badge.svg)](https://codecov.io/gh/juanitorduz/numpyro_forecast) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A JAX/NumPyro port of the ideas in Pyro's forecasting module.

📖 **Documentation:** <https://juanitorduz.github.io/numpyro_forecast/>

This is a **conceptual** port of [`pyro.contrib.forecast`](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast), not a line-by-line one. Pyro's module is class-based (`ForecastingModel`, `Forecaster`, `HMCForecaster`), while JAX and NumPyro follow the functional paradigm: pure functions, explicit `PRNGKey`s, no global parameter store. So `numpyro_forecast` keeps Pyro's ideas (the `_future`-site trick, horizon bookkeeping derived from shapes, prefix conditioning, backtesting) and expresses them as plain functions: a model is a function `(covariates, data=None)` built from model building blocks, and inference is whatever NumPyro (`SVI`, `MCMC`) or BlackJAX you write.

**Related project:** [PyMC-Forecast](https://github.com/pymc-labs/pymc-forecast) carries the same ideas (train/forecast plumbing, backtesting, evaluation) over to PyMC with a class-based API (`Forecaster`, `HMCForecaster`, `StatespaceForecaster`). It is in early development; pick it when your models live in PyMC.

## Scope

You write the generative model as a plain NumPyro model function; the package handles the train/forecast plumbing, memory-bounded prediction, and evaluation. Inference is plain NumPyro.

- **A single model both trains and forecasts.** In-sample time latents use a fixed site name (`drift`); the forecast horizon uses a separate `drift_future` site that the guide never sees, because fitting always happens at `future == 0`, so `Predictive` draws the suffix from the prior. The horizon itself is derived from shapes: `covariates` longer than `data`. The drivers (`forecast`, `predict_in_sample`, `to_datatree` and the `backtest` closures) read the `"forecast"` and `"obs"` sites by name, and nothing in the package matches on the `_future` suffix, which is why those two names must stay unscoped.
- **Inference is yours.** `SVI` with any autoguide, `MCMC` with any kernel (including the BlackJAX kernels in `numpyro_forecast.contrib.blackjax`, which are plain `MCMCKernel`s you hand to `MCMC`), or Pathfinder via `fit_multipathfinder` plus `multipathfinder_samples`. Nothing in the package wraps `svi.run` or `mcmc.run`.
- **Backtesting** over rolling and expanding windows, plus probabilistic and point metrics. Windows are embarrassingly parallel at the process level; the package ships no joblib or multiprocessing layer of its own.
- **Univariate, multivariate and hierarchical** models.

Arrays follow Pyro's layout: **time at axis `-2`**, the observation/event dimension at `-1`, and batch dimensions to the left.

It is **not** an AutoML or "fit-any-series" library: there is no model zoo and no automatic feature pipeline. You define the NumPyro model; the package gives you a clean path from model to forecasts and scores.

## Installation

Requires Python >= 3.12. Install from PyPI:

```bash
uv add numpyro_forecast
# or, with pip:
pip install numpyro_forecast
```

To install the latest development version from source:

```bash
uv add "numpyro_forecast @ git+https://github.com/juanitorduz/numpyro_forecast"
# or, with pip:
pip install "numpyro_forecast @ git+https://github.com/juanitorduz/numpyro_forecast"
```

For a local checkout:

```bash
uv sync --extra all
```

The optional extras are `dataframes` (pandas and polars, so `results_to_dataframe` can flatten backtest results), `optax` (optax optimizers, wrapped for SVI with `numpyro.optim.optax_to_numpyro`) and `blackjax` (the BlackJAX kernels and Pathfinder in `numpyro_forecast.contrib.blackjax`).

## Quickstart

Define a model, fit it with SVI, and draw probabilistic forecasts:

```python
>>> import jax.numpy as jnp
>>> import numpyro
>>> import numpyro.distributions as dist
>>> from jax import random
>>> from numpyro.infer import SVI, Trace_ELBO
>>> from numpyro.infer.autoguide import AutoNormal
>>> from numpyro.infer.reparam import LocScaleReparam
>>> from numpyro.optim import Adam
>>> from numpyro_forecast import Horizon, draw_posterior, eval_crps, forecast, innovations
>>> from numpyro_forecast import predict
>>> from numpyro_forecast.features import fourier_features
>>> def seasonal_model(covariates, data=None):
...     """Local-level random walk + Fourier seasonality, Student-T noise."""
...     h = Horizon.from_data(covariates, data)
...     num_features = covariates.shape[-1]
...     bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
...     weight = numpyro.sample(
...         "weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1)
...     )
...     drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-3.0, 1.0))
...     sigma = numpyro.sample("sigma", dist.LogNormal(-2.0, 1.0))
...     nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
...     # In-sample innovations at "drift", the forecast suffix at "drift_future".
...     drift = innovations(
...         h, "drift", lambda: dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0)
...     )
...     level = jnp.cumsum(drift, axis=-2)  # random-walk level
...     regression = (weight * covariates).sum(axis=-1, keepdims=True)
...     prediction = level + bias + regression
...     # Registers "obs" over the training window and "forecast" over the horizon.
...     predict(h, dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)
>>> # Synthetic weekly-seasonal series: time at axis -2, one observation dim at -1.
>>> period, t_obs, horizon = 52.0, 156, 26
>>> duration = t_obs + horizon
>>> covariates = fourier_features(duration, period=period, num_terms=3)
>>> t = jnp.arange(duration)[:, None]
>>> truth = jnp.sin(2 * jnp.pi * t / period) + 0.01 * t
>>> data = truth[:t_obs]
>>> key_fit, key_post, key_pred = random.split(random.PRNGKey(0), 3)
>>> guide = AutoNormal(seasonal_model)
>>> svi = SVI(seasonal_model, guide, Adam(step_size=0.01), Trace_ELBO())
>>> svi_result = svi.run(key_fit, 1_500, covariates[:t_obs], data, progress_bar=False)
>>> posterior = draw_posterior(key_post, guide, svi_result.params, num_samples=100)
>>> samples = forecast(key_pred, seasonal_model, posterior, data, covariates)
>>> samples.shape  # (draws, horizon, obs)
(100, 26, 1)
>>> bool(eval_crps(samples, truth[t_obs:]) < 1.0)
True

```

`samples` holds the forecast draws over the held-out horizon, shaped `(sample, *batch, future, obs)`: one row per posterior draw, ready for `eval_crps` and the rest of the evaluation helpers.

The examples on this page are executed by the test suite (`pytest --doctest-glob=README.md`), so they are always current.

## Model building blocks

A model is a plain NumPyro function `(covariates, data=None)` whose first line derives its `Horizon` from the shapes. The building blocks below are ordinary Python functions that call `numpyro.sample` and `numpyro.deterministic` on your behalf against that horizon.

| Building block | What it replaces in raw NumPyro | Sites it registers |
| --- | --- | --- |
| `Horizon.from_data(covariates, data)` | Deriving the train/forecast split by hand and carrying `t_obs`, `future` and `duration` around | none |
| `innovations(h, name, dist_fn)` | Two `numpyro.sample` calls under two time plates, plus the concatenation of prefix and suffix | `<name>`, `<name>_future` |
| `markov_series(h, name, init_carry, transition)` | Two `numpyro.contrib.control_flow.scan` calls, the second seeded by the first's final carry | `<name>`, `<name>_future` |
| `ssoe(h, name, y, init_carry, step, noise_dist)` | An in-sample error-feedback `lax.scan` filter, plus a generative forecast scan driven by iid future errors | `<name>_future` only |
| `predict(h, obs_dist, prediction)` | Slicing the observation distribution along time, conditioning it on the observed prefix and sampling the suffix | `obs` while training; also `obs_future` and the `forecast` deterministic when `h.future > 0` |

`ssoe` is the one block that does not close the loop for you: it registers only the error site, and the caller writes the likelihood against `r.mu` and registers `numpyro.deterministic("forecast", r.y_future)` when `h.future > 0`. In every block the observed data flows in through the `Horizon`, so `predict` has no `obs=` argument of its own.

The three latent blocks differ in where the sampling happens. `innovations` samples conditionally iid per-step innovations outside any loop, and you build the series arithmetically from them (a random walk is `jnp.cumsum(drift, axis=-2)`). `ssoe` is an iid error plate plus a deterministic scan that consumes those errors, the single-source-of-error form behind ARMA, exponential smoothing and Croston/TSB. `markov_series` samples inside `numpyro.contrib.control_flow.scan`, one step at a time, which is what you need when the per-step distribution depends on the previous state.

Reuse a group of sites across channels with NumPyro's `handlers.scope`, with one caveat: `scope` prefixes every site inside it, `obs`, `obs_future` and `forecast` included, after which the drivers can no longer find `"forecast"` and `"obs"`. Scope the latent helpers and call `predict` outside the scope (the Croston and TSB examples do exactly this). `scope` is the composition tool; it is not a replacement for the `_future` suffix, which is what keeps the guide's shape fixed.

Dimensions beyond `(time, obs)` stack leftward, and the two loop-shaped blocks take opposite approaches to plates. `innovations` is called inside the plates you open yourself: the [multi-series example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/hierarchical_forecasting_1.html) wraps it in `plate("n_series", n_series, dim=-1)`, and the [hierarchical origin-destination example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/hierarchical_forecasting_2.html) opens `origin` at `dim=-3` and `destin` at `dim=-1` and puts the call inside the latter. `markov_series` flips the idiom: it rejects an enclosing plate and takes `plates=[(name, size)]`, which it opens inside the scan body, the only placement NumPyro supports for scan plus plate.

## Producing draws and scoring

`draw_posterior(rng_key, guide, params, num_samples)` draws posterior samples of the latent sites from a fitted variational guide and its learned parameters. It is guide-only on purpose: MCMC users already hold their draws through `mcmc.get_samples()`, and the BlackJAX Pathfinder backend has its own analogous entry points, `pathfinder_samples` and `multipathfinder_samples`.

`forecast(rng_key, model, posterior, data, covariates)` runs `Predictive` over the full horizon and returns the `"forecast"` site, shaped `(sample, *batch, future, obs)`. The horizon is whatever `covariates` has beyond `data`.

`predict_in_sample(rng_key, model, posterior, covariates)` samples the in-sample posterior predictive of the `"obs"` site. It calls the model with `data=None`, so anything the model needs at prediction time has to travel through `covariates`.

`to_datatree(rng_key, model, posterior, data, covariates)` converts an already-drawn posterior into an ArviZ-schema `xarray.DataTree`: posterior, in-sample posterior predictive, observed data and covariates in one object, plus the forecast groups when `covariates` extends past `data`. It is posterior-first and never draws a posterior of its own.

`backtest(rng_key, data, covariates, model_fn, forecast_fn=...)` runs the moving-window loop and scores every window. Fitting and forecasting are delegated to closures you write, so `backtest` itself has no dependency on how a model is fit; `backtest_vectorized` is the estimator-equivalent shortcut that fits every rolling window in one vmapped SVI run.

On an accelerator, the draws are usually the largest allocation of the workflow, so every driver that materializes them takes `batch_size` (chunk the sample axis) and `device` (move each chunk off the accelerator as it is drawn, `device="host"` being the useful setting). This is spelled `batch_size`/`device` on `draw_posterior`, `forecast`, `predict_in_sample` and the Pathfinder samplers, `predictive_batch_size`/`predictive_device` on `to_datatree`, and `batch_size` on `backtest`, which forwards it to your closures. The [stockout example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/fresh_retail_stockout.html) walks through a panel where this matters.

The same model, fitted with NUTS instead of SVI:

```python
>>> from numpyro.infer import MCMC, NUTS
>>> mcmc = MCMC(
...     NUTS(seasonal_model), num_warmup=500, num_samples=500, num_chains=1, progress_bar=False
... )
>>> mcmc.run(key_fit, covariates[:t_obs], data)
>>> samples = forecast(key_pred, seasonal_model, mcmc.get_samples(), data, covariates)
>>> samples.shape
(500, 26, 1)

```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment management, [ruff](https://docs.astral.sh/ruff/) for linting/formatting, [ty](https://github.com/astral-sh/ty) for type checking, and [prek](https://github.com/j178/prek) to run the pre-commit hooks.

```bash
uv sync --extra all        # create the environment
prek install               # install git hooks
prek run --all-files       # lint + format + type check
uv run pytest              # run the tests (README examples included)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow and guidelines.

## License

Apache-2.0.
