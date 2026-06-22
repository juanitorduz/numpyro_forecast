# NumPyro Forecast

![Build](https://github.com/juanitorduz/numpyro_forecast/workflows/ci/badge.svg) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A JAX/NumPyro port of [Pyro's forecasting module](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast).

## Scope

`numpyro_forecast` is a small, focused toolkit for **Bayesian time-series
forecasting** with NumPyro. You write the generative model; the package handles
the train/forecast plumbing, inference, and evaluation:

- A single model both trains and forecasts. In-sample time latents use a fixed
  site name (`drift`); the forecast horizon uses a separate `_future` site so the
  variational guide is never resized and the forecast suffix is drawn from the
  prior. The horizon is inferred from shapes, `covariates` longer than `data`.
- Two inference backends: stochastic variational inference (`Forecaster`, via
  `AutoNormal`) and Hamiltonian Monte Carlo / NUTS (`HMCForecaster`).
- Backtesting over rolling windows plus probabilistic and point metrics.
- Works for univariate, multivariate, and hierarchical models.

Arrays follow Pyro's layout: **time at axis `-2`**, the observation/event
dimension at `-1`, and batch dimensions to the left.

It is **not** an AutoML or "fit-any-series" library — there are no pre-built
model zoo or automatic feature pipelines. You define the NumPyro model; the
package gives you a clean path from model to forecasts and scores.

## Installation

Requires Python >= 3.12. The package is not yet published on PyPI; install it
from source:

```bash
uv add "numpyro_forecast @ git+https://github.com/juanitorduz/numpyro_forecast"
# or, with pip:
pip install "numpyro_forecast @ git+https://github.com/juanitorduz/numpyro_forecast"
```

For a local checkout:

```bash
uv sync --all-extras   # or: pip install -e ".[dataframes]"
```

The optional `dataframes` extra adds `pandas`/`polars` support.

## Quickstart

Define a model, fit it with SVI, and draw probabilistic forecasts:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.evaluate import eval_crps
from numpyro_forecast.forecaster import Forecaster, ForecastingModel
from numpyro_forecast.util import fourier_features


class SeasonalForecaster(ForecastingModel):
    """Local-level random walk + Fourier seasonality, Student-T noise."""

    def model(self, zero_data, covariates):
        num_features = covariates.shape[-1]
        bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
        weight = numpyro.sample(
            "weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1)
        )
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-3.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-2.0, 1.0))
        nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))

        drift = self.time_series(
            "drift",
            lambda: dist.Normal(0.0, drift_scale),
            reparam=LocScaleReparam(0),
        )
        level = jnp.cumsum(drift, axis=-2)  # random-walk level
        regression = (weight * covariates).sum(axis=-1, keepdims=True)
        prediction = level + bias + regression

        self.predict(dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)


# Synthetic weekly-seasonal series: time at axis -2, one observation dim at -1.
period, t_obs, horizon = 52.0, 156, 26
duration = t_obs + horizon
covariates = fourier_features(duration, period=period, num_terms=3)
t = jnp.arange(duration)[:, None]
truth = jnp.sin(2 * jnp.pi * t / period) + 0.01 * t
data = truth[:t_obs]

key_fit, key_pred = random.split(random.PRNGKey(0))
forecaster = Forecaster(
    key_fit,
    SeasonalForecaster(),
    data,
    covariates[:t_obs],
    num_steps=1_500,
)

# Draw 100 forecast samples over the held-out horizon, shaped (sample, future, obs).
samples = forecaster(key_pred, data, covariates, num_samples=100)
print("forecast samples:", samples.shape)
print("CRPS:", eval_crps(samples, truth[t_obs:]))
```

> **`rng_key` comes first.** Following the JAX/NumPyro convention, every function
> that consumes randomness (`Forecaster`, `HMCForecaster`, `fit_svi`, `fit_mcmc`,
> `forecast`, `draw_posterior`, `backtest`, ...) takes the `PRNGKey` as its first
> argument.

## Two APIs: functional core and OOP shim

The package is built around a **pure functional core** (`numpyro_forecast.functional`)
and a thin **object-oriented shim** (`numpyro_forecast.forecaster`) that ports Pyro's
class-based API. The two are fully interchangeable: both produce the same NumPyro
model callable `(covariates, data=None)` and consume the same posterior dict of latent
draws, so you can fit with one and forecast with the other.

- **Functional core.** The train/forecast split is an explicit, immutable
  `Horizon` value (derived from the covariate and data shapes) that is threaded
  into pure primitives. You write a model body `(Horizon, covariates) -> None`
  that calls `time_series(...)` and `predict(...)`, wrap it with
  `forecasting_model(...)`, and drive inference with the free functions
  `fit_svi` / `fit_mcmc`, `draw_posterior`, and `forecast`. No global parameter
  store, explicit `PRNGKey` threading.
- **OOP shim (Pyro-compatible).** Subclass `ForecastingModel` and implement
  `model(self, zero_data, covariates)`, calling `self.time_series(...)` and
  `self.predict(...)` exactly as in Pyro's `pyro.contrib.forecast`. The
  `Forecaster` (SVI) and `HMCForecaster` (NUTS) classes carry the horizon as
  instance state and delegate to the functional core under the hood. This is the
  API used in the [Quickstart](#quickstart) above.

The same `SeasonalForecaster` model, written and run through the functional API:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.evaluate import eval_crps
from numpyro_forecast.functional import (
    Horizon,
    draw_posterior,
    fit_svi,
    forecast,
    forecasting_model,
    predict,
    time_series,
)
from numpyro_forecast.util import fourier_features


def seasonal_body(h: Horizon, covariates):
    """Local-level random walk + Fourier seasonality, Student-T noise."""
    num_features = covariates.shape[-1]
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1)
    )
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-3.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-2.0, 1.0))
    nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))

    drift = time_series(
        h, "drift", lambda: dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0)
    )
    level = jnp.cumsum(drift, axis=-2)  # random-walk level
    regression = (weight * covariates).sum(axis=-1, keepdims=True)
    prediction = level + bias + regression

    predict(h, dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)


# Same synthetic series as the Quickstart.
period, t_obs, horizon = 52.0, 156, 26
duration = t_obs + horizon
covariates = fourier_features(duration, period=period, num_terms=3)
t = jnp.arange(duration)[:, None]
truth = jnp.sin(2 * jnp.pi * t / period) + 0.01 * t
data = truth[:t_obs]

model = forecasting_model(seasonal_body)
key_fit, key_post, key_pred = random.split(random.PRNGKey(0), 3)

fit = fit_svi(key_fit, model, data, covariates[:t_obs], num_steps=1_500)
posterior = draw_posterior(key_post, fit, num_samples=100)
samples = forecast(key_pred, model, posterior, data, covariates)
print("forecast samples:", samples.shape)
print("CRPS:", eval_crps(samples, truth[t_obs:]))
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment management,
[ruff](https://docs.astral.sh/ruff/) for linting/formatting,
[ty](https://github.com/astral-sh/ty) for type checking, and
[prek](https://github.com/j178/prek) to run the pre-commit hooks.

```bash
uv sync --all-extras       # create the environment
prek install               # install git hooks
prek run --all-files       # lint + format + type check
uv run pytest              # run the tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow and guidelines.

## License

Apache-2.0.
