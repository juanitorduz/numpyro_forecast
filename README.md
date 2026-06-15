# NumPyro Forecast

A JAX/NumPyro port of [Pyro's forecasting module](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast).

`numpyro_forecast` keeps Pyro's familiar API:`ForecastingModel`, `Forecaster`,
`HMCForecaster`, `backtest`, and the `eval_crps` / `eval_mae` / `eval_rmse`
metrics, while embracing the functional style of JAX and NumPyro
(`jax.lax.scan`, explicit `PRNGKey` threading, `Predictive`, no global parameter
store).

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

## What's included

- **`ForecastingModel`**: abstract base class. Subclass it and implement
  `model(self, zero_data, covariates)`, calling `self.time_series(...)` for latent
  random walks and `self.predict(noise_dist, prediction)` exactly once.
- **`Forecaster`**: fit a model with SVI (`AutoNormal` guide by default).
- **`HMCForecaster`**: fit a model with NUTS.
- **`backtest`** / **`BacktestResult`**: evaluate forecasts over rolling windows.
- **Metrics**: `eval_crps`, `eval_mae`, `eval_rmse`, `DEFAULT_METRICS`, and the
  lower-level `crps_empirical`.
- **Seasonality helpers**: `fourier_features`, `periodic_repeat`,
  `prefix_condition`, and friends in `numpyro_forecast.util`.
- **Example datasets**: `datasets.load_bart_weekly` and
  `datasets.load_bart_hierarchical` (BART ridership).

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

from numpyro_forecast import Forecaster, ForecastingModel, eval_crps
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
    SeasonalForecaster(),
    data,
    covariates[:t_obs],
    num_steps=1500,
    rng_key=key_fit,
)

# Draw 100 forecast samples over the held-out horizon, shaped (sample, future, obs).
samples = forecaster(data, covariates, num_samples=100, rng_key=key_pred)
print("forecast samples:", samples.shape)
print("CRPS:", eval_crps(samples, truth[t_obs:]))
```

## Status

Early development (alpha). The public API mirrors Pyro's `pyro.contrib.forecast`.
For design context, read the module docstrings (`forecaster.py`, `evaluate.py`,
`util.py`) and the example notebooks. Contributor conventions live in
[`AGENTS.md`](AGENTS.md).

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
