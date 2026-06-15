# numpyro_forecast

A JAX/NumPyro port of [Pyro's forecasting module](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/forecast).

`numpyro_forecast` keeps Pyro's familiar API — `ForecastingModel`, `Forecaster`,
`HMCForecaster`, `backtest`, and the `eval_crps` / `eval_mae` / `eval_rmse`
metrics — while embracing the functional style of JAX and NumPyro
(`jax.lax.scan`, explicit `PRNGKey` threading, `Predictive`, no global parameter
store).

## Status

Early development. See [`implementation.md`](implementation.md) for the design
and the staged build plan.

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

## License

Apache-2.0.
