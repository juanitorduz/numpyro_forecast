"""numpyro_forecast: a JAX/NumPyro port of Pyro's forecasting module."""

from importlib.metadata import PackageNotFoundError, version

from jaxtyping import install_import_hook

with install_import_hook("numpyro_forecast", "beartype.beartype"):
    from numpyro_forecast import (  # noqa: F401
        datasets,
        evaluate,
        forecaster,
        functional,
        metrics,
        util,
    )

from numpyro_forecast.evaluate import (
    BacktestResult,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
)
from numpyro_forecast.forecaster import Forecaster, ForecastingModel, HMCForecaster
from numpyro_forecast.functional import forecasting_model

try:
    __version__ = version("numpyro_forecast")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "BacktestResult",
    "Forecaster",
    "ForecastingModel",
    "HMCForecaster",
    "__version__",
    "backtest",
    "eval_coverage",
    "eval_crps",
    "eval_mae",
    "eval_rmse",
    "evaluate_forecast",
    "forecasting_model",
]
