"""numpyro_forecast: a JAX/NumPyro port of Pyro's forecasting module."""

from importlib.metadata import PackageNotFoundError, version

from jaxtyping import install_import_hook

with install_import_hook("numpyro_forecast", "beartype.beartype"):
    from numpyro_forecast import (  # noqa: F401
        convert,
        datasets,
        evaluate,
        forecaster,
        functional,
        metrics,
        util,
    )

from numpyro_forecast.convert import add_forecast, to_datatree
from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    BacktestResult,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
    results_to_dataframe,
)
from numpyro_forecast.forecaster import (
    Forecaster,
    ForecastingModel,
    HMCForecaster,
    PathfinderForecaster,
)
from numpyro_forecast.functional import forecasting_model
from numpyro_forecast.util import register_elementwise

try:
    __version__ = version("numpyro_forecast")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEFAULT_METRICS",
    "BacktestResult",
    "Forecaster",
    "ForecastingModel",
    "HMCForecaster",
    "PathfinderForecaster",
    "__version__",
    "add_forecast",
    "backtest",
    "eval_coverage",
    "eval_crps",
    "eval_mae",
    "eval_rmse",
    "evaluate_forecast",
    "forecasting_model",
    "register_elementwise",
    "results_to_dataframe",
    "to_datatree",
]
