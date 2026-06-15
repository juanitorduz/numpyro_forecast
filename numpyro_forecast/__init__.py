"""numpyro_forecast: a JAX/NumPyro port of Pyro's forecasting module."""

from jaxtyping import install_import_hook

with install_import_hook("numpyro_forecast", "beartype.beartype"):
    from numpyro_forecast import datasets
    from numpyro_forecast.evaluate import (
        DEFAULT_METRICS,
        BacktestResult,
        backtest,
        eval_crps,
        eval_mae,
        eval_rmse,
    )
    from numpyro_forecast.forecaster import Forecaster, ForecastingModel, HMCForecaster
    from numpyro_forecast.metrics import crps_empirical
    from numpyro_forecast.util import prefix_condition

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_METRICS",
    "BacktestResult",
    "Forecaster",
    "ForecastingModel",
    "HMCForecaster",
    "__version__",
    "backtest",
    "crps_empirical",
    "datasets",
    "eval_crps",
    "eval_mae",
    "eval_rmse",
    "prefix_condition",
]
