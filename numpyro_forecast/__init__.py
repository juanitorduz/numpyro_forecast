"""numpyro_forecast: a JAX/NumPyro port of Pyro's forecasting module."""

from importlib.metadata import PackageNotFoundError, version

from jaxtyping import install_import_hook

with install_import_hook("numpyro_forecast", "beartype.beartype"):
    from numpyro_forecast import (  # noqa: F401
        arrays,
        convert,
        datasets,
        evaluate,
        exceptions,
        features,
        forecaster,
        functional,
        metrics,
        optional,
        surgery,
    )

from numpyro_forecast.convert import add_forecast_groups, to_datatree
from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    BacktestResult,
    VectorizedBacktestResult,
    backtest,
    backtest_vectorized,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
    results_to_dataframe,
)
from numpyro_forecast.exceptions import (
    BacktestWindowError,
    GuideResolutionError,
    GuideSampleArgsError,
    KernelConfigError,
    KernelResolutionError,
    MVNLayoutError,
    NumpyroForecastError,
    OptimizerResolutionError,
    VectorizedGuideError,
    VectorizedMetricError,
)
from numpyro_forecast.forecaster import (
    Forecaster,
    ForecastingModel,
    HMCForecaster,
    PathfinderForecaster,
)
from numpyro_forecast.functional import forecasting_model
from numpyro_forecast.surgery import register_elementwise

try:
    __version__ = version("numpyro_forecast")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEFAULT_METRICS",
    "BacktestResult",
    "BacktestWindowError",
    "Forecaster",
    "ForecastingModel",
    "GuideResolutionError",
    "GuideSampleArgsError",
    "HMCForecaster",
    "KernelConfigError",
    "KernelResolutionError",
    "MVNLayoutError",
    "NumpyroForecastError",
    "OptimizerResolutionError",
    "PathfinderForecaster",
    "VectorizedBacktestResult",
    "VectorizedGuideError",
    "VectorizedMetricError",
    "__version__",
    "add_forecast_groups",
    "backtest",
    "backtest_vectorized",
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
