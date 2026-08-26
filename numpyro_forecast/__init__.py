"""numpyro_forecast: a JAX/NumPyro port of Pyro's forecasting module."""

from importlib.metadata import PackageNotFoundError, version

from jaxtyping import install_import_hook

with install_import_hook("numpyro_forecast", "beartype.beartype"):
    from numpyro_forecast import (  # noqa: F401
        acf,
        arrays,
        convert,
        datasets,
        evaluate,
        exceptions,
        features,
        metrics,
        models,
        optional,
        predictive,
        surgery,
    )

from numpyro_forecast.convert import add_forecast_groups, predictions_to_datatree, to_datatree
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
    CovariateDimsError,
    KernelConfigError,
    MVNLayoutError,
    NumpyroForecastError,
    VectorizedGuideError,
    VectorizedMetricError,
)
from numpyro_forecast.models import (
    Horizon,
    Transition,
    markov_time_series,
    predict,
    predict_glm,
    time_series,
)
from numpyro_forecast.predictive import draw_posterior, forecast, predict_in_sample
from numpyro_forecast.surgery import register_elementwise

try:
    __version__ = version("numpyro_forecast")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEFAULT_METRICS",
    "BacktestResult",
    "BacktestWindowError",
    "CovariateDimsError",
    "Horizon",
    "KernelConfigError",
    "MVNLayoutError",
    "NumpyroForecastError",
    "Transition",
    "VectorizedBacktestResult",
    "VectorizedGuideError",
    "VectorizedMetricError",
    "__version__",
    "add_forecast_groups",
    "backtest",
    "backtest_vectorized",
    "draw_posterior",
    "eval_coverage",
    "eval_crps",
    "eval_mae",
    "eval_rmse",
    "evaluate_forecast",
    "forecast",
    "markov_time_series",
    "predict",
    "predict_glm",
    "predict_in_sample",
    "predictions_to_datatree",
    "register_elementwise",
    "results_to_dataframe",
    "time_series",
    "to_datatree",
]
