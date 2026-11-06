"""numpyro_forecast: a JAX/NumPyro port of the ideas in Pyro's forecasting module."""

from importlib.metadata import PackageNotFoundError, version

from jaxtyping import install_import_hook

# Imported before the hook: the hook resolves its typechecker by attribute access on the
# package, and the module holds only the beartype configuration (see its docstring).
import numpyro_forecast.typecheck

with install_import_hook("numpyro_forecast", "numpyro_forecast.typecheck.typechecker"):
    # Bound here so great-docs can resolve the `contrib.blackjax.*` and `contrib.dynestyx.*`
    # reference entries by attribute walk (API reference, llms-full.txt); the modules import
    # no optional extra. They stay inside the hook so they are runtime type-checked like the
    # core modules.
    import numpyro_forecast.contrib.blackjax
    import numpyro_forecast.contrib.dynestyx  # noqa: F401
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
        priors,
        reparam,
        surgery,
        var,
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
    DeviceMemoryError,
    DevicePlatformError,
    HostMemoryKindError,
    KernelConfigError,
    MVNLayoutError,
    NumpyroForecastError,
    VectorizedMetricError,
)
from numpyro_forecast.models import (
    Advance,
    Horizon,
    SSOEMean,
    SSOEResult,
    SSOEUpdate,
    Transition,
    innovations,
    markov_series,
    predict,
    ssoe,
)
from numpyro_forecast.predictive import draw_posterior, forecast, predict_in_sample
from numpyro_forecast.reparam import time_reparam
from numpyro_forecast.surgery import register_elementwise

try:
    __version__ = version("numpyro_forecast")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "DEFAULT_METRICS",
    "Advance",
    "BacktestResult",
    "BacktestWindowError",
    "CovariateDimsError",
    "DeviceMemoryError",
    "DevicePlatformError",
    "Horizon",
    "HostMemoryKindError",
    "KernelConfigError",
    "MVNLayoutError",
    "NumpyroForecastError",
    "SSOEMean",
    "SSOEResult",
    "SSOEUpdate",
    "Transition",
    "VectorizedBacktestResult",
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
    "innovations",
    "markov_series",
    "predict",
    "predict_in_sample",
    "predictions_to_datatree",
    "register_elementwise",
    "results_to_dataframe",
    "ssoe",
    "time_reparam",
    "to_datatree",
]
