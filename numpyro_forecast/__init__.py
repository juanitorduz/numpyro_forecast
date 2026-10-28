"""numpyro_forecast: a JAX/NumPyro port of the ideas in Pyro's forecasting module."""

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
        priors,
        reparam,
        stacking,
        surgery,
        var,
    )

# Bound here so great-docs can resolve the `contrib.blackjax.*` reference entries by
# attribute walk (API reference, llms-full.txt); the module imports no optional extra.
import numpyro_forecast.contrib.blackjax  # noqa: F401
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
    DensitySiteError,
    DeviceMemoryError,
    DevicePlatformError,
    HostMemoryKindError,
    KernelConfigError,
    MVNLayoutError,
    NumpyroForecastError,
    StackingShapeError,
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
from numpyro_forecast.predictive import (
    conditional_log_density,
    draw_posterior,
    forecast,
    forecast_log_density,
    log_density_ess,
    predict_density,
    predict_in_sample,
    reduce_log_density,
    ssoe_density,
)
from numpyro_forecast.reparam import time_reparam
from numpyro_forecast.stacking import (
    StackingCells,
    crps_stacking_weights,
    hierarchical_weights,
    log_score,
    mixture_crps_terms,
    pseudo_bma_weights,
    rectified_features,
    stack_forecasts,
    stacking_cells,
    stacking_factor,
    stacking_weights,
    window_design_effect,
)
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
    "DensitySiteError",
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
    "StackingCells",
    "StackingShapeError",
    "Transition",
    "VectorizedBacktestResult",
    "VectorizedMetricError",
    "__version__",
    "add_forecast_groups",
    "backtest",
    "backtest_vectorized",
    "conditional_log_density",
    "crps_stacking_weights",
    "draw_posterior",
    "eval_coverage",
    "eval_crps",
    "eval_mae",
    "eval_rmse",
    "evaluate_forecast",
    "forecast",
    "forecast_log_density",
    "hierarchical_weights",
    "innovations",
    "log_density_ess",
    "log_score",
    "markov_series",
    "mixture_crps_terms",
    "predict",
    "predict_density",
    "predict_in_sample",
    "predictions_to_datatree",
    "pseudo_bma_weights",
    "rectified_features",
    "reduce_log_density",
    "register_elementwise",
    "results_to_dataframe",
    "ssoe",
    "ssoe_density",
    "stack_forecasts",
    "stacking_cells",
    "stacking_factor",
    "stacking_weights",
    "time_reparam",
    "to_datatree",
    "window_design_effect",
]
