"""Smoke tests for package import and metadata."""

import subprocess
import sys

import jax
import jax.numpy as jnp

import numpyro_forecast
from numpyro_forecast import (
    DEFAULT_METRICS,
    BacktestResult,
    Forecaster,
    ForecastingModel,
    HMCForecaster,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
    forecasting_model,
)


def test_version() -> None:
    assert isinstance(numpyro_forecast.__version__, str)
    assert numpyro_forecast.__version__


def test_public_api_is_importable() -> None:
    # The curated top-level surface re-exported in ``__init__``.
    assert DEFAULT_METRICS  # the metrics mapping, a dict (unhashable, kept out of the set below)
    exported = {
        BacktestResult,
        Forecaster,
        ForecastingModel,
        HMCForecaster,
        backtest,
        eval_coverage,
        eval_crps,
        eval_mae,
        eval_rmse,
        evaluate_forecast,
        forecasting_model,
    }
    assert all(obj is not None for obj in exported)


def test_all_matches_exported_names() -> None:
    names = set(numpyro_forecast.__all__)
    assert names == {
        "DEFAULT_METRICS",
        "BacktestResult",
        "BacktestWindowError",
        "CovariateDimsError",
        "Forecaster",
        "ForecastingModel",
        "GuideResolutionError",
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
        "predictions_to_datatree",
        "register_elementwise",
        "results_to_dataframe",
        "to_datatree",
    }
    for name in names:
        assert hasattr(numpyro_forecast, name)


def test_base_import_no_extras() -> None:
    """Importing the package must not pull in optional deps (invariant I8).

    Runs in a fresh subprocess so a test that imported ``optax``/``blackjax``
    earlier cannot mask a real leak via ``sys.modules``.
    """
    code = (
        "import sys; import numpyro_forecast; "
        "leaked = [m for m in ('optax', 'blackjax') if m in sys.modules]; "
        "assert not leaked, leaked; print('OK')"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, trusted input
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_compile_harness_backend_available(count_compilations) -> None:
    """Canary: the compile-count harness observes a real backend compilation."""

    @jax.jit
    def _add_one(x: jax.Array) -> jax.Array:
        return x + 1.0

    array = jnp.ones(11)
    jax.block_until_ready(array)
    with count_compilations() as tally:
        jax.block_until_ready(_add_one(array))
    assert tally.count >= 1
