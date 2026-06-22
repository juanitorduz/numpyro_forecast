"""Smoke tests for package import and metadata."""

import numpyro_forecast
from numpyro_forecast import (
    BacktestResult,
    Forecaster,
    ForecastingModel,
    HMCForecaster,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    forecasting_model,
)


def test_version() -> None:
    assert isinstance(numpyro_forecast.__version__, str)
    assert numpyro_forecast.__version__


def test_public_api_is_importable() -> None:
    # The curated top-level surface re-exported in ``__init__``.
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
        forecasting_model,
    }
    assert all(obj is not None for obj in exported)


def test_all_matches_exported_names() -> None:
    names = set(numpyro_forecast.__all__)
    assert names == {
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
        "forecasting_model",
    }
    for name in names:
        assert hasattr(numpyro_forecast, name)
