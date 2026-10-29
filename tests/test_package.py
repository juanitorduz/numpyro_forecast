"""Smoke tests for package import and metadata."""

import importlib
import inspect
import pkgutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

import numpyro_forecast
from numpyro_forecast import (
    DEFAULT_METRICS,
    Advance,
    BacktestResult,
    Horizon,
    SSOEMean,
    SSOEResult,
    SSOEUpdate,
    Transition,
    backtest,
    draw_posterior,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
    forecast,
    innovations,
    markov_series,
    predict,
    predict_in_sample,
    ssoe,
)
from numpyro_forecast.contrib.blackjax import BlackjaxNUTSKernel, PathfinderFit


def test_version() -> None:
    assert isinstance(numpyro_forecast.__version__, str)
    assert numpyro_forecast.__version__


def test_public_api_is_importable() -> None:
    # The curated top-level surface re-exported in ``__init__``.
    assert DEFAULT_METRICS  # the metrics mapping, a dict (unhashable, kept out of the set below)
    exported = {
        Advance,
        BacktestResult,
        Horizon,
        SSOEMean,
        SSOEResult,
        SSOEUpdate,
        Transition,
        backtest,
        draw_posterior,
        eval_coverage,
        eval_crps,
        eval_mae,
        eval_rmse,
        evaluate_forecast,
        forecast,
        innovations,
        markov_series,
        predict,
        predict_in_sample,
        ssoe,
    }
    assert all(obj is not None for obj in exported)


def test_all_matches_exported_names() -> None:
    names = set(numpyro_forecast.__all__)
    assert names == {
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
        "markov_series",
        "predict",
        "predict_in_sample",
        "predictions_to_datatree",
        "register_elementwise",
        "results_to_dataframe",
        "innovations",
        "ssoe",
        "time_reparam",
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


def test_every_package_function_is_runtime_type_checked() -> None:
    """Every function of every package module is wrapped by the jaxtyping import hook (#213).

    The hook in ``numpyro_forecast/__init__.py`` only wraps modules first imported
    inside its ``with`` block; a module imported after the block closes (as
    ``contrib.blackjax`` once was) silently loses its runtime type checks.
    """
    unchecked = []
    for info in pkgutil.walk_packages(numpyro_forecast.__path__, "numpyro_forecast."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if not inspect.isfunction(obj) or obj.__module__ != info.name:
                continue
            if not hasattr(obj, "__wrapped__"):
                unchecked.append(f"{info.name}.{name}")
    assert not unchecked, f"not wrapped by the jaxtyping import hook: {unchecked}"


def test_contrib_blackjax_rejects_wrong_argument_types() -> None:
    """Wrong argument types to ``contrib.blackjax`` fail at the call, like the core (#213)."""
    with pytest.raises(TypeCheckError):
        BlackjaxNUTSKernel(model="not a model")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeCheckError):
        PathfinderFit(
            state=None,
            model=None,  # ty: ignore[invalid-argument-type]
            covariates="x",  # ty: ignore[invalid-argument-type]
            data=3,  # ty: ignore[invalid-argument-type]
            elbo="y",  # ty: ignore[invalid-argument-type]
        )


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
