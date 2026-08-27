"""Tests for the package exception hierarchy."""

import pytest

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


@pytest.mark.parametrize(
    ("exc_cls", "builtin"),
    [
        (BacktestWindowError, ValueError),
        (VectorizedMetricError, TypeError),
        (KernelConfigError, ValueError),
        (CovariateDimsError, ValueError),
        (MVNLayoutError, NotImplementedError),
        (DeviceMemoryError, RuntimeError),
        (HostMemoryKindError, RuntimeError),
        (DevicePlatformError, ValueError),
    ],
)
def test_exception_subclasses_base_and_builtin(
    exc_cls: type[NumpyroForecastError], builtin: type[Exception]
) -> None:
    """Each exception is catchable as the package base and as its historical builtin."""
    assert issubclass(exc_cls, NumpyroForecastError)
    assert issubclass(exc_cls, builtin)


@pytest.mark.parametrize(
    "exc_cls",
    [
        VectorizedMetricError,
        MVNLayoutError,
    ],
)
def test_default_message_classes_instantiate_bare(
    exc_cls: type[NumpyroForecastError],
) -> None:
    """Classes with a default message carry it when constructed without arguments."""
    assert exc_cls.default_message
    assert str(exc_cls()) == exc_cls.default_message
    assert str(exc_cls("override")) == "override"
