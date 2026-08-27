"""Tests for the ``ForecastModel`` runtime-checkable Protocol."""

from numpyro_forecast.typing import Array, ForecastModel


def test_plain_function_satisfies_forecast_model() -> None:
    def model(cov: Array, data: Array | None = None) -> None:
        return None

    assert isinstance(model, ForecastModel)


def test_lambda_satisfies_forecast_model() -> None:
    assert isinstance(lambda cov, data=None: None, ForecastModel)


def test_non_callable_does_not_satisfy_forecast_model() -> None:
    assert not isinstance(3, ForecastModel)


def test_wrong_signature_callable_still_satisfies_forecast_model() -> None:
    """Runtime protocols only check member existence, never signatures.

    A ``runtime_checkable`` Protocol's ``isinstance`` check reduces to
    ``callable(obj)`` here: Python never inspects parameter names, counts, or
    defaults at runtime, so a callable with an unrelated signature still
    passes. This is intentional and documented on `ForecastModel`; the
    real signature check is ``ty``'s static structural check at call sites,
    and a genuinely incompatible model only fails at the first driver call
    that invokes it with an unsupported argument.
    """

    def wrong_signature(x: int, y: int, z: int) -> int:
        return x + y + z

    assert isinstance(wrong_signature, ForecastModel)
