"""Tests for the BART dataset loaders."""

import pytest

from numpyro_forecast import datasets
from numpyro_forecast.datasets import (
    bart_available,
    load_bart_hierarchical,
    load_victoria_electricity,
)


def test_bart_available_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed download (any exception) makes ``bart_available`` return False."""

    def boom() -> None:
        raise RuntimeError("download failed")

    monkeypatch.setattr(datasets, "_load_counts", boom)
    assert bart_available() is False


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_load_bart_hierarchical_rejects_oversized_window() -> None:
    """An over-large training window must fail fast, not silently wrap."""
    with pytest.raises(ValueError, match="exceeds available history"):
        load_bart_hierarchical(train_days=10**6)


def test_load_victoria_electricity_shapes_and_values() -> None:
    """The bundled CSV loads into aligned demand/temperature arrays."""
    demand, temperature = load_victoria_electricity()
    assert demand.shape == (1_344, 1)  # eight weeks of hourly data
    assert temperature.shape == (1_344,)
    assert float(demand[0, 0]) == pytest.approx(3.794, abs=1e-3)
    assert float(temperature[0]) == pytest.approx(18.05, abs=1e-3)
