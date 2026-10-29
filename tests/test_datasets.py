"""Tests for the BART dataset loaders."""

import jax.numpy as jnp
import numpy as np
import pytest

from numpyro_forecast import datasets
from numpyro_forecast.datasets import (
    bart_available,
    load_bart_hierarchical,
    load_victoria_electricity,
)
from numpyro_forecast.typing import Array


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


_FAKE_HOURS = 3 * datasets.HOURS_PER_WEEK + 5  # 509 hours of fake counts


@pytest.fixture
def fake_bart_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the BART download with small fake ``(time, origin, destin)`` counts."""
    counts = jnp.asarray(np.arange(_FAKE_HOURS * 4, dtype=np.float32).reshape(_FAKE_HOURS, 2, 2))

    def fake_load_counts() -> tuple[Array, list[str]]:
        return counts, ["A", "B"]

    monkeypatch.setattr(datasets, "_load_counts", fake_load_counts)


@pytest.mark.usefixtures("fake_bart_counts")
def test_load_bart_hierarchical_valid_window() -> None:
    """A valid window keeps ``train_days * 24`` training hours and ``test_weeks`` test weeks."""
    y, split, stations = load_bart_hierarchical(train_days=7, test_weeks=1)
    assert y.shape == (2, 7 * 24 + datasets.HOURS_PER_WEEK, 2)
    assert split == 7 * 24
    assert stations == ["A", "B"]


@pytest.mark.usefixtures("fake_bart_counts")
def test_load_bart_hierarchical_zero_test_weeks() -> None:
    """``test_weeks=0`` returns the training window only (``split == y.shape[1]``)."""
    y, split, _ = load_bart_hierarchical(train_days=7, test_weeks=0)
    assert y.shape == (2, 7 * 24, 2)
    assert split == y.shape[1]


@pytest.fixture
def no_bart_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the loader reaches the BART download."""

    def fail() -> tuple[Array, list[str]]:
        raise AssertionError("argument validation must run before the download")

    monkeypatch.setattr(datasets, "_load_counts", fail)


@pytest.mark.usefixtures("no_bart_download")
@pytest.mark.parametrize("train_days", [0, -1])
def test_load_bart_hierarchical_rejects_non_positive_train_days(train_days: int) -> None:
    """A ``train_days`` below 1 names the argument and fails before the download."""
    with pytest.raises(ValueError, match=rf"train_days must be at least 1, got {train_days}"):
        load_bart_hierarchical(train_days=train_days, test_weeks=1)


@pytest.mark.usefixtures("no_bart_download")
def test_load_bart_hierarchical_rejects_negative_test_weeks() -> None:
    """A negative ``test_weeks`` fails before the download instead of returning an empty array."""
    with pytest.raises(ValueError, match=r"test_weeks must be non-negative, got -1"):
        load_bart_hierarchical(train_days=7, test_weeks=-1)


@pytest.mark.usefixtures("fake_bart_counts")
def test_load_bart_hierarchical_rejects_oversized_window_fake_counts() -> None:
    """A window longer than the history keeps the history message."""
    with pytest.raises(ValueError, match=f"exceeds available history of {_FAKE_HOURS} hours"):
        load_bart_hierarchical(train_days=30, test_weeks=1)


def test_load_victoria_electricity_shapes_and_values() -> None:
    """The bundled CSV loads into aligned demand/temperature arrays."""
    demand, temperature = load_victoria_electricity()
    assert demand.shape == (1_344, 1)  # eight weeks of hourly data
    assert temperature.shape == (1_344,)
    assert float(demand[0, 0]) == pytest.approx(3.794, abs=1e-3)
    assert float(temperature[0]) == pytest.approx(18.05, abs=1e-3)
