"""Tests for the BART dataset loaders."""

import pytest

from numpyro_forecast import datasets
from numpyro_forecast.datasets import bart_available, load_bart_hierarchical


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
