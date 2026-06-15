"""Tests for the BART dataset loaders."""

import pytest

from numpyro_forecast.datasets import bart_available, load_bart_hierarchical


@pytest.mark.skipif(not bart_available(), reason="BART dataset unavailable")
def test_load_bart_hierarchical_rejects_oversized_window() -> None:
    """An over-large training window must fail fast, not silently wrap."""
    with pytest.raises(ValueError, match="exceeds available history"):
        load_bart_hierarchical(train_days=10**6)
