"""Smoke tests for package import and metadata."""

import numpyro_forecast


def test_version() -> None:
    assert isinstance(numpyro_forecast.__version__, str)
    assert numpyro_forecast.__version__
