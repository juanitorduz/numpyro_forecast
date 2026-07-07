"""Tests for the optional-dependency gating helpers."""

import pytest

from numpyro_forecast import optional


def test_require_missing_module_message() -> None:
    """require() on a missing module names the extra and the pip install command."""
    with pytest.raises(ImportError) as excinfo:
        optional.require("definitely_not_a_module_xyz", extra="blackjax")
    msg = str(excinfo.value)
    assert "blackjax" in msg
    assert "pip install numpyro_forecast[blackjax]" in msg


def test_api_canary_missing_attr_message() -> None:
    """_api_canary on a real module with a fabricated attr raises a drift error."""
    with pytest.raises(AttributeError) as excinfo:
        optional._api_canary("math", ["definitely_not_an_attr"])
    msg = str(excinfo.value)
    assert "math" in msg
    assert "definitely_not_an_attr" in msg
    assert "drifted" in msg
