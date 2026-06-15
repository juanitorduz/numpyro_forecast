"""numpyro_forecast: a JAX/NumPyro port of Pyro's forecasting module."""

from jaxtyping import install_import_hook

with install_import_hook("numpyro_forecast", "beartype.beartype"):
    from numpyro_forecast import (  # noqa: F401
        datasets,
        evaluate,
        forecaster,
        metrics,
        util,
    )

__version__ = "0.1.0"
