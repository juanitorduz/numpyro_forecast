"""Validated example forecasting models."""

from numpyro_forecast.models.hierarchical import HierarchicalForecaster
from numpyro_forecast.models.univariate import UnivariateForecaster

__all__ = ["HierarchicalForecaster", "UnivariateForecaster"]
