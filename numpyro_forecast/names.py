"""Names the package creates in one module and reads back in another.

Each name is a ``Final`` string attribute of a plain class, never a
``StrEnum`` member: NumPyro keeps the member itself as the plate or site name
in the trace (``type(name) is str`` is ``False``, it pickles by class path and
``yaml.safe_dump`` rejects it), while a ``Final`` attribute is a plain ``str``
at runtime and ``ty`` still reports a typo such as ``Site.FORECASTT`` as
``unresolved-attribute``. The values are public API: users spell them in their
model functions and stored notebooks carry them, so they never change.

Package code imports these constants instead of spelling the literals (also
inside ``==``, ``in``, ``.get``, ``endswith`` and f-strings; enforced by
``tests/test_names.py``). Tests, notebooks, docstrings and the README keep the
literals on purpose: they pin the values. This module imports nothing from the
package, so every module can import it without a cycle.

What stays a literal: a name the user chooses (the ``name`` of a building
block, ``dims``, ``obs_dims``, ``covariate_dims``, ``coords``, the ``group`` of
`~~numpyro_forecast.convert.predictions_to_datatree()`, a scope prefix), a name
another library owns (the dynestyx site suffixes, NumPyro message keys), a
``Literal`` option value (``"haar"`` / ``"dct"`` and their auxiliary suffixes,
``"rolling"`` / ``"expanding"``, the device strings), the metric keys of
``DEFAULT_METRICS``, and the ``"t"`` and ``"series"`` names of
`~~numpyro_forecast.convert.predictions_to_datatree()`.
"""

from typing import Final


class Site:
    """NumPyro site names the building blocks register and the drivers read.

    Attributes
    ----------
    OBS
        ``"obs"``, the observation site. `~~numpyro_forecast.models.predict()`
        and `~~numpyro_forecast.contrib.dynestyx.state_space()` register it;
        `~~numpyro_forecast.predictive.predict_in_sample()` and
        `~~numpyro_forecast.convert.to_datatree()` read it and drop a posterior
        key with this name (issue #260).
    FORECAST
        ``"forecast"``, the forecast deterministic site over the horizon.
        `predict()` and `state_space()` register it;
        `~~numpyro_forecast.predictive.forecast()` and
        `~~numpyro_forecast.evaluate.backtest()` read it, and a posterior key
        with this name and values is rejected (issue #161).
    """

    OBS: Final = "obs"
    FORECAST: Final = "forecast"


class Plate:
    """Names of the plates the building blocks open.

    Attributes
    ----------
    TIME
        ``"time"``, the in-sample plate of
        `~~numpyro_forecast.models.innovations()`;
        `~~numpyro_forecast.reparam.time_reparam()` targets the sites under it by
        this name.
    TIME_FUTURE
        ``"time_future"``, the forecast-horizon plate of `innovations()` and
        `~~numpyro_forecast.models.ssoe()`.
    """

    TIME: Final = "time"
    TIME_FUTURE: Final = "time_future"


class Dim:
    """xarray dimension names of the trees `~~numpyro_forecast.convert.to_datatree()` writes.

    ``TIME`` has the value of ``Plate.TIME`` by convention; it is the tree's
    dimension, not NumPyro's plate, and only this one is compared against the
    stored dimension names.

    Attributes
    ----------
    TIME
        ``"time"``, the time axis (axis ``-2`` of every stored variable).
    OBS_DIM
        ``"obs_dim"``, the default name of the observation axis of ``obs``.
    COVARIATE_DIM
        ``"covariate_dim"``, the default name of the covariate axis of
        ``covariates``.
    CHAIN
        ``"chain"``, the first ArviZ sample dimension.
    DRAW
        ``"draw"``, the second ArviZ sample dimension.
    """

    TIME: Final = "time"
    OBS_DIM: Final = "obs_dim"
    COVARIATE_DIM: Final = "covariate_dim"
    CHAIN: Final = "chain"
    DRAW: Final = "draw"


class Var:
    """Variables of the trees `~~numpyro_forecast.convert.to_datatree()` writes.

    `~~numpyro_forecast.convert.add_forecast_groups()` reads both back to inherit
    the stored dimension names and labels.

    Attributes
    ----------
    OBS
        ``"obs"``, the observed, predicted and forecast values; it has the value
        of ``Site.OBS`` because the tree names the variable after the site.
    COVARIATES
        ``"covariates"``, the in-sample and forecast covariates of the
        constant-data groups.
    """

    OBS: Final = "obs"
    COVARIATES: Final = "covariates"


class Group:
    """DataTree groups that `~~numpyro_forecast.convert.to_datatree()` writes.

    `~~numpyro_forecast.convert.add_forecast_groups()` adds the two forecast
    groups to an existing tree.

    Attributes
    ----------
    POSTERIOR
        ``"posterior"``, the posterior draws.
    POSTERIOR_PREDICTIVE
        ``"posterior_predictive"``, the in-sample predictive draws.
    OBSERVED_DATA
        ``"observed_data"``, the training data.
    CONSTANT_DATA
        ``"constant_data"``, the in-sample covariates.
    PREDICTIONS
        ``"predictions"``, the forecast draws (when the covariates extend past
        the data).
    PREDICTIONS_CONSTANT_DATA
        ``"predictions_constant_data"``, the forecast covariates.
    """

    POSTERIOR: Final = "posterior"
    POSTERIOR_PREDICTIVE: Final = "posterior_predictive"
    OBSERVED_DATA: Final = "observed_data"
    CONSTANT_DATA: Final = "constant_data"
    PREDICTIONS: Final = "predictions"
    PREDICTIONS_CONSTANT_DATA: Final = "predictions_constant_data"


SAMPLE_DIMS: Final[tuple[str, str]] = (Dim.CHAIN, Dim.DRAW)
"""The ArviZ sample dimensions shared by the posterior groups (stored in ``tree.attrs``)."""

DEFAULT_OBS_DIMS: Final[tuple[str, str]] = (Dim.TIME, Dim.OBS_DIM)
"""Default names of the trailing ``(time, obs)`` axes of every stored ``obs`` variable."""

DEFAULT_COVARIATE_DIMS: Final[tuple[str, str]] = (Dim.TIME, Dim.COVARIATE_DIM)
"""Default dimension names for the stored covariates (the 2-D layout)."""

RESERVED_PREDICTION_GROUPS: Final[frozenset[str]] = frozenset(
    {Group.OBSERVED_DATA, Group.CONSTANT_DATA}
)
"""The groups `predictions_to_datatree()` always writes, so its ``group`` cannot name them."""

FUTURE_SUFFIX: Final = "_future"
"""Suffix of a forecast-horizon site, see `future_site()`."""


def future_site(name: str) -> str:
    """Return the name of the forecast-horizon site of the in-sample site ``name``.

    `~~numpyro_forecast.models.innovations()`,
    `~~numpyro_forecast.models.markov_series()` and
    `~~numpyro_forecast.models.ssoe()` sample the horizon steps of ``name`` at
    this site, and `~~numpyro_forecast.models.predict()` samples the forecast
    suffix at ``future_site(Site.OBS)``. The drivers reject a posterior key
    with values that ends in ``FUTURE_SUFFIX`` (issue #161).

    Parameters
    ----------
    name
        Name of the in-sample site.

    Returns
    -------
    str
        ``name`` followed by ``FUTURE_SUFFIX``.

    Examples
    --------
    >>> from numpyro_forecast.names import future_site
    >>> future_site("drift")
    'drift_future'
    """
    return f"{name}{FUTURE_SUFFIX}"
