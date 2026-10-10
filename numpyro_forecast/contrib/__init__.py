"""Optional extension backends for `numpyro_forecast`.

Each submodule wires a soft dependency (installed via a ``pyproject`` extra)
into the core resolution layer. No extra is imported at package import time:
importing `numpyro_forecast` binds ``numpyro_forecast.contrib.blackjax`` (so the
docs can introspect it) but never pulls in ``blackjax`` itself, which
`numpyro_forecast.optional.require()` loads the first time a kernel initializes.
Import the concrete backend explicitly, e.g. ``from
numpyro_forecast.contrib.blackjax import BlackjaxNUTSKernel``.
"""
