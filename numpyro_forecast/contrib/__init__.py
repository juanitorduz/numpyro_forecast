"""Optional extension backends for :mod:`numpyro_forecast`.

Each submodule wires a soft dependency (installed via a ``pyproject`` extra)
into the core resolution layer. Nothing here is imported at package import
time: importing :mod:`numpyro_forecast` never pulls in ``blackjax`` or any
other extra. Import the concrete backend explicitly, e.g. ``from
numpyro_forecast.contrib.blackjax import BlackjaxNUTSKernel``.
"""
