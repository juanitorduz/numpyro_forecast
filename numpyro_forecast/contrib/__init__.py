"""Optional extension backends for `numpyro_forecast`.

Each submodule wires a soft dependency (installed via a ``pyproject`` extra)
into the core resolution layer. No extra is imported at package import time:
importing `numpyro_forecast` binds ``numpyro_forecast.contrib.blackjax`` and
``numpyro_forecast.contrib.dynestyx`` (so the docs can introspect them) but
never pulls in ``blackjax`` or ``dynestyx`` themselves, which
`numpyro_forecast.optional.require()` loads at first use. Import the concrete
backend explicitly, e.g. ``from numpyro_forecast.contrib.blackjax import
BlackjaxNUTSKernel`` or ``from numpyro_forecast.contrib.dynestyx import
state_space``.
"""
