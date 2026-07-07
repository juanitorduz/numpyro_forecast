"""Optional-dependency gating behind ``pyproject`` extras.

Optional features (dataframes, blackjax, optax) are never imported at package
import time: :func:`require` imports them lazily at first use with an
actionable install hint, and :func:`_api_canary` asserts a pinned upstream API
surface so drift fails loudly.
"""

import importlib
from collections.abc import Sequence
from types import ModuleType


def require(module: str, *, extra: str) -> ModuleType:
    """Import an optional dependency, or raise a targeted ``ImportError``.

    Optional features (dataframes, blackjax, optax) live behind ``pyproject``
    extras and are never imported at package import time. This helper imports
    the backing module lazily at first use and, when it is missing, raises an
    ``ImportError`` naming the exact ``pip install`` invocation that provides it.

    Parameters
    ----------
    module
        The importable module name (e.g. ``"pandas"``).
    extra
        The ``numpyro_forecast`` extra that installs it (e.g. ``"dataframes"``).

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If ``module`` is not importable, with an actionable install hint.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        msg = (
            f"{module!r} is required for this feature; install it with "
            f"'pip install numpyro_forecast[{extra}]'."
        )
        raise ImportError(msg) from exc


def _api_canary(module: str, attrs: Sequence[str]) -> None:
    """Assert that ``module`` exposes every attribute in ``attrs``.

    A tripwire for optional-dependency API drift: extension modules call this at
    import (or in a dedicated canary test) so a renamed or removed upstream
    symbol fails with a precise message instead of a cryptic ``AttributeError``
    deep inside a call.

    Parameters
    ----------
    module
        The importable module name to probe.
    attrs
        Dotted attribute paths expected to resolve on the module (e.g.
        ``"vi.pathfinder.approximate"``).

    Raises
    ------
    AttributeError
        If any attribute path does not resolve; the message names the module,
        the missing path, and the installed version when available.
    """
    mod = importlib.import_module(module)
    missing: list[str] = []
    for attr in attrs:
        obj: object = mod
        for part in attr.split("."):
            if not hasattr(obj, part):
                missing.append(attr)
                break
            obj = getattr(obj, part)
    if missing:
        version = getattr(mod, "__version__", "unknown")
        msg = (
            f"{module} (version {version}) is missing expected attributes "
            f"{missing}; the pinned API surface has drifted."
        )
        raise AttributeError(msg)
