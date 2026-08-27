"""Keep the great-docs API reference in sync with the public package surface.

The ``reference:`` section of ``great-docs.yml`` is curated by hand (see
``AGENTS.md``). These tests make forgetting to update it a red test rather than a
silent omission: every public function/class defined in a package submodule must
be listed, and every listed name must still resolve.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import numpyro_forecast

yaml = pytest.importorskip("yaml")

GREAT_DOCS_YML = Path(__file__).resolve().parent.parent / "great-docs.yml"

# Internal submodules whose public symbols are intentionally undocumented.
IGNORED_MODULES: set[str] = set()

# Optional third-party dependencies: a module whose import fails on one of these
# is covered by the extras CI leg instead of this walk. Any other ImportError
# (e.g. a package-internal name) is a real bug and must fail the test loudly.
_OPTIONAL_DEPS: frozenset[str] = frozenset({"blackjax", "optax", "pandas", "polars"})


def _documented_names() -> set[str]:
    config = yaml.safe_load(GREAT_DOCS_YML.read_text(encoding="utf-8"))
    names: set[str] = set()
    for section in config.get("reference", []):
        for item in section.get("contents", []):
            names.add(item if isinstance(item, str) else item["name"])
    return names


def _public_api() -> set[str]:
    """Return ``module.name`` for every public function/class defined in the package.

    Uses `pkgutil.walk_packages()` (recursive) so subpackages such as
    ``contrib`` are scanned too, not just top-level modules. This relies on
    ``contrib`` submodules being importable without their optional dependency
    (they pull the extra in lazily via ``require``), so walking them does not
    import ``blackjax``/``optax`` at collection time and invariant I8 (no optional
    imports on the base leg) is preserved. An ``ImportError`` whose failing name
    is in `_OPTIONAL_DEPS` skips just that module (the extras leg covers
    it); any other import failure propagates so a broken module cannot silently
    shrink the scanned surface.
    """
    prefix = f"{numpyro_forecast.__name__}."
    api: set[str] = set()
    for info in pkgutil.walk_packages(
        numpyro_forecast.__path__, prefix=prefix, onerror=lambda _: None
    ):
        relative = info.name.removeprefix(prefix)
        if (
            any(part.startswith("_") for part in relative.split("."))
            or relative in IGNORED_MODULES
        ):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError as err:
            if err.name in _OPTIONAL_DEPS:
                continue  # extras leg covers this module
            raise  # a real import bug must fail the test loudly
        for name, obj in inspect.getmembers(
            module, lambda o: inspect.isfunction(o) or inspect.isclass(o)
        ):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", "") == info.name:
                api.add(f"{relative}.{name}")
    return api


def test_public_api_is_documented():
    missing = _public_api() - _documented_names()
    assert not missing, (
        f"Public API missing from great-docs.yml reference: {sorted(missing)}. "
        "Add each to the appropriate `reference:` section."
    )


def test_public_api_walk_raises_on_internal_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ImportError not caused by an optional dep propagates out of the walk.

    Guards against the blanket-skip regression where a genuinely broken module
    (syntax error, renamed internal import) silently shrank the scanned API.
    """
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "numpyro_forecast.contrib.blackjax":
            raise ImportError("boom", name="numpyro_forecast.contrib.blackjax")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match="boom"):
        _public_api()


def test_documented_names_resolve():
    unresolved: list[str] = []
    for qualified in sorted(n for n in _documented_names() if "." in n):
        modname, _, attr = qualified.rpartition(".")
        try:
            module = importlib.import_module(f"numpyro_forecast.{modname}")
        except ModuleNotFoundError:
            unresolved.append(qualified)
            continue
        if not hasattr(module, attr):
            unresolved.append(qualified)
    assert not unresolved, f"great-docs.yml references names that no longer exist: {unresolved}."
