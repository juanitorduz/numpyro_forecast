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


def _documented_names() -> set[str]:
    config = yaml.safe_load(GREAT_DOCS_YML.read_text(encoding="utf-8"))
    names: set[str] = set()
    for section in config.get("reference", []):
        for item in section.get("contents", []):
            names.add(item if isinstance(item, str) else item["name"])
    return names


def _public_api() -> set[str]:
    """Return ``module.name`` for every public function/class defined in the package."""
    api: set[str] = set()
    for info in pkgutil.iter_modules(numpyro_forecast.__path__):
        if info.name.startswith("_") or info.name in IGNORED_MODULES:
            continue
        module = importlib.import_module(f"numpyro_forecast.{info.name}")
        for name, obj in inspect.getmembers(
            module, lambda o: inspect.isfunction(o) or inspect.isclass(o)
        ):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", "") == f"numpyro_forecast.{info.name}":
                api.add(f"{info.name}.{name}")
    return api


def test_public_api_is_documented():
    missing = _public_api() - _documented_names()
    assert not missing, (
        f"Public API missing from great-docs.yml reference: {sorted(missing)}. "
        "Add each to the appropriate `reference:` section."
    )


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
