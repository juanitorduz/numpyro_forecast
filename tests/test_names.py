"""Tests of ``numpyro_forecast.names``: the contract of the constants and the no-literal guard."""

import ast
from pathlib import Path

from numpyro_forecast import names
from numpyro_forecast.names import FUTURE_SUFFIX, Dim, Group, Plate, Site, Var, future_site

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "numpyro_forecast"
ALLOWED_LITERALS: dict[str, frozenset[str]] = {
    "numpyro_forecast/validation.py": frozenset({"covariates"}),
    "numpyro_forecast/models.py": frozenset({"covariates"}),
    "numpyro_forecast/evaluate.py": frozenset({"covariates"}),
    "numpyro_forecast/predictive.py": frozenset({"covariates"}),
}
"""Literals that equal a names value but are not that name, per file relative to the root.

An entry exempts every occurrence of the value in that file, so it only suits
a file where the value never means the library name. ``"covariates"`` names
the ``covariates`` parameter of the model signature in ``require_time_axis``
and its callers, not the DataTree variable ``Var.COVARIATES``, which only
``convert.py`` writes and reads. Prefer a rewrite that leaves no bare constant
(a driver label is ``fn.__name__``, a prose placeholder sits inside a longer
string); extend this docstring with the reason for every entry.
"""


def test_future_site_appends_the_suffix() -> None:
    assert future_site("drift") == "drift_future"
    assert future_site(Site.OBS) == "obs_future"
    assert FUTURE_SUFFIX == "_future"


def test_shared_values_are_separate_constants() -> None:
    """The tree names its variable after the site and its time dim after the plate."""
    assert Site.OBS == Var.OBS
    assert Plate.TIME == Dim.TIME
    assert type(Site.OBS) is str
    assert type(Plate.TIME) is str


def _library_names() -> set[str]:
    values: set[str] = {FUTURE_SUFFIX, future_site(Site.OBS)}
    for cls in (Site, Plate, Dim, Var, Group):
        values.update(v for k, v in vars(cls).items() if not k.startswith("_"))
    return values


def _is_all(target: ast.expr) -> bool:
    return isinstance(target, ast.Name) and target.id == "__all__"


def _is_literal_hint(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "Literal"
    return isinstance(node, ast.Name) and node.id == "Literal"


def _exempt_constants(tree: ast.Module) -> set[int]:
    """Return the ids of the docstrings, ``__all__`` entries and ``Literal[...]`` option values."""
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    exempt.add(id(stmt.value))
        all_value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(map(_is_all, node.targets)):
            all_value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and _is_all(node.target):
            all_value = node.value
        if all_value is not None:
            exempt.update(id(n) for n in ast.walk(all_value) if isinstance(n, ast.Constant))
        if isinstance(node, ast.Subscript) and _is_literal_hint(node.value):
            exempt.update(id(n) for n in ast.walk(node.slice) if isinstance(n, ast.Constant))
    return exempt


def _spells(value: str, values: set[str]) -> bool:
    """Whether ``value`` is a library name or a ``/``-joined DataTree path of them."""
    return value in values or ("/" in value and all(part in values for part in value.split("/")))


def _name_literals(path: Path, values: set[str]) -> list[str]:
    """Return ``path:line: value`` for each bare string constant in ``path`` that is a name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exempt = _exempt_constants(tree)
    allowed = ALLOWED_LITERALS.get(path.relative_to(ROOT).as_posix(), frozenset())
    return [
        f"{path.relative_to(ROOT)}:{node.lineno}: {node.value!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _spells(node.value, values)
        and node.value not in allowed
        and id(node) not in exempt
    ]


def test_package_code_spells_no_library_name_literal() -> None:
    """Package and script code reads library-owned names from ``names``, never as literals.

    The walk covers f-string parts and ``"group/variable"`` tree paths too
    (``f"{name}_future"`` is caught) and skips docstrings, attribute docstrings,
    ``__all__`` and the option values of a ``Literal[...]`` hint. Tests, notebooks and
    the README keep the literals on purpose: they pin the values (see
    "Conventions" in ``AGENTS.md``).
    """
    values = _library_names()
    module = Path(names.__file__).resolve()
    paths = sorted(
        p
        for p in [*PACKAGE.rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]
        if p.resolve() != module
    )
    assert paths
    found = [hit for path in paths for hit in _name_literals(path, values)]
    assert not found, "library names spelled as literals (import them from names):\n" + "\n".join(
        found
    )
