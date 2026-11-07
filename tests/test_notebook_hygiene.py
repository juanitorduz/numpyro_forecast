"""Check the rules that hold for every example notebook: outputs, links and prose style.

The docs build renders the committed notebooks with their stored outputs (no re-execution),
so a warning from the author's environment, with its local ``site-packages`` path, would show
up on the published site. The site is also versioned (the stable root and ``/v/dev/``), so a
link to another example or to the API reference must be relative: an absolute link to the
site always lands on the stable root, whose reference is pruned to the last release. The
prose rules (percentages in LaTeX, no claim that ``SVI`` needs ``optax_to_numpyro``) were
broken in more than one notebook, so they are checked here for all of them.

The preliz checks guard the pattern that lets a prior elicited with ``maxent`` reach the model.
The default ``mass`` of ``maxent`` is ``rcParams["stats.ci_prob"]``, which went from 0.94
(preliz 0.22) to 0.89 (preliz 0.23), and two notebooks plotted the default-mass fit while
their models hard-coded the 0.94 fit (issue #198). So every call passes ``mass``, and the
fit is bound to a name that is read after the call (a name-binding heuristic: the checks do
not see whether the model uses the fit) instead of copied by hand.
"""

import ast
import re
from pathlib import Path
from typing import TypeGuard

import pytest
import yaml
from notebook_cells import NOTEBOOKS, Cell, read_cells, source, stderr

REPO = Path(__file__).resolve().parent.parent
SITE = "https://juanitorduz.github.io/numpyro_forecast/"
# Warnings that describe the author's environment, not the example.
ENVIRONMENT_WARNINGS = (
    "PyMC not installed",
    "When multiple credible intervals are plotted",
)
# A relative link from an example page to another example or to a reference page. Reference
# anchors contain dots (``#numpyro_forecast.models.ssoe``).
RELATIVE_LINK = re.compile(r"\]\((?!https?://)(?P<path>[\w./]+\.html)(?:#[^)\s]*)?\)")
NOTEBOOK_IDS = [path.stem for path in NOTEBOOKS]
# IPython magics (``%load_ext``, ``%%time``) and shell escapes (``!pip``) are not Python. The
# magic name follows the sigil directly, so a wrapped ``% 24`` operator line does not match.
MAGIC_LINE = re.compile(r"^(?P<indent>\s*)[%!]{1,2}[A-Za-z_]")


def _reference_entries() -> set[str]:
    """The page stems of the API reference: every entry under ``reference:``."""
    config = yaml.safe_load((REPO / "great-docs.yml").read_text())
    return {
        entry if isinstance(entry, str) else entry["name"]
        for section in config["reference"]
        for entry in section["contents"]
    }


def _code_trees(name: str, cells: list[Cell]) -> list[tuple[int, ast.Module]]:
    """Parse the code cells of notebook ``name`` as Python.

    A magic line becomes ``pass`` at the same indentation, so line numbers and block bodies
    survive. A cell that is still not Python (a ``%%bash`` body, ``x = !cmd``) is skipped,
    unless its source mentions ``maxent``: the maxent checks read the fit from that cell, so
    it fails the test by name (``pz.maxent?``).
    """
    trees = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        text = source(cell)
        lines = [
            f"{match['indent']}pass" if (match := MAGIC_LINE.match(line)) else line
            for line in text.splitlines()
        ]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError as exc:
            if "maxent" not in text:
                continue
            pytest.fail(
                f"{name} cell {index} is not Python once IPython magics are blanked: {exc}. "
                "The maxent checks (mass passed, fit read after the call) need every cell "
                "that mentions maxent to parse as Python."
            )
        trees.append((index, tree))
    return trees


def _is_maxent(node: ast.AST | None) -> TypeGuard[ast.Call]:
    """Whether ``node`` calls preliz's ``maxent``, as ``pz.maxent(...)`` or ``maxent(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == "maxent") or (
        isinstance(func, ast.Name) and func.id == "maxent"
    )


def _maxent_calls(name: str, cells: list[Cell]) -> list[tuple[int, ast.Call, set[str]]]:
    """Return ``(cell index, call, names that hold the fit)`` for every ``maxent`` call.

    ``maxent`` fits its distribution argument in place and returns it (as ``(distribution,
    axes)`` when it plots, the default), so the fit is in the argument when it is a name and
    in the first assignment target. A list, set or generator comprehension whose element is
    the call binds its fits to the assignment target as well.
    """
    calls = []
    for index, tree in _code_trees(name, cells):
        assigned: dict[int, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if isinstance(value, ast.ListComp | ast.GeneratorExp | ast.SetComp):
                value = value.elt
            if not _is_maxent(value):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            firsts = [t.elts[0] if isinstance(t, ast.Tuple | ast.List) else t for t in targets]
            assigned[id(value)] = {t.id for t in firsts if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            if not _is_maxent(node):
                continue
            argument = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "distribution":
                    argument = keyword.value
            names = assigned.get(id(node), set())
            if isinstance(argument, ast.Name):
                names = names | {argument.id}
            calls.append((index, node, names - {"_"}))
    return calls


def _is_none(node: ast.expr) -> bool:
    """Whether ``node`` is the literal ``None``."""
    return isinstance(node, ast.Constant) and node.value is None


def _passes_mass(call: ast.Call) -> bool:
    """Whether a ``maxent`` call passes ``mass``, other than an explicit ``None``.

    ``mass`` is the fourth positional parameter; ``**kwargs`` (``arg is None``) may hold it.
    A ``None`` mass, positional or keyword, still resolves to ``rcParams["stats.ci_prob"]``.
    """
    return (len(call.args) >= 4 and not _is_none(call.args[3])) or any(
        keyword.arg is None or (keyword.arg == "mass" and not _is_none(keyword.value))
        for keyword in call.keywords
    )


def _check_maxent_calls_pass_mass(name: str, cells: list[Cell]) -> None:
    """Fail when a ``maxent`` call of notebook ``name`` leaves ``mass`` to the preliz default."""
    found = [
        f"cell {index}, line {call.lineno}"
        for index, call, _ in _maxent_calls(name, cells)
        if not _passes_mass(call)
    ]
    assert not found, (
        f"{name} calls maxent without mass; its default follows "
        'rcParams["stats.ci_prob"], which differs between preliz releases:\n' + "\n".join(found)
    )


def _check_maxent_fits_are_used(name: str, cells: list[Cell]) -> None:
    """Fail when a ``maxent`` fit of notebook ``name`` is bound to no name that is read later.

    A fit is read when one of the names that hold it (the first assignment target or the
    distribution argument, ``_`` excluded) is loaded after the call ends: later on the same
    line, on a later line of the cell, or in a later cell.
    """
    loads = [
        (index, node.lineno, node.col_offset, node.id)
        for index, tree in _code_trees(name, cells)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    ]
    found = [
        f"cell {index}, line {call.lineno}"
        for index, call, names in _maxent_calls(name, cells)
        if not any(
            name in names
            and (cell, line, column)
            > (index, call.end_lineno or call.lineno, call.end_col_offset or call.col_offset)
            for cell, line, column, name in loads
        )
    ]
    assert not found, (
        f"{name} discards a maxent fit: every fit must be bound to a name that is read after "
        "the call (not bound to _), so the model's prior can be built from it (for example "
        "dist.InverseGamma(concentration=fit.alpha, rate=fit.beta)) instead of copying its "
        "parameters:\n" + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_maxent_calls_pass_mass(path: Path) -> None:
    """Every preliz ``maxent`` call passes ``mass``: its default changed between releases."""
    _check_maxent_calls_pass_mass(path.name, read_cells(path))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_maxent_fits_are_used(path: Path) -> None:
    """Every ``maxent`` fit is bound to a name that is read after the call."""
    _check_maxent_fits_are_used(path.name, read_cells(path))


def _cells(*sources: str) -> list[Cell]:
    """Build the code cells of an in-memory notebook, one per source string."""
    return [{"cell_type": "code", "source": text} for text in sources]


def test_maxent_check_rejects_a_discarded_fit() -> None:
    """A fit bound to ``_`` and never read fails, naming the rule and the cell."""
    cells = _cells("import preliz as pz", "_ = pz.maxent(pz.Gamma(), 0, 10, mass=0.9)")
    with pytest.raises(AssertionError, match=r"discards a maxent fit(.|\n)*cell 1, line 1"):
        _check_maxent_fits_are_used("synthetic.ipynb", cells)


def test_maxent_check_rejects_a_plot_only_call() -> None:
    """A call whose fit is neither bound nor read fails, even when it only draws a plot."""
    cells = _cells("pz.maxent(pz.Gamma(), 0, 10, mass=0.9)")
    with pytest.raises(AssertionError, match="discards a maxent fit"):
        _check_maxent_fits_are_used("synthetic.ipynb", cells)


def test_maxent_check_counts_a_read_on_the_same_line() -> None:
    """A read after the call on the same line counts as using the fit."""
    cells = _cells("fit, ax = pz.maxent(pz.Gamma(), 0, 10, mass=0.9); print(fit.alpha)")
    _check_maxent_fits_are_used("synthetic.ipynb", cells)


def test_maxent_check_counts_a_comprehension_target() -> None:
    """A comprehension of fits binds them to the assignment target, read in a later cell."""
    cells = _cells(
        "fits = [pz.maxent(d, 0, 10, mass=0.9, plot=False) for d in (pz.Gamma(), pz.Beta())]",
        "print(fits)",
    )
    _check_maxent_fits_are_used("synthetic.ipynb", cells)


def test_maxent_check_rejects_mass_none() -> None:
    """``mass=None`` still takes the preliz default, so it does not pass the check."""
    cells = _cells("fit, ax = pz.maxent(pz.Gamma(), 0, 10, mass=None)\nprint(fit)")
    with pytest.raises(AssertionError, match="calls maxent without mass"):
        _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
    _check_maxent_calls_pass_mass("synthetic.ipynb", _cells("fit = pz.maxent(d, mass=0.9)"))


def test_code_trees_skip_a_cell_that_is_not_python_without_maxent() -> None:
    """A ``%%bash`` cell does not parse once its first line is blanked, so it is skipped."""
    cells = _cells("x = 1", "%%bash\necho hello world", "y = x")
    assert [index for index, _ in _code_trees("synthetic.ipynb", cells)] == [0, 2]
    _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
    _check_maxent_fits_are_used("synthetic.ipynb", cells)


def test_code_trees_fail_on_a_maxent_cell_that_is_not_python() -> None:
    """A cell that mentions ``maxent`` and does not parse fails, naming the maxent rule."""
    with pytest.raises(
        pytest.fail.Exception,
        match=r"synthetic\.ipynb cell 1(.|\n)*invalid syntax(.|\n)*maxent",
    ):
        _code_trees("synthetic.ipynb", _cells("x = 1", "pz.maxent?"))


def test_maxent_check_rejects_a_positional_none_mass() -> None:
    """A positional ``None`` mass takes the preliz default like the keyword form."""
    cells = _cells("fit, ax = pz.maxent(pz.Gamma(), 0, 10, None)\nprint(fit)")
    with pytest.raises(AssertionError, match="calls maxent without mass"):
        _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
    _check_maxent_calls_pass_mass("synthetic.ipynb", _cells("pz.maxent(pz.Gamma(), 0, 10, 0.9)"))


def test_maxent_check_counts_a_fitted_argument_bound_to_underscore() -> None:
    """``maxent`` fits its argument in place, so ``_ = pz.maxent(d, ...)`` with ``d`` read passes."""
    cells = _cells("d = pz.Gamma()", "_ = pz.maxent(d, 0, 10, mass=0.9)", "print(d.alpha)")
    _check_maxent_fits_are_used("synthetic.ipynb", cells)


def test_code_trees_keep_the_body_of_an_indented_magic() -> None:
    """An indented ``%time`` becomes ``pass``, so the enclosing block still parses."""
    (tree,) = [
        tree for _, tree in _code_trees("s.ipynb", _cells("for _ in range(2):\n    %time x = 1"))
    ]
    (loop,) = tree.body
    assert isinstance(loop, ast.For) and isinstance(loop.body[0], ast.Pass)


def test_code_trees_keep_a_wrapped_modulo_operator() -> None:
    """A continuation line that starts with the ``%`` operator is not a magic."""
    (tree,) = [tree for _, tree in _code_trees("s.ipynb", _cells("x = (\n    hours\n    % 24\n)"))]
    (assign,) = tree.body
    assert isinstance(assign, ast.Assign) and isinstance(assign.value, ast.BinOp)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_no_stored_environment_warnings(path: Path) -> None:
    """No stored stderr output repeats an environment warning or a ``site-packages`` path."""
    found = [
        f"cell {index}: {text.splitlines()[0][:100]}"
        for index, text in stderr(read_cells(path))
        if "site-packages/" in text or any(warning in text for warning in ENVIRONMENT_WARNINGS)
    ]
    assert not found, (
        f"{path.name} stores environment warnings; filter them in the setup cell:\n"
        + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_links_to_the_docs_site_are_relative(path: Path) -> None:
    """Markdown cells link to other examples and to the reference with relative paths."""
    found = [
        f"cell {index}: {link}"
        for index, cell in enumerate(read_cells(path))
        if cell["cell_type"] == "markdown"
        for link in re.findall(re.escape(SITE) + r"\S*?(?=[)\s])", source(cell))
    ]
    assert not found, (
        f"{path.name} has absolute links to the docs site; use `<example>.html` or "
        "`../../reference/<entry>.html`:\n" + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_relative_links_point_at_existing_pages(path: Path) -> None:
    """Every relative link names an example notebook or an entry of the API reference."""
    examples = {notebook.stem for notebook in NOTEBOOKS}
    reference = _reference_entries()
    broken = []
    for index, cell in enumerate(read_cells(path)):
        if cell["cell_type"] != "markdown":
            continue
        for match in RELATIVE_LINK.finditer(source(cell)):
            link = match.group("path")
            if link.startswith("../../reference/"):
                ok = link.removeprefix("../../reference/").removesuffix(".html") in reference
            else:
                ok = "/" not in link and link.removesuffix(".html") in examples
            if not ok:
                broken.append(f"cell {index}: {link}")
    assert not broken, f"{path.name} links to pages that do not exist:\n" + "\n".join(broken)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_percentages_in_prose_use_latex(path: Path) -> None:
    r"""Markdown cells write percentages as LaTeX (``$30\%$``), not as code spans or plain text."""
    found = []
    for index, cell in enumerate(read_cells(path)):
        if cell["cell_type"] != "markdown":
            continue
        text = re.sub(r"\$\$.*?\$\$", "", source(cell), flags=re.DOTALL)
        text = re.sub(r"\$[^$\n]*\$", "", text)
        found += [f"cell {index}: {match}" for match in re.findall(r"\S*\d%\S*", text)]
    assert not found, f"{path.name} writes percentages outside LaTeX:\n" + "\n".join(found)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_no_claim_that_svi_needs_optax_to_numpyro(path: Path) -> None:
    """``numpyro.infer.SVI`` wraps an optax ``GradientTransformation`` itself."""
    found = [
        f"cell {index}"
        for index, cell in enumerate(read_cells(path))
        if cell["cell_type"] == "markdown" and "expects a NumPyro optimizer" in source(cell)
    ]
    assert not found, f"{path.name} says that SVI needs optax_to_numpyro: {found}"
