"""Check the rules that hold for every example notebook: outputs, links and prose style.

The docs build renders the committed notebooks with their stored outputs (no re-execution),
so a warning from the author's environment, with its local ``site-packages`` path, would show
up on the published site. The site is also versioned (the stable root and ``/v/dev/``), so a
link to another example or to the API reference must be relative: an absolute link to the
site always lands on the stable root, whose reference is pruned to the last release. The
prose rules (percentages in LaTeX, no claim that ``SVI`` needs ``optax_to_numpyro``) were
broken in more than one notebook, so they are checked here for all of them.

The preliz check guards the elicited priors of issue #198. The default ``mass`` of ``maxent``
is ``rcParams["stats.ci_prob"]``, which went from 0.94 (preliz 0.22) to 0.89 (preliz 0.23),
and two notebooks plotted the default-mass fit while their models hard-coded the 0.94 fit. So
every ``maxent`` call passes ``mass``, other than ``None``.
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
ENVIRONMENT_WARNINGS = (
    "PyMC not installed",
    "When multiple credible intervals are plotted",
)
RELATIVE_LINK = re.compile(r"\]\((?!https?://)(?P<path>[\w./]+\.html)(?:#[^)\s]*)?\)")
NOTEBOOK_IDS = [path.stem for path in NOTEBOOKS]
MAGIC_LINE = re.compile(r"^(?P<indent>\s*)[%!]{1,2}[A-Za-z_]")
TIME_MAGIC_LINE = re.compile(r"^(?P<indent>\s*)%time(?:it)?\s+(?P<stmt>[^-\s].*)$")
MAXENT_CALL = re.compile(r"\bmaxent\s*\(")


def _reference_entries() -> set[str]:
    """The page stems of the API reference: every entry under ``reference:``."""
    config = yaml.safe_load((REPO / "great-docs.yml").read_text())
    return {
        entry if isinstance(entry, str) else entry["name"]
        for section in config["reference"]
        for entry in section["contents"]
    }


def _unmagic(line: str) -> str:
    """Return ``line`` as Python: a flag-less time magic keeps its statement, other magics pass.

    The kept statement is unmagicked in turn, so ``%time !ls`` becomes ``pass``.
    """
    if match := TIME_MAGIC_LINE.match(line):
        return _unmagic(f"{match['indent']}{match['stmt']}")
    if match := MAGIC_LINE.match(line):
        return f"{match['indent']}pass"
    return line


def _code_trees(name: str, cells: list[Cell]) -> list[tuple[int, ast.Module]]:
    """Parse the code cells of notebook ``name`` as Python.

    A flag-less ``%time``/``%timeit`` line magic keeps its statement at the same indentation;
    any other magic line becomes ``pass``, so line numbers and block bodies survive. A cell
    that is still not Python (a ``%%bash`` body, ``x = !cmd``) is skipped, unless its source
    calls ``maxent``: the mass check reads that call, so the cell fails the test by name.
    """
    trees = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        text = source(cell)
        lines = [_unmagic(line) for line in text.splitlines()]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError as exc:
            if not MAXENT_CALL.search(text):
                continue
            pytest.fail(
                f"{name} cell {index} is not Python once IPython magics are blanked: {exc}. "
                "The maxent check (every call passes mass) needs every cell that calls "
                "maxent to parse as Python."
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


def _maxent_calls(name: str, cells: list[Cell]) -> list[tuple[int, ast.Call]]:
    """Return ``(cell index, call)`` for every ``maxent`` call of notebook ``name``."""
    return [
        (index, node)
        for index, tree in _code_trees(name, cells)
        for node in ast.walk(tree)
        if _is_maxent(node)
    ]


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
        for index, call in _maxent_calls(name, cells)
        if not _passes_mass(call)
    ]
    assert not found, (
        f"{name} calls maxent without mass; its default follows "
        'rcParams["stats.ci_prob"], which differs between preliz releases:\n' + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_maxent_calls_pass_mass(path: Path) -> None:
    """Every preliz ``maxent`` call passes ``mass``: its default changed between releases."""
    _check_maxent_calls_pass_mass(path.name, read_cells(path))


def _cells(*sources: str) -> list[Cell]:
    """Build the code cells of an in-memory notebook, one per source string."""
    return [{"cell_type": "code", "source": text} for text in sources]


def test_maxent_check_rejects_mass_none() -> None:
    """``mass=None`` still takes the preliz default, so it does not pass the check."""
    cells = _cells("fit, ax = pz.maxent(pz.Gamma(), 0, 10, mass=None)\nprint(fit)")
    with pytest.raises(AssertionError, match="calls maxent without mass"):
        _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
    _check_maxent_calls_pass_mass("synthetic.ipynb", _cells("fit = pz.maxent(d, mass=0.9)"))


def test_code_trees_skip_a_cell_that_only_mentions_maxent() -> None:
    """A ``%%bash`` or ``%%html`` cell does not parse and calls no ``maxent``, so it is skipped."""
    cells = _cells("x = 1", "%%bash\necho maxent", "%%html\n<b>maxent</b>", "y = x")
    assert [index for index, _ in _code_trees("synthetic.ipynb", cells)] == [0, 3]
    _check_maxent_calls_pass_mass("synthetic.ipynb", cells)


def test_code_trees_fail_on_a_maxent_cell_that_is_not_python() -> None:
    """A cell that calls ``maxent`` and does not parse fails, naming the mass rule."""
    cells = _cells("x = 1", "fit = pz.maxent(pz.Gamma(), 0, 10, mass=0.9) ?")
    with pytest.raises(
        pytest.fail.Exception,
        match=r"synthetic\.ipynb cell 1(.|\n)*every call passes mass",
    ):
        _check_maxent_calls_pass_mass("synthetic.ipynb", cells)


def test_maxent_check_rejects_a_positional_none_mass() -> None:
    """A positional ``None`` mass takes the preliz default like the keyword form."""
    cells = _cells("fit, ax = pz.maxent(pz.Gamma(), 0, 10, None)\nprint(fit)")
    with pytest.raises(AssertionError, match="calls maxent without mass"):
        _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
    _check_maxent_calls_pass_mass("synthetic.ipynb", _cells("pz.maxent(pz.Gamma(), 0, 10, 0.9)"))


def test_code_trees_keep_the_body_of_an_indented_magic() -> None:
    """An indented ``%timeit -n 3`` becomes ``pass``, so the enclosing block still parses."""
    cells = _cells("for _ in range(2):\n    %timeit -n 3 x = 1")
    (tree,) = [tree for _, tree in _code_trees("s.ipynb", cells)]
    (loop,) = tree.body
    assert isinstance(loop, ast.For) and isinstance(loop.body[0], ast.Pass)


def test_maxent_check_reads_the_statement_of_a_time_magic() -> None:
    """A flag-less ``%time`` or ``%timeit`` keeps its statement, so its ``maxent`` is checked."""
    for magic in ("%time", "%timeit"):
        cells = _cells(f"x = 1\n{magic} f = pz.maxent(pz.Gamma(), 0, 10)")
        with pytest.raises(AssertionError, match="calls maxent without mass"):
            _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
        cells = _cells(f"x = 1\n{magic} f = pz.maxent(pz.Gamma(), 0, 10, mass=0.9)")
        _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
        (tree,) = [tree for _, tree in _code_trees("s.ipynb", cells)]
        assert isinstance(tree.body[1], ast.Assign) and tree.body[1].lineno == 2


def test_unmagic_blanks_a_magic_behind_a_time_magic() -> None:
    """The kept statement of a ``%time`` is unmagicked too, so no magic is left in the cell."""
    assert _unmagic("    %time !ls") == "    pass"
    assert _unmagic("%time %run x") == "pass"
    assert _unmagic("%time %timeit f = pz.maxent(d)") == "f = pz.maxent(d)"
    cells = _cells("x = 1\n%time !ls\n%timeit %run x")
    (tree,) = [tree for _, tree in _code_trees("s.ipynb", cells)]
    assert [type(node) for node in tree.body] == [ast.Assign, ast.Pass, ast.Pass]


def test_maxent_check_skips_a_timeit_magic_with_flags() -> None:
    """A ``%timeit`` with flags still becomes ``pass``, at the same line."""
    cells = _cells("x = 1\n%timeit -n 3 pz.maxent(pz.Gamma(), 0, 10)")
    _check_maxent_calls_pass_mass("synthetic.ipynb", cells)
    (tree,) = [tree for _, tree in _code_trees("s.ipynb", cells)]
    assert isinstance(tree.body[1], ast.Pass) and tree.body[1].lineno == 2


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
