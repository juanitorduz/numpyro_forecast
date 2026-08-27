"""Forbid Sphinx roles and RST directives in docstrings and notebook prose.

The docs site renders docstrings as Quarto Markdown with great-docs, which does
**not** run Sphinx/RST conversion for numpy-style docstrings (see the "Docstring
markup" section of ``AGENTS.md``): Sphinx cross-reference roles, math directives,
literal-block markers and old-style RST hyperlinks all render literally or as dead
code spans instead of the intended cross reference, math or code block. This
module scans every module/class/function docstring under ``numpyro_forecast/``,
``tests/`` and ``scripts/``, plus every markdown cell of the example notebooks,
for the RST markup great-docs cannot render.
"""

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PY_FILES = (
    sorted(REPO.glob("numpyro_forecast/**/*.py"))
    + sorted(REPO.glob("tests/**/*.py"))
    + sorted(REPO.glob("scripts/*.py"))
)
NOTEBOOKS = sorted(REPO.glob("docs/examples/*.ipynb"))

BANNED: dict[str, re.Pattern[str]] = {
    "sphinx-role": re.compile(
        r":(?:py:)?(?:func|class|meth|attr|mod|data|exc|obj|const|type|math|ref|any|term|doc):`"
    ),
    "rst-directive": re.compile(r"^\s*\.\.\s+\w+::", re.MULTILINE),
    "rst-literal-block": re.compile(r"::\s*$", re.MULTILINE),
    "rst-hyperlink": re.compile(r"`[^`\n]+ <https?://[^>\n]+>`_"),
    "single-tilde": re.compile(r"`~(?!~)[A-Za-z_.]"),
}

_SEE_ALSO_HELP = 'see the "Docstring markup" section of AGENTS.md'


def _docstrings(path: Path) -> list[tuple[int, str, bool]]:
    """Collect every module/class/function docstring in a Python source file.

    Parameters
    ----------
    path
        Path to a ``.py`` file.

    Returns
    -------
    list[tuple[int, str, bool]]
        One ``(lineno, text, is_raw)`` tuple per docstring, where ``lineno`` is the
        line the docstring's string literal starts on and ``is_raw`` is whether that
        literal opens with a raw-string prefix (``r`` or ``R``, single or double
        quoted).
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    nodes: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = [tree]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            nodes.append(node)

    results: list[tuple[int, str, bool]] = []
    for node in nodes:
        text = ast.get_docstring(node, clean=False)
        if text is None:
            continue
        doc_node = tree.body[0] if node is tree else node.body[0]
        segment = ast.get_source_segment(source, doc_node)
        is_raw = segment is not None and segment.lstrip()[:2] in {'r"', 'R"', "r'", "R'"}
        results.append((doc_node.lineno, text, is_raw))
    return results


def _scan(text: str) -> list[tuple[int, str]]:
    """Find every banned-pattern hit in ``text``, as ``(line_offset, rule_name)``.

    Parameters
    ----------
    text
        The docstring or notebook cell source to scan.

    Returns
    -------
    list[tuple[int, str]]
        One entry per hit: the zero-based line offset within ``text`` and the
        matching rule name from `BANNED`.
    """
    hits: list[tuple[int, str]] = []
    for rule, pattern in BANNED.items():
        for match in pattern.finditer(text):
            offset_line = text.count("\n", 0, match.start())
            hits.append((offset_line, rule))
    return hits


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_python_docstrings_use_great_docs_markup(path: Path) -> None:
    """Every docstring in ``path`` must avoid Sphinx roles and RST directives."""
    rel = path.relative_to(REPO)
    messages: list[str] = []
    for lineno, text, _is_raw in _docstrings(path):
        for offset_line, rule in _scan(text):
            messages.append(f"{rel}:{lineno + offset_line}: {rule}")
    assert not messages, (
        f"found Sphinx/RST markup great-docs cannot render, {_SEE_ALSO_HELP}:\n"
        + "\n".join(messages)
    )


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_math_docstrings_are_raw(path: Path) -> None:
    """Every docstring containing both ``$`` and a backslash must be a raw string."""
    rel = path.relative_to(REPO)
    messages: list[str] = []
    for lineno, text, is_raw in _docstrings(path):
        if "$" in text and "\\" in text and not is_raw:
            messages.append(f"{rel}:{lineno}: math docstring is not a raw string")
    assert not messages, (
        f'docstrings with LaTeX and a backslash must be r""", {_SEE_ALSO_HELP}:\n'
        + "\n".join(messages)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO)))
def test_notebook_markdown_uses_great_docs_markup(path: Path) -> None:
    """Every markdown cell in ``path`` must avoid Sphinx roles and RST directives."""
    rel = path.relative_to(REPO)
    notebook = json.loads(path.read_text())
    messages: list[str] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else source
        for _offset_line, rule in _scan(text):
            messages.append(f"{rel}: cell {cell_index}: {rule}")
    assert not messages, (
        f"found Sphinx/RST markup great-docs cannot render, {_SEE_ALSO_HELP}:\n"
        + "\n".join(messages)
    )
