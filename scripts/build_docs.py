"""Build the documentation site, generating the example pages on the fly.

great-docs renders ``.qmd``/``.md`` files as documentation pages but does not
render Jupyter notebooks directly, and it only stages a section's ``.qmd`` pages
and asset *subdirectories* into its build tree (never a loose top-level
notebook). Each notebook in ``docs/examples/`` is therefore shown through a thin
``.qmd`` wrapper that embeds the notebook's stored outputs from a transient
``_src/`` subdirectory.

All of that is pure boilerplate derived from the notebooks, so instead of
committing it we generate it just before building and remove it again afterwards
(even if the build fails). The notebooks themselves stay in ``docs/examples/``.

Usage::

    python scripts/build_docs.py build      # generate, build, clean up
    python scripts/build_docs.py preview    # same, but serve with live reload
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"
# Notebooks are copied here at build time so great-docs stages them for the
# embed shortcode; the leading underscore stops Quarto rendering them as pages.
EMBED_DIR = EXAMPLES_DIR / "_src"
# Notebook headings follow "<Title> with `numpyro_forecast`"; drop the suffix.
TITLE_SUFFIX = " with `numpyro_forecast`"


def notebook_title(path: Path) -> str:
    """Derive a page title from a notebook's first level-1 heading.

    Parameters
    ----------
    path
        Path to the ``.ipynb`` file.

    Returns
    -------
    str
        The first markdown ``# `` heading with the trailing
        ``" with `numpyro_forecast`"`` suffix removed, or a title derived from
        the file name when no heading is found.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for line in source.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                if title.endswith(TITLE_SUFFIX):
                    title = title[: -len(TITLE_SUFFIX)]
                return title
    return path.stem.replace("_", " ").capitalize()


def generate_pages() -> list[Path]:
    """Stage the notebooks and write a ``.qmd`` wrapper for each.

    Returns
    -------
    list[pathlib.Path]
        The wrapper files that were created (the ``_src/`` copies are removed
        wholesale by :func:`clean_pages`).
    """
    EMBED_DIR.mkdir(exist_ok=True)
    created: list[Path] = []
    for notebook in sorted(EXAMPLES_DIR.glob("*.ipynb")):
        shutil.copy2(notebook, EMBED_DIR / notebook.name)
        wrapper = EXAMPLES_DIR / f"{notebook.stem}.qmd"
        title = notebook_title(notebook)
        wrapper.write_text(
            f"---\ntitle: {title}\n---\n\n{{{{< embed _src/{notebook.name} echo=true >}}}}\n",
            encoding="utf-8",
        )
        created.append(wrapper)
    return created


def clean_pages(wrappers: list[Path]) -> None:
    """Delete generated wrappers and the transient ``_src/`` copies.

    Parameters
    ----------
    wrappers
        Wrapper files to remove.
    """
    for wrapper in wrappers:
        wrapper.unlink(missing_ok=True)
    shutil.rmtree(EMBED_DIR, ignore_errors=True)


def main() -> int:
    """Generate pages, run the requested great-docs command, then clean up.

    Returns
    -------
    int
        The exit code of the great-docs command.
    """
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    wrappers = generate_pages()
    try:
        return subprocess.run(["great-docs", command], check=False).returncode  # noqa: S603, S607
    finally:
        clean_pages(wrappers)


if __name__ == "__main__":
    raise SystemExit(main())
