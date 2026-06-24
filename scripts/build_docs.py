"""Build the documentation site, generating the example wrappers on the fly.

great-docs renders ``.qmd``/``.md`` files as documentation pages but does not
render Jupyter notebooks directly. Each notebook under
``docs/examples/_notebooks/`` is therefore shown through a thin ``.qmd`` wrapper
that embeds the notebook's stored outputs. Those wrappers are pure boilerplate
derived from the notebooks, so instead of committing them we generate them just
before building and remove them again afterwards (even if the build fails).

Usage::

    python scripts/build_docs.py build      # generate wrappers, build, clean up
    python scripts/build_docs.py preview    # same, but serve with live reload
"""

import json
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"
NOTEBOOKS_DIR = EXAMPLES_DIR / "_notebooks"
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


def generate_wrappers() -> list[Path]:
    """Write a ``.qmd`` wrapper for each example notebook.

    Returns
    -------
    list[pathlib.Path]
        The wrapper files that were created.
    """
    created: list[Path] = []
    for notebook in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        wrapper = EXAMPLES_DIR / f"{notebook.stem}.qmd"
        title = notebook_title(notebook)
        wrapper.write_text(
            f"---\ntitle: {title}\n---\n\n"
            f"{{{{< embed _notebooks/{notebook.name} echo=true >}}}}\n",
            encoding="utf-8",
        )
        created.append(wrapper)
    return created


def clean_wrappers(wrappers: list[Path]) -> None:
    """Delete generated wrapper files.

    Parameters
    ----------
    wrappers
        Wrapper files to remove.
    """
    for wrapper in wrappers:
        wrapper.unlink(missing_ok=True)


def main() -> int:
    """Generate wrappers, run the requested great-docs command, then clean up.

    Returns
    -------
    int
        The exit code of the great-docs command.
    """
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    wrappers = generate_wrappers()
    try:
        return subprocess.run(["great-docs", command], check=False).returncode  # noqa: S603, S607
    finally:
        clean_wrappers(wrappers)


if __name__ == "__main__":
    raise SystemExit(main())
