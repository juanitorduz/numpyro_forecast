"""Build the documentation site, generating the example pages on the fly.

great-docs renders ``.qmd``/``.md`` files as documentation pages but does not
render Jupyter notebooks directly, and it only stages a section's ``.qmd`` pages
and asset *subdirectories* into its build tree (never a loose top-level
notebook). Each notebook in ``docs/examples/`` is therefore shown through a thin
``.qmd`` wrapper that embeds the notebook's stored outputs from a transient
``_src/`` subdirectory.

The section index that great-docs auto-generates renders pages as image cards
when their front matter carries ``description`` and ``image`` fields, so each
wrapper also gets a card description (the notebook's ``metadata.description``,
falling back to the first sentence of its intro paragraph) and a thumbnail PNG
extracted from the notebook's stored outputs into a transient ``thumbnails/``
subdirectory (the first figure of the cell tagged ``thumbnail``, falling back
to the first figure in the notebook).

All of that is pure boilerplate derived from the notebooks, so instead of
committing it we generate it just before building and remove it again afterwards
(even if the build fails). The notebooks themselves stay in ``docs/examples/``.

Usage::

    python scripts/build_docs.py build      # generate, build, clean up
    python scripts/build_docs.py preview    # same, but serve with live reload
"""

import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"
# Notebooks are copied here at build time so great-docs stages them for the
# embed shortcode; the leading underscore stops Quarto rendering them as pages.
EMBED_DIR = EXAMPLES_DIR / "_src"
# Card thumbnails extracted from the notebooks' stored outputs; great-docs
# stages the whole directory as a section asset.
THUMBNAILS_DIR = EXAMPLES_DIR / "thumbnails"
# Notebook headings follow "<Title> with `numpyro_forecast`"; drop the suffix.
TITLE_SUFFIX = " with `numpyro_forecast`"
# Cell tag marking the figure to use as the notebook's index-card thumbnail.
THUMBNAIL_TAG = "thumbnail"


def load_notebook(path: Path) -> dict[str, Any]:
    """Load a notebook's JSON document.

    Parameters
    ----------
    path
        Path to the ``.ipynb`` file.

    Returns
    -------
    dict[str, Any]
        The parsed notebook document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_title(notebook: dict[str, Any], path: Path) -> str:
    """Derive a page title from a notebook's first level-1 heading.

    Parameters
    ----------
    notebook
        The parsed notebook document.
    path
        Path to the ``.ipynb`` file, used as a fallback title source.

    Returns
    -------
    str
        The first markdown ``# `` heading with the trailing
        ``" with `numpyro_forecast`"`` suffix removed, or a title derived from
        the file name when no heading is found.
    """
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


def _strip_markdown(text: str) -> str:
    """Reduce inline markdown to plain text for use in raw HTML.

    Parameters
    ----------
    text
        Markdown text.

    Returns
    -------
    str
        The text with links reduced to their label and emphasis/code markers
        removed (great-docs injects card descriptions into HTML unescaped).
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("`", "").replace("*", "")


def _intro_first_sentence(notebook: dict[str, Any]) -> str:
    """Extract the first sentence of the intro paragraph after the H1 heading.

    Parameters
    ----------
    notebook
        The parsed notebook document.

    Returns
    -------
    str
        The first sentence of the first non-heading paragraph in the first
        markdown cell, stripped of inline markdown, or ``""`` if none exists.
    """
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for paragraph in re.split(r"\n\s*\n", source):
            paragraph = " ".join(paragraph.split())
            if not paragraph or paragraph.startswith("#"):
                continue
            sentence = re.split(r"(?<=[.!?])\s", _strip_markdown(paragraph), maxsplit=1)[0]
            return sentence.strip()
        return ""
    return ""


def notebook_description(notebook: dict[str, Any]) -> str:
    """Derive the index-card description for a notebook.

    Parameters
    ----------
    notebook
        The parsed notebook document.

    Returns
    -------
    str
        The notebook's ``metadata.description`` if set (plain text by
        convention: great-docs injects it into HTML unescaped), otherwise the
        first sentence of the intro paragraph, or ``""`` if neither exists.
    """
    description = notebook.get("metadata", {}).get("description", "")
    if description:
        return str(description).strip()
    return _intro_first_sentence(notebook)


def _cell_image(cell: dict[str, Any]) -> str | None:
    """Return the base64 payload of a cell's first PNG output.

    Parameters
    ----------
    cell
        A notebook cell.

    Returns
    -------
    str | None
        The base64-encoded PNG (nbformat stores it as a string or a list of
        strings), or ``None`` if the cell has no PNG output.
    """
    for output in cell.get("outputs", []):
        png = output.get("data", {}).get("image/png")
        if png is not None:
            return "".join(png) if isinstance(png, list) else png
    return None


def extract_thumbnail(notebook: dict[str, Any], dest: Path) -> bool:
    """Write a notebook's thumbnail figure to ``dest``.

    Uses the first PNG output of the first cell tagged ``thumbnail``, falling
    back to the first PNG output in the notebook.

    Parameters
    ----------
    notebook
        The parsed notebook document.
    dest
        Path of the PNG file to write.

    Returns
    -------
    bool
        ``True`` if a figure was found and written, ``False`` otherwise (the
        page then falls back to great-docs' plain link list).
    """
    cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
    tagged = [c for c in cells if THUMBNAIL_TAG in c.get("metadata", {}).get("tags", [])]
    for cell in [*tagged, *cells]:
        payload = _cell_image(cell)
        if payload is not None:
            dest.write_bytes(base64.b64decode(payload))
            return True
    return False


def generate_pages() -> list[Path]:
    """Stage the notebooks and write a ``.qmd`` wrapper for each.

    Returns
    -------
    list[pathlib.Path]
        The wrapper files that were created (the ``_src/`` and ``thumbnails/``
        directories are removed wholesale by :func:`clean_pages`).
    """
    EMBED_DIR.mkdir(exist_ok=True)
    THUMBNAILS_DIR.mkdir(exist_ok=True)
    created: list[Path] = []
    for path in sorted(EXAMPLES_DIR.glob("*.ipynb")):
        notebook = load_notebook(path)
        shutil.copy2(path, EMBED_DIR / path.name)
        front_matter = [f"title: {json.dumps(notebook_title(notebook, path))}"]
        description = notebook_description(notebook)
        if description:
            front_matter.append(f"description: {json.dumps(description)}")
        if extract_thumbnail(notebook, THUMBNAILS_DIR / f"{path.stem}.png"):
            front_matter.append(f"image: thumbnails/{path.stem}.png")
        wrapper = EXAMPLES_DIR / f"{path.stem}.qmd"
        wrapper.write_text(
            "---\n"
            + "\n".join(front_matter)
            + f"\n---\n\n{{{{< embed _src/{path.name} echo=true >}}}}\n",
            encoding="utf-8",
        )
        created.append(wrapper)
    return created


def clean_pages(wrappers: list[Path]) -> None:
    """Delete generated wrappers and the transient ``_src``/``thumbnails`` copies.

    Parameters
    ----------
    wrappers
        Wrapper files to remove.
    """
    for wrapper in wrappers:
        wrapper.unlink(missing_ok=True)
    shutil.rmtree(EMBED_DIR, ignore_errors=True)
    shutil.rmtree(THUMBNAILS_DIR, ignore_errors=True)


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
