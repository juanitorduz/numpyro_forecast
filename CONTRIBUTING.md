# Contributing

Thanks for your interest in `numpyro_forecast`!

## Development setup

```bash
uv sync --all-extras
prek install
```

## Workflow

- **Lint & format:** `uv run ruff check .` and `uv run ruff format .`
- **Type check:** `uv run ty check numpyro_forecast/`
- **Tests:** `uv run pytest`
- **Notebooks:** `uv run pytest --nbmake docs/examples/` (executes the example notebooks)
- **Docs site:** `make docs` (build) or `make docs-preview` (live preview)
- **All hooks:** `prek run --all-files`

## Example notebooks

Example notebooks live in `docs/examples/` (only the executed `.ipynb` is committed; see `AGENTS.md` for the jupytext authoring workflow). The docs build generates a `.qmd` wrapper page per notebook, and the Examples index renders each page as a card with a thumbnail image and a short description. Both are derived from the notebook itself by `scripts/build_docs.py`:

- **Description:** set a `description` key in the notebook-level metadata. In JupyterLab, use the Property Inspector (gear icon in the right sidebar) under "Advanced Tools", inside "Notebook metadata". Cursor and VS Code have no UI for notebook-level metadata: with the notebook open, run "View: Reopen Editor With..." from the command palette (`Cmd+Shift+P`), choose "Text Editor" to see the raw JSON, and add `"description": "..."` to the top-level `"metadata"` object (near the bottom of the file, next to `"kernelspec"`); or use the `nbformat` snippet below. Keep it to one or two sentences of plain text: no markdown, LaTeX, or HTML special characters (`<`, `>`, `&`), since it is injected into raw HTML. It also appears under the page title.
- **Thumbnail:** add a `thumbnail` cell tag to the code cell whose figure should be the card image. In JupyterLab, use "Common Tools" in the Property Inspector; in Cursor and VS Code, click the `...` (More Actions) menu on the cell toolbar and choose "Add Cell Tag". Pick a representative results plot, for example the forecast with its HDI bands, rather than the raw-data plot, and prefer a wide figure so the card grid stays even.

Alternatively, set both programmatically:

```python
import nbformat

nb = nbformat.read("docs/examples/my_example.ipynb", as_version=4)
nb.metadata["description"] = "One or two plain-text sentences for the card."
nb.cells[42].metadata.setdefault("tags", []).append("thumbnail")
nbformat.write(nb, "docs/examples/my_example.ipynb")
```

If they are missing, the build falls back to the first sentence of the notebook's intro paragraph and to the first figure in the notebook, so set them explicitly to control how the card reads and looks. Check the result on the Examples index with `make docs-preview`.

## Guidelines

- Every function (public and private) must have complete input and return type
  hints. Type checking is enforced with `ty`.
- Every public function and class must have a NumPy-style docstring.
- Array shapes are annotated with `jaxtyping`, with a leading space in the shape
  string (e.g. `Float[Array, " time obs"]`).
- Follow the array convention: time at axis `-2`, the observation dim at `-1`,
  batch dims to the left.
- Add tests for new functionality. Keep one logical change per pull request.
