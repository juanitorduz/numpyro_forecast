.PHONY: setup activate tests prek docs docs-preview

setup:
	uv sync --all-extras

activate:
	@echo "Run: source .venv/bin/activate"

tests:
	uv run pytest

prek:
	uv run prek run --all-files

# Build the documentation site into great-docs/_site/ (the API reference is
# generated from docstrings; notebooks render from their stored outputs, so
# MCMC does not re-run at build time). The per-notebook .qmd wrappers are
# generated on the fly and removed afterwards (see scripts/build_docs.py).
docs:
	uv run python scripts/build_docs.py build

# Build and serve the site locally with live reload.
docs-preview:
	uv run python scripts/build_docs.py preview
