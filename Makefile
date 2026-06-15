.PHONY: setup activate tests prek

setup:
	uv sync --all-extras

activate:
	@echo "Run: source .venv/bin/activate"

tests:
	uv run pytest

prek:
	uv run prek run --all-files
