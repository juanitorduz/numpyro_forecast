.PHONY: all figures pdf clean

all: figures pdf

figures:
	uv run python scripts/extract_figures.py
	uv run python scripts/make_schematic.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
