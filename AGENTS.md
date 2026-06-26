# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`numpyro_forecast` is a JAX/NumPyro port of Pyro's `pyro.contrib.forecast`
module. The design context lives in the module docstrings (`forecaster.py`,
`evaluate.py`, `util.py`) and the example notebooks under `docs/examples/` — read
the relevant ones before making non-trivial changes.

## Conventions

- **Array layout:** time at axis `-2`, observation/event dim at `-1`, batch dims
  to the left (matches Pyro).
- **Train vs forecast:** a single model handles both. In-sample time latents use
  a fixed site name (`drift`); the forecast horizon uses a separate `_future`
  site so `AutoNormal` never resizes and `Predictive` draws the suffix from the
  prior. The horizon is derived from shapes (`covariates` longer than `data`).
- **Functional style:** pure model functions, explicit `PRNGKey` threading,
  vectorized latent levels (a random walk is the `jnp.cumsum` of its per-step
  drift), no global parameter store.
- **`rng_key` first:** every JAX/NumPyro function that consumes randomness takes `rng_key: Array` as its first parameter (first after `self` for methods), required and positional (not keyword-only). The one exception is `functools.singledispatch` generics, which must dispatch on their type argument: keep the dispatched generic private (dispatching on its type arg) and expose a thin public wrapper with `rng_key` first (see `draw_posterior` / `_draw_posterior_impl` in `functional.py`).
- **Integer literals:** write integers with four or more digits using underscore separators so zeros are easy to count: `1_000`, `10_000`, `1_234_567_890` (not `1000`, `1234567890`).

## Hard requirements

- Every function (public and private) has complete input and return type hints,
  checked with `ty`.
- Every public function/class has a NumPy-style docstring (ruff `D`/`DOC`).
- **jaxtyping:** annotate array shapes as `Float[Array, " time obs"]` with a
  **leading space** in the shape string (per the jaxtyping FAQ this turns ruff's
  `F821` into `F722`, which we ignore globally; `F821` stays active otherwise).
  Do **not** use `from __future__ import annotations` (incompatible with runtime
  type checking).

## Tests

For the tests, we use `pytest`.

## Docstrings

We use Numpy-like docstrings: https://numpydoc.readthedocs.io/en/latest/format.html

## Documentation

The docs site is built with [great-docs](https://github.com/posit-dev/great-docs) (config in `great-docs.yml`) and published to https://juanitorduz.github.io/numpyro_forecast/. Build it locally with `make docs` (or `make docs-preview`).

The API reference is a curated list under the `reference:` section of `great-docs.yml`. **When you add (or rename/remove) a public function or class in any module, update `reference:` accordingly** by adding its `module.name` to the right section. `tests/test_docs_reference.py` enforces this: it fails if a public symbol is missing from the reference, or if a listed name no longer exists. New example notebooks just go in `docs/examples/` (the `.qmd` wrappers are generated at build time by `scripts/build_docs.py`).

### Developing example notebooks

Author notebooks with [jupytext](https://jupytext.readthedocs.io/) as a `py:percent` script rather than editing the `.ipynb` JSON by hand: it keeps clean text diffs and is lintable like any other `.py`. Write `docs/examples/<name>.py` with `# %%` cell markers, then convert and execute it in one step with `uv run jupytext --to notebook --execute docs/examples/<name>.py`, which produces `docs/examples/<name>.ipynb` with all outputs (figures, tables) embedded. Only the `.ipynb` is committed: delete the `.py` afterwards (the two files are intentionally not paired). The committed notebook stores its outputs, so the docs build never re-executes it.

## Writing

### No em-dashes

Do not use em-dashes (`—`) in any prose. Use the most natural alternative for the grammatical role the dash was playing: a colon for an explanation or expansion, a comma (or pair of commas) for a parenthetical aside, parentheses for a softer aside, a semicolon for a closely-related independent clause, or a full stop to start a new sentence. Pick the form that reads most cleanly; do not just substitute one punctuation mark mechanically for another.

### No hard line breaks in prose

When writing text files (`.txt`, `.md`, `.qmd`, and similar), do **not** wrap prose at a fixed column. Write each paragraph as a single long line and let the editor/renderer handle visual wrapping.

- Yes: one line per paragraph, one line per bullet.
- No: inserting newlines every 80 (or 100, or any other) characters inside a paragraph.

Exceptions: code blocks, tables, YAML front matter, and anything where the newline is semantically meaningful (e.g. markdown lists, mermaid diagrams) — keep those formatted normally.

### American English spelling

Use American English spelling. Do not use British English spelling.

## Commands

See the Makefile for the full workflow.

```bash
# Install dependencies
uv sync --all-extras
# Run pre-commit hooks
prek run --all-files
# Lint and format
uv run ruff check . && uv run ruff format --check .
# Type check
uv run ty check numpyro_forecast/
# Run tests
uv run pytest
# Build the documentation site (output in great-docs/_site/)
make docs
# Preview the documentation locally with live reload
make docs-preview
```

Building the docs requires [Quarto](https://quarto.org/docs/get-started/) to be installed (a system binary, separate from the Python dependencies).
