# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`numpyro_forecast` is a functional, JAX-native port of the ideas in Pyro's `pyro.contrib.forecast` module (the `_future`-site trick, prefix conditioning, horizon bookkeeping, backtesting), not of its class hierarchy: inference is whatever plain NumPyro you write. The design context lives in the module docstrings (`models.py`, `predictive.py`, `_offload.py`, `evaluate.py`, `surgery.py`) and the example notebooks under `docs/examples/`; read the relevant ones before making non-trivial changes.

The model-side API is a small set of **model building blocks** (`Horizon.from_data`, `innovations`, `markov_series`, `ssoe`, `predict`): plain functions that call `numpyro.sample` and `numpyro.deterministic` on your behalf inside a model function `(covariates, data=None)`. Call them "building blocks" in headings and "model functions" in prose, never "primitives" (that word means `numpyro.primitives`: `sample`, `plate`, ...).

Vector autoregression components (`var_mean`, `var_step`, `companion_matrix`, `impulse_response`) live in `var.py` and prior helpers (`minnesota_prior`) in `priors.py`; the two modules are decoupled by design (neither imports the other, they share only the `(lags, obs, obs)` coefficient layout) so a prior is always the caller's `numpyro.sample` and never baked into a recursion.

Dependencies: `arviz` is a core dependency (the ArviZ export is part of the package contract), while `matplotlib` lives in the `dev` extra only. The extras are `dataframes`, `optax`, `blackjax`, `dev`, `docs`, `cuda`, plus the umbrellas `all` (everything but cuda) and `all_cuda`; there are no dependency groups, so `uv sync --extra all` is the way to build the environment.

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
- **Host offload contract:** `device="host"` on `draw_posterior`, `forecast`, `predict_in_sample` and the pathfinder samplers returns draws as `Array | np.ndarray` (jax Arrays committed to the CPU device, or NumPy arrays when no CPU backend is initialized). Any new signature that consumes draws must accept both, and the placement contract is documented once, on `draw_posterior`; other drivers point at it.
- **`rng_key` first:** every JAX/NumPyro function that consumes randomness takes `rng_key: Array` as its first parameter (first after `self` for methods), required and positional (not keyword-only), always.
- **Integer literals:** write integers with four or more digits using underscore separators so zeros are easy to count: `1_000`, `10_000`, `1_234_567_890` (not `1000`, `1234567890`).

## Hard requirements

- Every function (public and private) has complete input and return type hints,
  checked with `ty`.
- Every public function/class has a NumPy-style docstring (ruff `D`). The preview-only `DOC` (pydoclint) rules are deliberately not selected: without `preview = true` they are inert and warn on every ruff run, and enabling preview would switch all stable rules to their preview behavior. Revisit when `DOC` stabilizes.
- **Line length:** the formatter targets 99 characters, but `E501` only fires above 120. The 100 to 120 band is an intentional grace zone for lines the formatter cannot wrap (trailing `# type: ignore` comments, long string literals); do not write new code past 99 on purpose.
- **jaxtyping:** annotate array shapes as `Float[Array, " time obs"]` with a
  **leading space** in the shape string (per the jaxtyping FAQ this turns ruff's
  `F821` into `F722`, which we ignore globally; `F821` stays active otherwise).
  Do **not** use `from __future__ import annotations` (incompatible with runtime
  type checking).

## Tests

For the tests, we use `pytest`. `make tests` runs the suite in parallel with pytest-xdist (`-n auto`); a plain `uv run pytest tests/test_foo.py` stays sequential for debugging. CI splits the suite into duration-balanced parallel jobs with [pytest-split](https://pypi.org/project/pytest-split/), driven by the committed `.test_durations` file. Refreshing that file is optional: tests missing from it are assigned the average duration, so staleness only degrades group balance, never correctness. When you add or remove notably slow tests, or CI group times drift apart, refresh it with `make store-durations` (a full sequential run) and commit the result.

`README.md` is a pytest doctest file (`--doctest-glob`): every `>>>` example in it runs in the suite, so keep the examples self-contained, prompt-formatted and under 99 characters per line, and keep the blank line before each closing fence (without it pytest reads the fence as expected output).

## Docstrings

We use Numpy-like docstrings: https://numpydoc.readthedocs.io/en/latest/format.html

### Docstring markup

The docs site renders docstrings as Quarto Markdown with great-docs (pinned to `great-docs==0.17.0`), which does **not** run any Sphinx/RST conversion for numpy-style docstrings: `:func:`, `:class:`, `:math:`, `.. math::`, `sentence::` literal blocks and `` `text <url>`_ `` links all show up literally or as dead code spans. Write the markup great-docs renders instead:

- Cross references are code spans that great-docs autolinks against the API reference: `` `~~numpyro_forecast.models.ssoe()` `` links to the `ssoe` page and displays `ssoe()` (the `~~` prefix shortens the display; a single `~` breaks the link); a bare `` `backtest()` `` links by suffix match; `` `numpyro_forecast.models.Horizon` `` links and shows the full path. Symbols outside the package (`jax.Array`, `numpyro.infer.autoguide.AutoGuide`, `functools.partial()`) cannot link, so write them as plain full paths. A symbol only links when it is listed under `reference:` in `great-docs.yml`.
- Math is `$...$` inline and a `$$` block on its own lines (blank line before and after); any docstring containing a backslash must be a raw string (`r"""`).
- Literal code blocks are fenced (```` ```python ````) after a sentence ending in a single colon, at the docstring's indentation; ruff's `docstring-code-format` reformats what parses as Python.
- Links are Markdown `[text](url)`. `>>>` doctest lines are fine (great-docs fences them).
- NumPy `See Also` sections keep bare entry names at column 0 (`name : description`); great-docs links the names itself.

`tests/test_docstring_markup.py` enforces this for every docstring under `numpyro_forecast/`, `tests/` and `scripts/` and for the markdown and code cells of the example notebooks.

## Documentation

The docs site is built with [great-docs](https://github.com/posit-dev/great-docs) (config in `great-docs.yml`) and published to https://juanitorduz.github.io/numpyro_forecast/. Build it locally with `make docs` (or `make docs-preview`).

The site is multi-version, derived entirely at build time by `scripts/build_docs.py`: the newest bare-semver git tag becomes the stable site root and the current checkout becomes `/v/dev/`, with a navbar version dropdown and `/v/stable/` plus `/v/latest/` redirect aliases (rewritten to absolute URLs post-build, since great-docs emits root-relative targets that break under the GitHub Pages project subpath). Prose and example notebooks always render from the current checkout; only the API reference is pinned per release, by pruning reference pages to the symbols in a snapshot generated on demand into the gitignored `.great-docs-cache/` (a temporary worktree of the tag, resolved by `scripts/api_snapshot.py`, which keys symbols by their `reference:` entry names so the pruning can match page stems; do not use `great-docs api-snapshot`, which records only bare top-level exports and silently falls back to the installed package for tags). The `versions:` block is injected into `great-docs.yml` for the duration of the build and restored afterwards; a hard-killed build can leave it patched, which `git restore great-docs.yml` fixes.

**Releases need no docs steps:** publish the GitHub Release as usual and the release-triggered docs build flips the stable root to the new tag automatically (the tag only exists once the release is published, so earlier builds keep serving the previous release).

The API reference is a curated list under the `reference:` section of `great-docs.yml`. **When you add (or rename/remove) a public function or class in any module, update `reference:` accordingly** by adding its `module.name` to the right section. `tests/test_docs_reference.py` enforces this: it fails if a public symbol is missing from the reference, or if a listed name no longer exists. New example notebooks just go in `docs/examples/` (the `.qmd` wrappers are generated at build time by `scripts/build_docs.py`).

### Developing example notebooks

Author notebooks with [jupytext](https://jupytext.readthedocs.io/) as a `py:percent` script rather than editing the `.ipynb` JSON by hand: it keeps clean text diffs and is lintable like any other `.py`. Write `docs/examples/<name>.py` with `# %%` cell markers, then convert and execute it in one step with `uv run jupytext --to notebook --execute docs/examples/<name>.py`, which produces `docs/examples/<name>.ipynb` with all outputs (figures, tables) embedded. Only the `.ipynb` is committed: delete the `.py` afterwards (the two files are intentionally not paired). The committed notebook stores its outputs, so the docs build never re-executes it.

Each notebook also feeds the card grid on the Examples index page: set a short plain-text `description` in the notebook-level metadata (no markdown, LaTeX, or HTML special characters; great-docs injects it into raw HTML), and tag the code cell whose figure should be the card thumbnail with a `thumbnail` cell tag. Both have fallbacks in `scripts/build_docs.py` (the intro paragraph's first sentence and the notebook's first figure, respectively), but set them explicitly so the card copy reads well and the thumbnail is a representative results plot (for example the forecast with HDI bands) rather than the raw-data plot (the exception is a dense multi-panel grid such as the hierarchical BART forecasts, which is unreadable at card size: those two notebooks keep a single-panel figure, the data overview for `hierarchical_forecasting_1` and the prior predictive check for `hierarchical_forecasting_2`, matching the stable site). Both survive re-execution; a jupytext round-trip keeps the `description` only when the notebook metadata carries `jupytext.notebook_metadata_filter: description` (write `description:` into the `.py` header and keep that filter), and keeps cell tags unless `cell_metadata_filter: -all` is set.

- Do not use `plt.show()` in notebooks.

## Writing

### No em-dashes

Do not use em-dashes (`—`) in any prose. Use the most natural alternative for the grammatical role the dash was playing: a colon for an explanation or expansion, a comma (or pair of commas) for a parenthetical aside, parentheses for a softer aside, a semicolon for a closely-related independent clause, or a full stop to start a new sentence. Pick the form that reads most cleanly; do not just substitute one punctuation mark mechanically for another.

### No hard line breaks in prose

When writing text files (`.txt`, `.md`, `.qmd`, and similar), do **not** wrap prose at a fixed column. Write each paragraph as a single long line and let the editor/renderer handle visual wrapping.

- Yes: one line per paragraph, one line per bullet.
- No: inserting newlines every 80 (or 100, or any other) characters inside a paragraph.

Exceptions: code blocks, tables, YAML front matter, and anything where the newline is semantically meaningful (e.g. markdown lists, mermaid diagrams) — keep those formatted normally.

### Latex Formulas

Use explicit name distributions like $\text{Normal}(\mu, \sigma)$ instead of $\mathcal{N}(\mu, \sigma)$.

### HDI Bands

- Use latex format r"$94\%$ HDI" instead of "94% HDI" for matplotlib plots.
- In the text, also use LaTex like $94\%$ HDI instead of "94% HDI".

### American English spelling

Use American English spelling. Do not use British English spelling.

### Technical Writing

Work on using Language ASD-STE100 simplified technical english.

## Commands

See the Makefile for the full workflow.

```bash
# Install dependencies
uv sync --extra all
# Run pre-commit hooks
prek run --all-files
# Lint and format
uv run ruff check . && uv run ruff format --check .
# Type check
uv run ty check
# Run tests
uv run pytest
# Build the documentation site (output in great-docs/_site/)
make docs
# Preview the documentation locally with live reload
make docs-preview
```

Building the docs requires [Quarto](https://quarto.org/docs/get-started/) to be installed (a system binary, separate from the Python dependencies).
