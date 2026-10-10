# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`numpyro_forecast` is a functional, JAX-native port of the ideas in Pyro's `pyro.contrib.forecast` module (the `_future`-site trick, prefix conditioning, horizon bookkeeping, backtesting, `time_reparam`), not of its class hierarchy: inference is whatever plain NumPyro you write. The design context lives in the module docstrings (`models.py`, `reparam.py`, `predictive.py`, `_offload.py`, `evaluate.py`, `surgery.py`, `var.py`) and the example notebooks under `docs/examples/`; read the relevant ones before making non-trivial changes.

The model-side API is a small set of **model building blocks** (`Horizon.from_data`, `innovations`, `markov_series`, `ssoe`, `predict`): plain functions that call `numpyro.sample` and `numpyro.deterministic` on your behalf inside a model function `(covariates, data=None)`. Call them "building blocks" in headings and "model functions" in prose, never "primitives" (that word means `numpyro.primitives`: `sample`, `plate`, ...). They are not effect handlers and none of them returns a closure: the block owns every sample site, and the step functions a caller passes in (`Transition`/`Advance` for `markov_series`, `SSOEMean`/`SSOEUpdate` for `ssoe`, all PEP 695 `type` aliases of `Callable`, not Protocols) never call `numpyro.sample` or `numpyro.deterministic` (`ssoe` enforces this by tracing them). Distributions are passed as **instances** (`innovations(h, name, dist.Normal(0.0, scale))`, `ssoe(..., noise_dist=dist.Normal(0.0, sigma))`, `predict(h, obs_dist, prediction)` where `obs_dist` is a zero-centered instance or a link `Callable[[Array], Distribution]`), never as classes or thunks.

`time_reparam(model, "haar" | "dct")` in `reparam.py` is the port of the `time_reparam` option of Pyro's `Forecaster`: it returns `handlers.reparam(model, config=...)`, targets the non-observed continuous sample sites under the in-sample `time` plate (so `innovations` sites, never `_future`, `markov_series` scan or `ssoe` error sites), adds a `<name>_haar`/`<name>_dct` auxiliary site and makes `<name>` deterministic. The wrapped model is a single object to create once (the drivers jit with the model as a static argument) and hand to the guide, `SVI`/`MCMC`, `forecast`, `predict_in_sample`, `to_datatree` and the `model_fn` of `backtest`; nesting two `time_reparam` calls raises `ValueError`. Two deliberate differences from Pyro, documented in the module docstring: the names are literal (`"haar"` applies `HaarTransform`, `"dct"` applies `DiscreteCosineTransform`; Pyro's mapping is swapped), and discrete sites are skipped rather than failing; the `smooth`/`flip` knobs are not exposed.

Vector autoregression components (`var_mean`, `var_step`, `companion_matrix`, `impulse_response`) live in `var.py` and prior helpers (`minnesota_prior`) in `priors.py`; the two modules are decoupled by design (neither imports the other, they share only the `(lags, obs, obs)` coefficient layout) so a prior is always the caller's `numpyro.sample` and never baked into a recursion.

Dependencies: `arviz` is a core dependency (the ArviZ export is part of the package contract), and so is `equinox`, used only for `Horizon` (an `eqx.Module` so the horizon is a JAX pytree with static integer fields; do not add other Modules or a class hierarchy on top of it), while `matplotlib` lives in the `dev` extra only. The extras are `dataframes`, `optax`, `blackjax`, `dev`, `docs`, `cuda`, plus the umbrellas `all` (everything but cuda) and `all_cuda`; there are no dependency groups, so `uv sync --extra all` is the way to build the environment. The `docs` extra also carries `dynestyx>=0.5.1`, needed only by the dynestyx state space integration notebook.

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
- **Pytrees:** `Horizon` is an `eqx.Module` (static ints, `data` leaf) and `SSOEResult` a `NamedTuple`; a new result type is a `NamedTuple`, a new value type that mixes arrays and static Python scalars is an `eqx.Module` with `eqx.field(static=True)`; never `jax.tree_util.register_dataclass`, whose unflatten re-runs the beartype-wrapped `__init__` and rejects NumPy leaves.
- **Time plate names:** `PlateName` (a `StrEnum` in `models.py`: `TIME = "time"`, `TIME_FUTURE = "time_future"`) is the single spelling of the time plate names. `innovations`, `ssoe` and `time_reparam` read the enum; never spell the literal `"time"` in package code. It is documented in the reference but deliberately not re-exported from the package root.

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
- **Sampled values and `ty`:** numpyro annotates `numpyro.sample` as returning `ArrayLike`. In `numpyro_forecast/`, `tests/` and `scripts/` narrow with `jnp.asarray(numpyro.sample(...))` when the value is indexed, attribute-accessed or passed to an `Array`-typed parameter (never `typing.cast`). In the example notebooks write the plain `numpyro.sample(...)`: `pyproject.toml` ignores `not-subscriptable`, `invalid-argument-type`, `invalid-return-type` and `unresolved-attribute` under `docs/examples/**` and nothing else. Those four are therefore blind in notebooks (a wrong argument type, a misspelled attribute or a stale return annotation is not reported there), which is why a notebook's model cell must be smoke-executed after any package API change; every other rule still applies, so keep the notebooks clean otherwise.

## Tests

For the tests, we use `pytest`. `make tests` runs the suite in parallel with pytest-xdist (`-n auto`); a plain `uv run pytest tests/test_foo.py` stays sequential for debugging. CI splits the suite into duration-balanced parallel jobs with [pytest-split](https://pypi.org/project/pytest-split/), driven by the committed `.test_durations` file. Refreshing that file is optional: tests missing from it are assigned the average duration, so staleness only degrades group balance, never correctness. When you add or remove notably slow tests, or CI group times drift apart, refresh it with `make store-durations` (a full sequential run) and commit the result.

`README.md` is a pytest doctest file (`--doctest-glob`): every `>>>` example in it runs in the suite, so keep the examples self-contained, prompt-formatted and under 99 characters per line, and keep the blank line before each closing fence (without it pytest reads the fence as expected output). The README renders on three surfaces: GitHub, PyPI (`readme = "README.md"`) and the docs landing page (great-docs inlines it into `index.qmd`). Its figures live in `docs/images/` (extracted from the stored outputs of the example notebooks, under the 500 KB `check-added-large-files` limit) and are referenced by absolute `https://raw.githubusercontent.com/juanitorduz/numpyro_forecast/main/docs/images/<name>.png` URLs, because relative paths break on PyPI and on the landing page; they therefore 404 on a PR until it is merged. PyPI renders no math, so `$...$` in the README shows literally there.

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

great-docs also generates the agent-facing files at build time: `skill.md` (the [Agent Skills](https://agentskills.io/) cheat sheet, served at `/skill.md` and under `/.well-known/agent-skills/`, with an install page at `/skills.html`), `llms.txt` (an index of the `reference:` sections) and `llms-full.txt` (full signatures and docstrings). All three derive from `pyproject.toml`, the `reference:` sections and the docstrings, so a new public symbol reaches them through `reference:` alone. The hand-maintained part is the `skill:` block of `great-docs.yml` (decision table, gotchas, best practices), which encodes the conventions of this file for coding agents: **update it when a convention here changes** (a renamed driver, a new building block, a changed `rng_key` or layout rule). There is no curated `skills/numpyro_forecast/SKILL.md`; adding one would replace the generated skill entirely and silence the `skill:` block. `.well-known/` is a dot directory, so the docs workflow uploads the site artifact with `include-hidden-files: true`; without it the discovery manifest is dropped and `npx skills add <site>` returns 404. The reference and `llms-full.txt` resolve `contrib.blackjax.*` entries by attribute walk from the imported package, which is why `numpyro_forecast/__init__.py` imports `numpyro_forecast.contrib.blackjax` (the module itself pulls in no optional dependency; `tests/test_package.py` guards that no extra leaks at import time). Without that import great-docs silently falls back to static analysis for the whole reference and leaves the contrib section of `llms-full.txt` empty.

Internal design documents live in `docs/dev/` with lower-case file names (for example `docs/dev/dynestyx_integration_design.md`). They are versioned with the code but stay out of the published site: great-docs stages only the directories listed under `sections:` in `great-docs.yml` (today `docs/examples/`) plus a `user_guide/` or `user-guide/` directory at the repository root, so nothing under `docs/dev/` is rendered, and pytest's `--doctest-glob` is limited to `README.md`, so their examples never run as doctests.

### Developing example notebooks

Author notebooks with [jupytext](https://jupytext.readthedocs.io/) as a `py:percent` script rather than editing the `.ipynb` JSON by hand: it keeps clean text diffs and is lintable like any other `.py`. Write `docs/examples/<name>.py` with `# %%` cell markers, then convert and execute it in one step with `uv run jupytext --to notebook --execute docs/examples/<name>.py`, which produces `docs/examples/<name>.ipynb` with all outputs (figures, tables) embedded. Only the `.ipynb` is committed: delete the `.py` afterwards (the two files are intentionally not paired). The committed notebook stores its outputs, so the docs build never re-executes it.

Each notebook also feeds the card grid on the Examples index page: set a short plain-text `description` in the notebook-level metadata (no markdown, LaTeX, or HTML special characters; great-docs injects it into raw HTML), and tag the code cell whose figure should be the card thumbnail with a `thumbnail` cell tag. Both have fallbacks in `scripts/build_docs.py` (the intro paragraph's first sentence and the notebook's first figure, respectively), but set them explicitly so the card copy reads well and the thumbnail is a representative results plot (for example the forecast with HDI bands) rather than the raw-data plot (the exception is a dense multi-panel grid such as the hierarchical BART forecasts, which is unreadable at card size: those two notebooks keep a single-panel figure, the data overview for `hierarchical_forecasting_1` and the prior predictive check for `hierarchical_forecasting_2`, matching the stable site). Both survive re-execution; a jupytext round-trip keeps the `description` only when the notebook metadata carries `jupytext.notebook_metadata_filter: description` (write `description:` into the `.py` header and keep that filter), and keeps cell tags unless `cell_metadata_filter: -all` is set.

**The thumbnail must be a single landscape panel, about twice as wide as tall.** The card grid scales the image to the card width, so a square or tall figure (a stacked multi-panel grid, a forest plot, a facet grid) makes its card several times taller than the others and breaks the layout. Pick a one-axes figure with `figsize=(12, 6)` or `(10, 6)` (every other notebook's thumbnail has an aspect ratio between 1.3 and 2.0), typically the headline forecast with its HDI bands, and if the best results figure is a grid, tag a single-panel figure instead. `tests/test_build_docs.py` checks the aspect ratio of every committed notebook's thumbnail.

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

### Math

- Define every symbol in prose before it appears in a display formula (what is $n$, what is $N_j$). Introduce the abstract estimand before the estimator.
- Write distributions as `\text{Normal}`, `\text{Binomial}`, `\text{HalfNormal}`, never `\mathcal{N}`. Use `\text{E}` for expectations, `\text{P}` for probabilities, `\perp` for independence and `\mid` for conditioning.
- Add short remarks for the natural variants (for example, what changes if the outcome is continuous).

### Causal DAGs

Draw DAGs with graphviz, never as markdown or ASCII art. Build `dag = gr.Digraph()` (with `import graphviz as gr`), add nodes and edges, and leave `dag` as the last expression of the cell without `;`.

- Nodes are `style="filled"` with this palette: blue `#2a2eec80` for the treatment, exposure or covariates of interest; green `#328c0680` for the outcome; orange `#fa7c1780` for a secondary node of interest (mediator, collider, conditioned or selection node); `lightgray` for unobserved variables.
- Draw a conditioned node as a box with `shape="box"`. Put the role in the label on a second line, for example `label="U\n(unobserved)"` or `label="I = 1\n(responded)"`.
- Color the arrows that break an assumption red (`color="red"`). Use `dag.subgraph(name="cluster_...")` with a `label` to show two DAGs side by side.

```python
dag = gr.Digraph()
dag.node("X", label="X\n(covariates)", color="#2a2eec80", style="filled")
dag.node("Y", label="Y\n(outcome)", color="#328c0680", style="filled")
dag.node("I", label="I = 1\n(responded)", shape="box", color="#fa7c1780", style="filled")
dag.edge("X", "I")
dag.edge("X", "Y")
dag
```

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
