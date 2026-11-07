# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`numpyro_forecast` is a functional, JAX-native port of the ideas in Pyro's `pyro.contrib.forecast` module (the `_future`-site trick, prefix conditioning, horizon bookkeeping, backtesting, `time_reparam`), not of its class hierarchy: inference is whatever plain NumPyro you write. The design context lives in the module docstrings (`models.py`, `reparam.py`, `predictive.py`, `offload.py`, `evaluate.py`, `surgery.py`, `var.py`) and the example notebooks under `docs/examples/`; read the relevant ones before making non-trivial changes.

The model-side API is a small set of **model building blocks** (`Horizon.from_data`, `innovations`, `markov_series`, `ssoe`, `predict`): plain functions that call `numpyro.sample` and `numpyro.deterministic` on your behalf inside a model function `(covariates, data=None)`. Call them "building blocks" in headings and "model functions" in prose, never "primitives" (that word means `numpyro.primitives`: `sample`, `plate`, ...). They are not effect handlers and none of them returns a closure: the block owns every sample site, and the step functions a caller passes in (`Transition`/`Advance` for `markov_series`, `SSOEMean`/`SSOEUpdate` for `ssoe`, all PEP 695 `type` aliases of `Callable`, not Protocols) never call `numpyro.sample` or `numpyro.deterministic` (`ssoe` enforces this by tracing them). Distributions are passed as **instances** (`innovations(h, name, dist.Normal(0.0, scale))`, `ssoe(..., noise_dist=dist.Normal(0.0, sigma))`, `predict(h, obs_dist, prediction)` where `obs_dist` is a noise instance with location `0` or a link `Callable[[Array], Distribution]`), never as classes or thunks. `predict` shifts the location of the noise onto the predictor (`shift_loc`) and `ssoe` adds the noise draws to the step mean, so the predictor is the mean only for mean-zero noise (`Normal`, `StudentT` with `df > 1`, `Laplace`, `MultivariateNormal`): it is the mode for `Gumbel` and `AsymmetricLaplace`, whose mean is offset from `loc` (for `AsymmetricLaplace` unless `asymmetry = 1`), and `Cauchy` has no mean. The `shift_loc` docstring lists the statistic and the offset per family.

`numpyro_forecast.contrib.dynestyx.state_space` is the one optional building block: it wraps a single `dsx.sample` call, reads the observed window from the covariates like `ssoe`, registers `"forecast"` from the dynestyx Simulator rollout (requesting one extra step at the last observation, because the discrete simulator places its initial state at the first predict time) and `"obs"` from the smoothing marginals (`Smoother`) or the posterior path (`LatentPathBuilder`) pushed through the observation model, and returns the state draws as a `StateSpaceResult`. The dynestyx handler goes around every call that runs the model (fit, `draw_posterior`, the drivers), never inside the block, and the block raises when no `Filter`, `Smoother` or `LatentPathBuilder` is active (dynestyx 0.8 would otherwise pick one silently); it reads dynestyx's private `handlers._dynestyx_stack_kind` query, which the canary test in `tests/contrib/test_dynestyx.py` pins. The observation and control grids are jax arrays computed with `jax.ensure_compile_time_eval()`, so they stay concrete under the drivers' jit: the `LatentPathBuilder` indexes them in traced code (also when `numpyro.render_model` samples the path from its prior), and a `Filter` rollout and the Simulator read them on the host.

`time_reparam(model, "haar" | "dct")` in `reparam.py` is the port of the `time_reparam` option of Pyro's `Forecaster`: it returns `handlers.reparam(model, config=...)`, targets the non-observed continuous sample sites under the in-sample `time` plate (so `innovations` sites, never `_future`, `markov_series` scan or `ssoe` error sites), adds a `<name>_haar`/`<name>_dct` auxiliary site and makes `<name>` deterministic. The wrapped model (not the original) is the object to hand to the guide, `SVI`/`MCMC`, `forecast`, `predict_in_sample`, `to_datatree` and the `model_fn` of `backtest`; nesting two `time_reparam` calls raises `ValueError`. Three differences from Pyro, documented in the module docstring: the names are literal (`"haar"` applies `HaarTransform`, `"dct"` applies `DiscreteCosineTransform`; Pyro's mapping is swapped), discrete sites are skipped rather than failing, and a targeted site under a subsampled plate (a minibatched panel, including a plate the guide subsamples through `create_plates`) raises `ValueError`, because the auxiliary site is sampled outside every enclosing plate and would lose the plate's scale (Pyro keeps the plates left of time live); the `smooth`/`flip` knobs are not exposed. The rotation `T` acts on the unconstrained value `u = biject_to(support).inv(x)`, as in Pyro's and NumPyro's `UnitJacobianReparam`, so the posterior is unchanged and the unconstrained potential energy is the same function of the rotated coordinates for every support, but for a positive or bounded site the value of `numpyro.infer.util.log_density` of the wrapped model at the auxiliary value `T(u)` is the plain model's value at `x` plus the log absolute Jacobian determinant of `biject_to(support)` at `u`, and the `AutoDelta` MAP estimate is the mode in `u` instead of the mode in `x` (issue #206). For a site with a constrained support, do not write that the model's log density is unchanged without that qualification; a site with real support (such as `drift_decentered` in the example notebooks) keeps the plain claim.

Vector autoregression components (`var_mean`, `var_step`, `companion_matrix`, `impulse_response`) live in `var.py` and prior helpers (`minnesota_prior`) in `priors.py`; the two modules are decoupled by design (neither imports the other, they share only the `(lags, obs, obs)` coefficient layout) so a prior is always the caller's `numpyro.sample` and never baked into a recursion.

Dependencies: `arviz` is a core dependency (the ArviZ export is part of the package contract), and so is `equinox`, used only for `Horizon` (an `eqx.Module` so the horizon is a JAX pytree with static integer fields; do not add other Modules or a class hierarchy on top of it), while `matplotlib` lives in the `dev` extra only. The extras are `dataframes`, `optax`, `blackjax`, `dynestyx`, `dev`, `docs`, `cuda`, plus the umbrellas `all` (everything but cuda) and `all_cuda`; there are no dependency groups, so `uv sync --extra all` is the way to build the environment. The `dynestyx` extra (`dynestyx>=0.8.0,<0.9`, also pulled in by `docs`) backs `numpyro_forecast.contrib.dynestyx` and the two dynestyx example notebooks; the cap is deliberate, because dynestyx changes its handler semantics between minor releases. dynestyx declares `Requires-Python <3.14`: uv ignores that upper bound and installs it on 3.14 (CI and the docs build), pip does not.

## Conventions

- **Array layout:** time at axis `-2`, observation/event dim at `-1`, batch dims to the left (matches Pyro). The `obs` and `forecast` sites, with their enclosing plates, carry every axis of `data`: a predictor shared by `n` series is `(duration, n)` or sits in a size-`n` plate at `dim=-1`. A `(duration, 1)` predictor alone draws one forecast for all series, so `forecast` and `to_datatree` raise (issue #259); `predict_in_sample` gets no data and cannot check.
- **Train vs forecast:** a single model handles both. In-sample time latents use a fixed site name (`drift`); the forecast horizon uses a separate `_future` site so `AutoNormal` never resizes and `Predictive` draws the suffix from the prior. The horizon is derived from shapes (`covariates` longer than `data`). The drivers substitute every posterior sample site by name, so `forecast`, `predict_in_sample` and `to_datatree` reject a posterior key with values that ends in `_future`, contains `_future_` or is `forecast` (the marks of a `Predictive` call over a horizon, issue #161); `forecast` and `predict_in_sample` drop such a key when its leaf is empty (a site registered on every call, as a training run records it), and `to_datatree` still exports it in the posterior group. `predict_in_sample` and `to_datatree` also drop an `obs` key (issue #260), so the output of a `Predictive` call with `data` does not pin the in-sample draws to the observed data or land in the posterior group; a posterior that holds only `obs` raises `ValueError`. `to_datatree` also raises `ValueError` before any draw for a posterior key that contains `/` (issue #238), including an empty `_future` key, which the export keeps: `xarray.DataTree` rejects `/` in a variable name, and `numpyro.handlers.scope` joins prefix and site with `/` by default, so pass `divider="_"` to every `scope` call of a model you export. The divider changes only the names: a fit drawn under `/` exports once the model uses `divider="_"` and the posterior keys are renamed the same way. Both must change, because the drivers ignore a posterior key that matches no site and draw that site from the prior with no error (the `posterior` parameter of `to_datatree` has the details).
- **Functional style:** pure model functions, explicit `PRNGKey` threading, vectorized latent levels (a random walk is the `jnp.cumsum` of its per-step drift), no global parameter store.
- **Host offload contract:** `device="host"` on `draw_posterior`, `forecast`, `predict_in_sample` and the pathfinder samplers returns draws as `Array | np.ndarray` (jax Arrays committed to the CPU device, or NumPy arrays when no CPU backend is initialized). Any new signature that consumes draws must accept both, and the placement contract is documented once, on `draw_posterior`; other drivers point at it.
- **`rng_key` first:** every JAX/NumPyro function that consumes randomness takes `rng_key: Array` as its first parameter (first after `self` for methods), required and positional (not keyword-only), always.
- **Integer literals:** write integers with four or more digits using underscore separators so zeros are easy to count: `1_000`, `10_000`, `1_234_567_890` (not `1000`, `1234567890`).
- **Pytrees:** `Horizon` is an `eqx.Module` (static ints, `data` leaf) and `SSOEResult` a `NamedTuple`; a new result type is a `NamedTuple`, a new value type that mixes arrays and static Python scalars is an `eqx.Module` with `eqx.field(static=True)`; never `jax.tree_util.register_dataclass`, whose unflatten re-runs the beartype-wrapped `__init__` and rejects NumPy leaves.
- **Runtime type checks:** `numpyro_forecast/__init__.py` installs the jaxtyping import hook with `numpyro_forecast.typecheck.typechecker`, beartype with `BeartypeConf(is_pep484_tower=True)`: an `int` satisfies a `float` hint (and an `int` or `float` a `complex` hint) at runtime, as it does for `ty`, and a `float` never satisfies an `int` hint. The tower applies to return hints too, so a `-> float` function may return an `int` unchecked: a driver that promises `float` values converts them explicitly (`evaluate_forecast` passes every result through `scalar_metric_value` from `validation.py` and lists the stacked floating array with `.tolist()`) instead of relying on the check. `typecheck.py` is the first import inside the hook block, because the hook resolves the typechecker lazily by attribute access on the package: keep it free of function definitions. NumPy scalars are not in the tower (`np.int64` is not an `int`): a parameter that should take one is annotated `IntScalar` or `RealScalar` from `numpyro_forecast.typing` and converted at the function boundary with `operator.index` for a count or `float` for a NumPy float where the body needs a Python float (never `int()`, which truncates `10.7` silently when the checks are off with `JAXTYPING_DISABLE=1`), so the body and any memo key see Python scalars only (issue #165). `fourier_features` and `periodic_repeat` convert every such argument; `pad_future` converts `future` and passes `value` to `jnp.full` unchanged, so that an integer fill stays exact (a `float()` would round an int64 above $2^{53}$ and turn `-1` into `0` for `uint8`). All three raise `ValueError` for a negative count. Do not widen `int` or `float` globally with beartype `hint_overrides`: the runtime would accept what `ty` rejects, and NumPy overflow semantics would reach the function bodies.
- **Time plate names:** `PlateName` (a `StrEnum` in `models.py`: `TIME = "time"`, `TIME_FUTURE = "time_future"`) is the single spelling of the time plate names. `innovations`, `ssoe` and `time_reparam` read the enum; never spell the literal `"time"` in package code. It is documented in the reference but deliberately not re-exported from the package root.
- **No private imports across modules:** package modules have no leading underscore (`__init__.py` aside), and code under `numpyro_forecast/` and `scripts/` never imports an underscore-prefixed `numpyro_forecast` name from another module. When a helper is needed in a second module, rename it to a public name; if the helper is internal, put it in an internal module (`offload.py` for device placement and chunking, `validation.py` for argument checks) instead of adding it to a documented public module. Internal modules are listed in `IGNORED_MODULES` of `tests/test_docs_reference.py`, so their public names stay out of `reference:`. `tests/test_package.py::test_no_private_package_imports` enforces the rule. The rule does not cover `tests/`: a test of a private helper imports that helper directly.

## Hard requirements

- Every function (public and private) has complete input and return type hints, checked with `ty`.
- Every public function/class has a NumPy-style docstring (ruff `D`). The preview-only `DOC` (pydoclint) rules are deliberately not selected: without `preview = true` they are inert and warn on every ruff run, and enabling preview would switch all stable rules to their preview behavior. Revisit when `DOC` stabilizes.
- **Line length:** the formatter targets 99 characters, but `E501` only fires above 120. The 100 to 120 band is an intentional grace zone for lines the formatter cannot wrap (trailing `# type: ignore` comments, long string literals); do not write new code past 99 on purpose.
- **jaxtyping:** annotate array shapes as `Float[Array, " time obs"]` with a **leading space** in the shape string (per the jaxtyping FAQ this turns ruff's `F821` into `F722`, which we ignore globally; `F821` stays active otherwise). Do **not** use `from __future__ import annotations` (incompatible with runtime type checking).
- **Sampled values and `ty`:** numpyro annotates `numpyro.sample` as returning `ArrayLike`. In `numpyro_forecast/`, `tests/` and `scripts/` narrow with `jnp.asarray(numpyro.sample(...))` when the value is indexed, attribute-accessed or passed to an `Array`-typed parameter (never `typing.cast`). In the example notebooks write the plain `numpyro.sample(...)`: `pyproject.toml` ignores `not-subscriptable`, `invalid-argument-type`, `invalid-return-type` and `unresolved-attribute` under `docs/examples/**` and nothing else. Those four are therefore blind in notebooks (a wrong argument type, a misspelled attribute or a stale return annotation is not reported there), which is why a notebook's model cell must be smoke-executed after any package API change; every other rule still applies, so keep the notebooks clean otherwise.

## Tests

For the tests, we use `pytest`. `make tests` runs the suite in parallel with pytest-xdist (`-n auto`); a plain `uv run pytest tests/test_foo.py` stays sequential for debugging. CI splits the suite into duration-balanced parallel jobs with [pytest-split](https://pypi.org/project/pytest-split/), driven by the committed `.test_durations` file. Refreshing that file is optional: tests missing from it are assigned the average duration, so staleness only degrades group balance, never correctness. When you add or remove notably slow tests, or CI group times drift apart, refresh it with `make store-durations` (a full sequential run) and commit the result.

Do not add tests for notebook-specific claims: a test that compares the prose of one example notebook with its stored outputs or code cells (a number, a percentage, an interval name, a phrase) breaks on every re-execution and checks no package behavior. When you change a notebook, check its claims by hand against the stored outputs instead. The rules that apply to every notebook (`tests/test_docstring_markup.py`, `tests/test_build_docs.py`, `tests/test_examples.py`, `tests/test_notebook_hygiene.py`) stay.

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

great-docs also generates the agent-facing files at build time: `skill.md` (the [Agent Skills](https://agentskills.io/) cheat sheet, served at `/skill.md` and under `/.well-known/agent-skills/`, with an install page at `/skills.html`), `llms.txt` (an index of the `reference:` sections) and `llms-full.txt` (full signatures and docstrings). All three derive from `pyproject.toml`, the `reference:` sections and the docstrings, so a new public symbol reaches them through `reference:` alone. The hand-maintained part is the `skill:` block of `great-docs.yml` (decision table, gotchas, best practices), which encodes the conventions of this file for coding agents: **update it when a convention here changes** (a renamed driver, a new building block, a changed `rng_key` or layout rule). There is no curated `skills/numpyro_forecast/SKILL.md`; adding one would replace the generated skill entirely and silence the `skill:` block. `.well-known/` is a dot directory, so the docs workflow uploads the site artifact with `include-hidden-files: true`; without it the discovery manifest is dropped and `npx skills add <site>` returns 404. The reference and `llms-full.txt` resolve `contrib.blackjax.*` and `contrib.dynestyx.*` entries by attribute walk from the imported package, which is why `numpyro_forecast/__init__.py` imports `numpyro_forecast.contrib.blackjax` and `numpyro_forecast.contrib.dynestyx` (the modules themselves pull in no optional dependency; `tests/test_package.py` guards that no extra leaks at import time). Without that import great-docs silently falls back to static analysis for the whole reference and leaves the contrib section of `llms-full.txt` empty.

Internal design documents live in `docs/dev/` with lower-case file names (for example `docs/dev/dynestyx_integration_design.md`). They are versioned with the code but stay out of the published site: great-docs stages only the directories listed under `sections:` in `great-docs.yml` (today `docs/examples/`) plus a `user_guide/` or `user-guide/` directory at the repository root, so nothing under `docs/dev/` is rendered, and pytest's `--doctest-glob` is limited to `README.md`, so their examples never run as doctests.

### Developing example notebooks

Author notebooks with [jupytext](https://jupytext.readthedocs.io/) as a `py:percent` script rather than editing the `.ipynb` JSON by hand: it keeps clean text diffs and is lintable like any other `.py`. Write `docs/examples/<name>.py` with `# %%` cell markers, then convert and execute it in one step with `uv run jupytext --to notebook --execute docs/examples/<name>.py`, which produces `docs/examples/<name>.ipynb` with all outputs (figures, tables) embedded. Only the `.ipynb` is committed: delete the `.py` afterwards (the two files are intentionally not paired). The committed notebook stores its outputs, so the docs build never re-executes it.

Each notebook also feeds the card grid on the Examples index page: set a short plain-text `description` in the notebook-level metadata (no markdown, LaTeX, or HTML special characters; great-docs injects it into raw HTML), and tag the code cell whose figure should be the card thumbnail with a `thumbnail` cell tag. Both have fallbacks in `scripts/build_docs.py` (the intro paragraph's first sentence and the notebook's first figure, respectively), but set them explicitly so the card copy reads well and the thumbnail is a representative results plot (for example the forecast with HDI bands) rather than the raw-data plot (the exception is a dense multi-panel grid such as the hierarchical BART forecasts, which is unreadable at card size: those two notebooks keep a single-panel figure, the data overview for `hierarchical_forecasting_1` and the prior predictive check for `hierarchical_forecasting_2`, matching the stable site). Both survive re-execution; a jupytext round-trip keeps the `description` only when the notebook metadata carries `jupytext.notebook_metadata_filter: description` (write `description:` into the `.py` header and keep that filter), and keeps cell tags unless `cell_metadata_filter: -all` is set. The thumbnail figure is a single landscape panel about twice as wide as tall.

**The thumbnail must be a single landscape panel, about twice as wide as tall.** The card grid scales the image to the card width, so a square or tall figure (a stacked multi-panel grid, a forest plot, a facet grid) makes its card several times taller than the others and breaks the layout. Pick a one-axes figure with `figsize=(12, 6)` or `(10, 6)` (every other notebook's thumbnail has an aspect ratio between 1.3 and 2.0), typically the headline forecast with its HDI bands, and if the best results figure is a grid, tag a single-panel figure instead. `tests/test_build_docs.py` checks the aspect ratio of every committed notebook's thumbnail.

### Example notebook style

These conventions apply to every example notebook under `docs/examples/`. They complement the jupytext workflow, the `description` metadata and the thumbnail rule above, and the [Writing](#writing) rules below apply to every markdown cell; `m5_forecasting.ipynb` and `m5_hierarchical_state_space.ipynb` are the reference implementations.

#### Storyline

- One idea per section, in the order a reader needs it: data and exploration, the metric, the model (specification, priors, the model function), fitting, posterior, forecasts, evaluation, cost, discussion. A reader must be able to follow the notebook top to bottom without jumping ahead.
- **Every figure and every table gets a takeaway**, in a markdown cell right **after** it: what the reader should see, with the concrete numbers read off the output (ranges, which group is highest, where the band fails). A figure without a takeaway is either removed or explained. The markdown cell **before** a figure says what is plotted; the one after says what it means.
- Prose that reads a figure comes after the figure, never before it.
- The discussion closes with what the results say about modeling this kind of data, the open weaknesses, and the fairness notes (what the model knows about the horizon).
- Every number in the prose is read from the executed outputs of the same run. After re-executing, re-read the outputs and patch the prose; never leave a number from an earlier run.

#### Model specification

- Write the model as display math (`$$ \begin{align*} ... \end{align*} $$`) with a short definition of every symbol, then list **every prior as display math, one line per prior**, followed by a paragraph of intuition: what the scale means on the scale of the data (a factor on the log scale, units of a typical daily change), why it is wide or tight, what the posterior will have to decide.
- Visualize the informative priors with preliz (`pz.LogNormal(-2, 1).plot_pdf(ax=ax, color="C0", legend=None)` and the distribution in the panel title or a figure legend). Standard normals need no plot; when every prior is a standard normal, plot what matters instead (for example the implied prior on a mean through a link function).
- Run a prior predictive check, show it, and say what the priors allow before the data.
- A model-selection ladder or ablation that is not re-run in the notebook is still shown, as a figure built from the literal scores, with a sentence saying it is not re-run.

#### Helpers and code layout

- **Define each helper function right before its first use**, with a sentence that motivates it. No "shared helpers" section that defines everything at once; constants follow the same rule.
- **Every `for` loop has a blank line before and after it**, at every indentation level (the only exception is a loop that is the first statement of a block, right after the `def`/`with`/`if` line).
- **Every `print` has a blank line before and after it.**
- No long squashed blocks: separate the steps of a cell with blank lines and a one-line comment per step; comment model bodies and tensor code shape by shape (`# (days, n)`), in particular every `einsum`, gather, transpose and broadcast.
- Comment the construction of every covariate array: which column or channel is what, in which order, and the shape; print the shape.
- One topic per cell; split a cell that defines a helper and uses it, or that fits and plots.
- No dead code: no covariate channel that the model does not read, no variable that is never used, no unused import.
- Pass optax optimizers to `SVI` directly; never import `numpyro.optim._NumPyroOptim` or other private names for a type hint.
- Split long f-strings so the code stays within the 99-character line length of the hard requirements.
- No `print` for information that belongs in a figure: the step count and wall time of a fit go in the loss-plot title, not in a print above it. Prints are for shapes, counts and the summary numbers the prose quotes.
- No global `pl.Config` calls. When a table needs more rows than the default, use a local `with pl.Config(tbl_rows=n): display(df)`.
- Do not use `plt.show()`.
- Do not store environment warnings in the outputs. Filter them with `warnings.filterwarnings` in the setup cell. preliz emits "PyMC not installed" at import time, so its filter goes before `import preliz`; arviz-plots emits "When multiple credible intervals are plotted" when `aes` maps `alpha`. `tests/test_notebook_hygiene.py` rejects stored stderr that contains these two warnings or a `site-packages/` path, and checks that every relative link names an example notebook or a `reference:` entry.
- When a notebook elicits a prior with preliz's `maxent`, every call passes `mass` explicitly, not `None` (its default is `rcParams["stats.ci_prob"]`, which went from 0.94 to 0.89 between preliz releases), and the model's prior is built from the returned fit object, for example `dist.InverseGamma(concentration=fit.alpha, rate=fit.beta)`, never from copied literals (issue #198). See `tests/test_notebook_hygiene.py` for what is checked.

#### Figures and tables

- **Prefer a figure to a table whenever possible**: a per-level comparison is a bar chart, a timing table is a bar chart, a posterior summary is a forest plot. Keep a table only for small lookups that are data, not results, and keep a `print` of the summary numbers next to a figure so the prose can be checked.
- **Legends never overlap the plot and are never smaller than the default font** (no `fontsize` below 10 anywhere). Put legends outside the axes: `ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))` for a single axes; `fig.legend(handles, labels, loc="outside lower center", ncols=n)` for a constrained-layout figure with several panels; `fig.legend(..., loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=n)` below a facet grid; the distribution in the panel title for preliz plots. Never per-axes legends below adjacent panels (constrained layout squeezes the axes to nothing).
- A legend lists every band it shows. A plot with in-sample and forecast bands has entries for both (`r"in-sample $94\%$ HDI"`, `r"forecast $94\%$ HDI"`); a prior predictive plot labels its bands as prior bands. With arviz `plot_lm`, a second call on the same plot collection replaces the band artists of the first in `pc.viz`, so collect the handles of the first call before the second.
- Use `ConciseDateFormatter` on date axes that get narrow (a legend outside the axes narrows them and the default date ticks collide).
- Multi-panel figures use `layout="constrained"`, a bold `suptitle`, and `width_ratios` when the panels differ in content.

#### Notebook prose

- Link to other example notebooks and to the API reference with relative paths: `<example>.html` for another example and `../../reference/<entry>.html` for a reference page. Do not use absolute `https://juanitorduz.github.io/numpyro_forecast/` URLs. They always open the stable site root, also from the `/v/dev/` pages and from PR previews, and the stable reference has only the symbols of the last release. The example wrappers that `scripts/build_docs.py` writes set `notebook-view: false`, so Quarto renders no `_src/<name>-preview.html` copy of a notebook, where these relative links would not resolve.
- Answer a reviewer's question in the notebook, not only in the PR: if something looks odd (a warning that is filtered, a noisy loss, a parametrization), say what it is, why it is harmless or not, and show the check that proves it.
- When a decision was tested (a prior widened, a component dropped), state the measured effect in one sentence rather than an opinion.

## Writing

### Storyline and clarity

These patterns come from review rounds of notebook prose; they apply to every markdown cell and to PR descriptions.

- **Connected story, not patched sentences.** Each section opens with what it does and why the reader needs it, and closes with the step that leads to the next one ("Now that the loss is defined, we fit the model to..."). A paragraph that reads like independent facts in a row needs a through-line or a split.
- **Open with the point, explicitly.** No question headlines ("Why the margin?"), slogans ("We ask for a service level instead of a cost.") or abrupt verbs ("We minimize ...") as openers.
- **Define before use.** Define every symbol, term and concept before its first use, and do not mention one that comes later. One symbol per meaning (no `x` next to `x_t`, no `S` next to `S_t`). Write a new concept in bold where it is defined.
- **One idea per sentence.** Split a sentence that carries several numbers or several clauses. Put formulas in display math, not packed inline.
- **Bullets for lists.** Results, comparisons, steps and reasons go in bullet points, and the items keep the same order everywhere (the order in which they were introduced, which is also the color order of the figures).
- **Explain, do not assert.** Give the intuition or the derivation behind a claim (why a plan overstocks, where a formula comes from), and cite a reference for a result that the notebook does not derive.
- **Plain words.** Spell out private shorthand ("within a divergence ball", "the level enters the rate through an exponential"); prefer standard statistical terms (right-skewed, point mass at zero).
- **Read a figure, do not describe its construction.** Say why the figure is there and what it shows; the plotting function and the panel layout are in the code. Keep the takeaway, not a re-narration of every line in the plot.
- **Cut what carries no information:** sentences that hedge without content ("we read this as a consistency check"), comparisons with models the notebook does not fit, and details the reader does not need for the next step.
- **Name sections in italics** when the text refers to them, for example *The model*.

### No em-dashes

Do not use em-dashes (`—`) in any prose. Use the most natural alternative for the grammatical role the dash was playing: a colon for an explanation or expansion, a comma (or pair of commas) for a parenthetical aside, parentheses for a softer aside, a semicolon for a closely-related independent clause, or a full stop to start a new sentence. Pick the form that reads most cleanly; do not just substitute one punctuation mark mechanically for another.

### No hard line breaks in prose

When writing text files (`.txt`, `.md`, `.qmd`, and similar), do **not** wrap prose at a fixed column. Write each paragraph as a single long line and let the editor/renderer handle visual wrapping.

- Yes: one line per paragraph, one line per bullet.
- No: inserting newlines every 80 (or 100, or any other) characters inside a paragraph.

Exceptions: code blocks, tables, YAML front matter, and anything where the newline is semantically meaningful (e.g. markdown lists, mermaid diagrams): keep those formatted normally.

### Latex Formulas

Use explicit name distributions like $\text{Normal}(\mu, \sigma)$ instead of $\mathcal{N}(\mu, \sigma)$.

### HDI Bands

- Use latex format r"$94\%$ HDI" instead of "94% HDI" for matplotlib plots.
- In the text, also use LaTex like $94\%$ HDI instead of "94% HDI".

### Percentages

In notebook prose, write every percentage as LaTeX, for example $30\%$, not as `30%` in a code span or as plain text. `tests/test_notebook_hygiene.py` checks the markdown cells of every example notebook.

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
