# Issue #116 review follow-up: design and implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the five review comments in [issue #116](https://github.com/juanitorduz/numpyro_forecast/issues/116): make `Horizon` and `SSOEResult` JAX pytrees, let `innovations()` take a distribution instance, replace the `(value, carry_fn)` step protocol of `ssoe()` and `markov_series()` with two plain functions, remove the `cast`/`jnp.asarray` narrowing noise from the example notebooks, and record the decision to keep `h` as the first argument.

**Architecture:** All model-side changes land in `numpyro_forecast/models.py` (plus `var.py` for the VAR step factory) as clean cutovers with no compatibility shims. `SSOEResult` becomes a `typing.NamedTuple`; `Horizon` becomes an `equinox.Module` with static integer fields; `innovations()` takes a `numpyro.distributions.Distribution`; `markov_series()` takes `transition(carry, x_t) -> dist_t` plus an optional `advance(carry, z_t, x_t) -> carry`; `ssoe()` takes `mean(carry, x_t) -> mu_t` and `update(carry, y_t, eps_t, x_t) -> carry`. Notebooks are edited in place (no re-execution: every rewrite consumes the PRNG identically, see Evidence). The type-checker noise in notebooks is removed by a `[[tool.ty.overrides]]` block scoped to `docs/examples/**`; a separate upstream numpyro track narrows the return types at the source.

**Tech Stack:** Python 3.12+, jax 0.11, numpyro 0.22, jaxtyping + beartype import hook, equinox 0.13 (new core dependency), ruff 0.16.9, ty 0.0.84, nbformat/nbclient (already in the `dev` extra via jupytext/nbmake).

**Spec:** the issue text plus the decisions in the next section. This document is both the design record and the plan.

## Global Constraints

- `AGENTS.md` rules apply to every task: complete type hints checked by `ty`; NumPy docstrings on every public symbol; jaxtyping shape strings with a leading space; no `from __future__ import annotations`; `rng_key` first; integers with four or more digits use underscores; no em-dashes, no hard-wrapped prose, American English.
- Great-docs docstring markup only (no RST roles): cross references are code spans such as `` `~~numpyro_forecast.models.ssoe()` ``; `tests/test_docstring_markup.py` enforces this for `numpyro_forecast/`, `tests/`, `scripts/` and the notebooks.
- Every public symbol must appear under `reference:` in `great-docs.yml` (`tests/test_docs_reference.py` fails otherwise).
- Clean cutover: migrate every caller (package, tests, README doctest, notebooks); delete obsolete aliases (`SSOEStep`, the test-only `CarryFn`), no deprecated paths.
- Notebooks are edited in place with `nbformat` (procedure below), never re-executed; outputs stay as committed. Every code rewrite in this plan is op-for-op identical to the code it replaces (same expressions on the same inputs, same PRNG consumption), which is what keeps the stored outputs valid; a rewrite that is only approximately equal (for example reconstructing a mean as `y_t - eps_t`) is not allowed.
- Run the gates once per task, at the end of the task: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest <touched tests>`; the full `make tests` at the end of Task 7.
- Do not comment on the GitHub issue as part of this plan (the argument-order decision is recorded here only).

## Decisions and evidence

| # | Review comment | Decision | Evidence |
| --- | --- | --- | --- |
| 1 | `Horizon`/`SSOEResult` are opaque to JAX | `SSOEResult` becomes a `NamedTuple`; `Horizon` becomes an `eqx.Module` with `t_obs`, `future`, `duration` as `eqx.field(static=True)`; `equinox` joins the core dependencies. | Under the package's beartype import hook, `jax.tree_util.register_dataclass` rebuilds instances through the wrapped `__init__` on every unflatten, so `jax.tree.map(np.asarray, h)` raises `TypeCheckError` (probe on jax 0.11.0, jaxtyping hook, beartype 0.22). `eqx.Module` unflattens through `object.__new__` and passed `jit`, `vmap`, `eval_shape`, NumPy tree maps, `__check_init__` and the beartype constructor check. Equinox 0.13.8 is already in `uv.lock` (transitive via dynestyx in the `docs` extra); its own dependencies are jax, jaxtyping, typing-extensions, wadler-lindig; importing it after jax costs 0.04 s. |
| 2 | `innovations` takes `lambda: dist` | Take a `Distribution` instance, no callable form. | One `dist.Normal(0.0, scale)` instance sampled at `drift` under `plate("time")` and at `drift_future` under `plate("time_future")`, with and without `LocScaleReparam(0)`, produces traces identical to the lazy form and leaves the instance's `batch_shape` untouched (`plate` calls `fn.expand`, which returns a new object). Every call site in the repo passes `lambda: dist.Normal(0.0, x)`. |
| 3 | Step functions read "non-human" | Two-function protocol: `ssoe(h, name, y, init_carry, mean, update, noise_dist)`, `markov_series(..., transition, advance=None)`. | Every `markov_series` transition in the repo returns `lambda z: z` except the VAR window shift; every `ssoe` step's `carry_fn` closes over the step scope and can be written as a top-level `update(carry, y_t, eps_t, x_t)` because `ssoe` already hands both `y_t` and `eps_t` to it. |
| 4 | `cast("Array", numpyro.sample(...))` noise | Local: a `[[tool.ty.overrides]]` block for `docs/examples/**` that ignores the four rules the `ArrayLike` union triggers, then strip every `cast("Array", ...)` and `jnp.asarray(numpyro.sample(...))` wrapper from the notebooks. Package and test `.py` files keep `jnp.asarray(...)` narrowing until the upstream fix ships. Upstream: a numpyro PR narrowing `numpyro.sample`/`numpyro.deterministic`. | numpyro 0.22.0 annotates `sample(...) -> ArrayLike` and `deterministic(...) -> ArrayLike` (`numpyro/primitives.py`); PR #2206 narrowed distribution methods only, and `DistributionMeta.__call__ -> Any` makes every `dist.Normal(...)` `Any` to ty. `uv run ty check` checks `docs/examples/*.ipynb` by default (verified with `-v`). Stripping every wrapper from the notebooks fires exactly `not-subscriptable` (30 diagnostics, 6 sites), `invalid-argument-type` (13), `invalid-return-type` (4), `unresolved-attribute` (1); nothing else. The override syntax was validated on ty 0.0.83 (the ignored rule vanished, an unrelated `invalid-assignment` in the same notebook still fired). |
| 5 | Site name is the second argument | Keep `h` first. No issue reply. | `h` is the receiver every block shares, including `predict(h, obs_dist, prediction)`, which has no name; name-first would put `h` second in three blocks and first in one. |

Tooling: ruff 0.16.9 and ty 0.0.84 are the latest releases (PyPI, checked 2026-09-29); both pass on the current tree. Note that ruff (the current pin included) formats Python code blocks inside Markdown: `uv run ruff format --check .` already flags one line of `docs/dev/dynestyx_integration_design.md`, which CI never saw because the `ruff-format` hook only receives Python and Jupyter files; this plan's own fences are formatted with it. The pre-commit revs (`ruff-pre-commit v0.16.3`, `ty-pre-commit v0.0.72`) lag the pins and move to the same versions.

## Notebook editing procedure (shared by Tasks 1, 3, 4, 5, 6)

Notebooks are 4 to 18 MB JSON files; edit only cell `source` and write back with `nbformat` so the JSON layout stays canonical. Three committed notebooks are not byte-stable under an `nbformat` round trip even with no source change (`var.ipynb` stores non-ASCII output as `\uXXXX` escapes, `forecasting_univariate.ipynb` and `inference_methods_comparison.ipynb` store a markdown source as a single string and lack a cell `id`; ruff's notebook writer produces the same bytes as `nbformat`), so Task 0 normalizes them once in a source-free commit and every later notebook diff is limited to the edited lines. Use a throwaway script under `/tmp` (never commit it):

```python
# /tmp/nb_rewrite.py: apply a source-to-source rewrite to the cells of a notebook.
import sys
from collections.abc import Callable

import nbformat


def rewrite_notebook(
    path: str, rewrite: Callable[[str], str], *, cell_type: str = "code", expect: int | None = 1
) -> int:
    nb = nbformat.read(path, as_version=4)
    changed = 0
    for cell in nb.cells:
        if cell.cell_type != cell_type:
            continue
        new = rewrite(cell.source)
        if new != cell.source:
            cell.source = new
            changed += 1
    if expect is not None and changed != expect:
        msg = f"{path}: expected {expect} changed {cell_type} cell(s), got {changed}"
        raise SystemExit(msg)
    if changed:
        nbformat.write(nb, path)
    return changed
```

Each task below supplies its `rewrite` function (regex for mechanical edits, literal `str.replace` on the verbatim cell text for the step-function rewrites; the verbatim "before" text is quoted in the task and a silent no-op is impossible because `expect` raises). Markdown cells that describe a changed protocol are rewritten the same way with `cell_type="markdown"`. After editing, run `uv run ruff check --fix docs/examples && uv run ruff format docs/examples && uv run ty check` (ruff lints and formats `.ipynb`; the `--fix` removes imports that became unused, such as `from typing import cast`).

Smoke test for a rewritten model cell (run per notebook whose model changed, throwaway): truncate the notebook after the cell that defines the model, append a cell that traces the model on the notebook's own training arrays and asserts the expected sites, and execute the truncated copy with `nbclient`:

```python
# /tmp/nb_smoke.py <notebook> <model_cell_index> "<smoke cell source>"
import sys

import nbformat
from nbclient import NotebookClient

path, idx, smoke = sys.argv[1], int(sys.argv[2]), sys.argv[3]
nb = nbformat.read(path, as_version=4)
nb.cells = nb.cells[: idx + 1] + [nbformat.v4.new_code_cell(smoke)]
NotebookClient(nb, kernel_name="python3", timeout=1_200).execute()
print("smoke OK:", path)
```

The smoke cell is written per notebook from the names the notebook uses. Model-cell indices (all cells, zero-based) and names: `arma.ipynb` cell 13 (`arma_1_1`, `covariates`, `data`, `duration`), `croston.ipynb` cell 12, `tsb.ipynb` cell 12, `availability_tsb.ipynb` cell 15, `exponential_smoothing_state_space.ipynb` cell 11, `censored_demand.ipynb` cell 10, `var.ipynb` cell 10, `fresh_retail_stockout.ipynb` cell 45 (read the model function and array names from each cell before writing the smoke cell). For `arma.ipynb`:

```python
from numpyro.handlers import seed, trace

tr = trace(seed(arma_1_1, jax.random.PRNGKey(0))).get_trace(covariates, data)
assert "eps_future" in tr
assert tr["forecast"]["value"].shape[-2] == duration - data.shape[-2]
```

(`fresh_retail_stockout.ipynb` loads its data from Hugging Face; its smoke run needs network access and takes a few minutes. Run it once at the end of Task 5.)

---

### Task 0: Bump ruff and ty

**Files:**
- Modify: `pyproject.toml:65-66` (dev extra pins)
- Modify: `.pre-commit-config.yaml:6,14` (hook revs)
- Modify: `uv.lock` (via `uv lock`)
- Modify: `docs/dev/dynestyx_integration_design.md:491` (one code block reformatted by ruff 0.16.9)

**Interfaces:**
- Produces: the tool versions every later task's gates run with.

- [ ] **Step 1: Pin the new versions**

In `pyproject.toml` replace `"ruff==0.16.8"` with `"ruff==0.16.9"` and `"ty==0.0.83"` with `"ty==0.0.84"`. In `.pre-commit-config.yaml` set `rev: v0.16.9` for `astral-sh/ruff-pre-commit` and `rev: v0.0.84` for `astral-sh/ty-pre-commit`.

- [ ] **Step 2: Re-lock and sync**

Run: `uv lock && uv sync --extra all`
Expected: the lock updates `ruff` and `ty` only.

- [ ] **Step 3: Apply the new Markdown code-block formatting**

Run: `uv run ruff format .`
Expected: exactly one file changed, `docs/dev/dynestyx_integration_design.md` (the `state_space_series(...)` call at line 491 wraps to three lines). Then `uv run ruff check . && uv run ruff format --check . && uv run ty check` all pass.

- [ ] **Step 4: Verify the hooks**

Run: `prek run --all-files` (with `SKIP=no-commit-to-branch` if on `main`)
Expected: all hooks pass with the new revs.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .pre-commit-config.yaml uv.lock docs/dev/dynestyx_integration_design.md
git commit -m "chore: bump ruff to 0.16.9 and ty to 0.0.84"
```

- [ ] **Step 6: Normalize the three notebooks that do not round-trip byte-identically**

Round-trip `docs/examples/var.ipynb`, `docs/examples/forecasting_univariate.ipynb` and `docs/examples/inference_methods_comparison.ipynb` through `nbformat.read`/`nbformat.write` with no source change (see the procedure section for why), confirm `git diff --stat` touches only those three files and that `python -c "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]"` parses them, then run `uv run ruff check docs/examples && uv run ruff format --check docs/examples && uv run ty check` and commit separately:

```bash
git add docs/examples/var.ipynb docs/examples/forecasting_univariate.ipynb docs/examples/inference_methods_comparison.ipynb
git commit -m "chore: normalize notebook JSON through nbformat (no source change)"
```

---

### Task 1: `SSOEResult` as a `NamedTuple`

**Files:**
- Modify: `numpyro_forecast/models.py:15,348-368` (`dataclass` import, class body)
- Test: `tests/test_ssoe.py`
- Modify: `docs/dev/dynestyx_integration_design.md:135` (the `StateSpaceResult` code block) and `docs/examples/dynestyx_integration.ipynb` cell 4 (the notebook defines the same `@dataclass(frozen=True) class StateSpaceResult` verbatim, per the design doc's line 604)

**Interfaces:**
- Produces: `class SSOEResult(NamedTuple)` with fields `mu`, `mu_future`, `y_future` (unchanged names and shapes); a pytree with three leaves; positional unpacking `mu, mu_future, y_future = ssoe(...)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ssoe.py` (next to the shape tests; `ARMA_SSOE` and `_capture` already exist at the top of the file):

```python
def test_ssoe_result_is_a_pytree_and_unpacks() -> None:
    model, box = _capture(_arma_ssoe_body)
    covariates = _series(12)
    get_trace(model, covariates, covariates[:10])
    (r,) = box
    mu, mu_future, y_future = r
    assert mu.shape == (10, 1) and mu_future.shape == (2, 1) and y_future.shape == (2, 1)
    assert len(jax.tree.leaves(r)) == 3
    doubled = jax.tree.map(lambda x: 2 * x, r)
    assert isinstance(doubled, SSOEResult)
    assert jnp.array_equal(doubled.mu, 2 * r.mu)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ssoe.py::test_ssoe_result_is_a_pytree_and_unpacks -v`
Expected: FAIL (`cannot unpack non-iterable SSOEResult object` or `len(jax.tree.leaves(r)) == 0`).

- [ ] **Step 3: Replace the dataclass**

In `numpyro_forecast/models.py` change the import line `from typing import cast` to `from typing import NamedTuple, cast` and drop `from dataclasses import dataclass` once Task 2 has also removed its use (leave the import for now if `Horizon` still needs it). Replace the class:

```python
class SSOEResult(NamedTuple):
    """The means and sampled future values produced by `ssoe()`.

    A named tuple, hence a JAX pytree with three array leaves: it unpacks as
    ``mu, mu_future, y_future = ssoe(...)`` and passes through ``jax.tree.map``,
    ``jax.jit`` and ``jax.vmap`` unchanged.

    Attributes
    ----------
    mu
        In-sample one-step-ahead means, shape ``(*batch, t_obs, obs)``: the
        predictor the caller writes its likelihood against.
    mu_future
        Forecast-horizon one-step-ahead means, shape ``(*batch, future, obs)``
        (a size-0 time axis while training).
    y_future
        Sampled future values ``mu_future + eps``, shape ``(*batch, future, obs)``
        (a size-0 time axis while training); the caller registers them as the
        ``"forecast"`` deterministic when ``h.future > 0``.
    """

    mu: Float[Array, " *batch t_obs obs"]
    mu_future: Float[Array, " *batch future obs"]
    y_future: Float[Array, " *batch future obs"]
```

The two constructor calls at the end of `ssoe()` (`SSOEResult(mu=mu, mu_future=empty, y_future=empty)` and the keyword form after the forecast scan) are unchanged. Known and accepted: under the jaxtyping import hook a frozen dataclass had its `__init__` shape-checked (the cross-field `*batch`/`future` binding), and a `NamedTuple` does not; `_validate_future_errors` and `_validate_ssoe_mean` already enforce those shapes before construction, so no behavior is lost.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ssoe.py tests/test_var.py tests/test_docs_reference.py tests/test_docstring_markup.py -v`
Expected: PASS.

- [ ] **Step 5: Align the dynestyx design note and notebook**

In `docs/dev/dynestyx_integration_design.md`, at the `@dataclass(frozen=True) class StateSpaceResult:` block (line 135), replace the decorator line and class line with `class StateSpaceResult(NamedTuple):` and add one sentence after the code block: "It is a `NamedTuple`, like `SSOEResult`, so it is a JAX pytree." Then apply the same source-only change to `docs/examples/dynestyx_integration.ipynb` (procedure section; no output depends on the class definition): in cell 4 replace the two lines `@dataclass(frozen=True)` and `class StateSpaceResult:` with `class StateSpaceResult(NamedTuple):`; in cell 2 replace `from dataclasses import dataclass` with `from typing import NamedTuple` next to the existing `from typing import Any` (ruff `I` sorts them; `ruff check --fix` merges the two `typing` imports). `rewrite_notebook` is called twice with `expect=1`.

- [ ] **Step 6: Commit**

```bash
git add numpyro_forecast/models.py tests/test_ssoe.py docs/dev/dynestyx_integration_design.md docs/examples/dynestyx_integration.ipynb
git commit -m "feat: SSOEResult is a NamedTuple (JAX pytree)"
```

---

### Task 2: `Horizon` as an `equinox.Module`

**Files:**
- Modify: `pyproject.toml:33-40` (add `"equinox>=0.13"` to `dependencies`), `uv.lock`
- Modify: `numpyro_forecast/models.py:13-31,46-130`
- Modify: `AGENTS.md` ("Dependencies:" paragraph in "What this is")
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `class Horizon(eqx.Module)` with `data: Array | None` (the only leaf) and static `t_obs`, `future`, `duration`; the same constructor keywords, `from_data` classmethod, `zero_data` property and error messages as today.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py` (add `import jax` and `import numpy as np` to the imports):

```python
def test_horizon_is_a_pytree_with_static_shapes() -> None:
    h = Horizon.from_data(jnp.zeros((25, 0)), jnp.ones((20, 1)))
    # data is the only leaf; the three ints are static metadata.
    (leaf,) = jax.tree.leaves(h)
    assert leaf is h.data

    def forecast_zeros(h: Horizon) -> Array:
        # future is static metadata, so it is a legal shape inside jit and vmap.
        assert h.data is not None
        return jnp.zeros((h.future, 1)) + h.data.sum()

    assert jax.jit(forecast_zeros)(h).shape == (5, 1)
    batched = Horizon.from_data(jnp.zeros((3, 25, 0)), jnp.ones((3, 20, 1)))
    assert jax.vmap(forecast_zeros)(batched).shape == (3, 5, 1)


def test_horizon_unflattens_with_host_leaves() -> None:
    # The beartype hook checks construction, not tree_unflatten: NumPy leaves are fine.
    h = Horizon.from_data(jnp.zeros((25, 0)), jnp.ones((20, 1)))
    host = jax.tree.map(np.asarray, h)
    assert isinstance(host.data, np.ndarray)
    assert (host.t_obs, host.future, host.duration) == (20, 5, 25)
    with pytest.raises(TypeError):
        Horizon(data=np.ones((20, 1)), t_obs=20, future=5, duration=25)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_models.py -k horizon -v`
Expected: the two new tests FAIL. On the current frozen dataclass `jax.tree.leaves(h)` is `[h]` (an unregistered object is one opaque leaf), so `assert leaf is h.data` fails in the first test, and `jax.tree.map(np.asarray, h)` in the second returns a 0-d object array rather than a `Horizon`, so `host.data` raises `AttributeError`. The eight existing `test_horizon_*` tests keep passing.

- [ ] **Step 3: Add the dependency**

In `pyproject.toml` add `"equinox>=0.13",` to `dependencies` (alphabetical: after `"beartype>=0.22"`). Run `uv lock && uv sync --extra all`. Expected: the lock records `equinox` under `numpyro-forecast`'s dependencies; no other version changes.

- [ ] **Step 4: Replace the dataclass**

In `numpyro_forecast/models.py` add `import equinox as eqx` (after `import jax.numpy as jnp`), remove `from dataclasses import dataclass`, and replace the class:

```python
class Horizon(eqx.Module):
    """The train/forecast split for a single model call.

    An immutable value derived once per model call from the covariate and data
    shapes by `from_data()`; every building block (`innovations()`,
    `markov_series()`, `ssoe()`, `predict()`) takes it as its first
    argument.

    A JAX pytree (an `equinox.Module`): ``data`` is the only leaf, while
    ``t_obs``, ``future`` and ``duration`` are static metadata. A jitted function
    that takes a `Horizon` can therefore use the three integers as shapes and
    recompiles once per horizon length.

    Attributes
    ----------
    data
        Observed in-sample data with time at axis ``-2`` (``None`` during pure
        prior sampling).
    t_obs
        Number of observed (in-sample) time steps ``t``.
    future
        Number of forecast time steps ``f`` (``0`` while training).
    duration
        Total horizon length ``t + future`` (in time steps).
    """

    data: Array | None
    t_obs: int = eqx.field(static=True)
    future: int = eqx.field(static=True)
    duration: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate that the horizon fields are internally consistent."""
        if self.t_obs < 0 or self.future < 0:
            msg = "t_obs and future must be non-negative"
            raise ValueError(msg)
        if self.duration != self.t_obs + self.future:
            msg = "duration must equal t_obs + future"
            raise ValueError(msg)
```

Keep `zero_data` and `from_data` verbatim. `__check_init__` (not `__post_init__`) is equinox's post-construction hook: it runs after `__init__` on construction and is skipped on `tree_unflatten`, which is what keeps `jax.tree.map(np.asarray, h)` working.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_models.py tests/test_markov.py tests/test_ssoe.py tests/test_package.py tests/test_docstring_markup.py -v`
Expected: PASS, including the existing `test_horizon_rejects_inconsistent_duration` and `test_horizon_rejects_negative_future` (same messages) and `test_base_import_no_extras` (equinox is not in the leak list).

- [ ] **Step 6: Document the dependency**

In `AGENTS.md`, "What this is", extend the "Dependencies:" paragraph: after "`arviz` is a core dependency (the ArviZ export is part of the package contract)", add ", and so is `equinox`, used only for `Horizon` (an `eqx.Module` so the horizon is a JAX pytree with static integer fields; do not add other Modules or a class hierarchy on top of it)". Also add to the "Conventions" list: "**Pytrees:** `Horizon` is an `eqx.Module` (static ints, `data` leaf) and `SSOEResult` a `NamedTuple`; a new result type is a `NamedTuple`, a new value type that mixes arrays and static Python scalars is an `eqx.Module` with `eqx.field(static=True)`; never `jax.tree_util.register_dataclass`, whose unflatten re-runs the beartype-wrapped `__init__` and rejects NumPy leaves."

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock numpyro_forecast/models.py tests/test_models.py AGENTS.md
git commit -m "feat: Horizon is an equinox Module (pytree with static shapes)"
```

---

### Task 3: `innovations()` takes a distribution instance

**Files:**
- Modify: `numpyro_forecast/models.py:132-188` (`_sample_time_block`, `innovations`)
- Modify: `README.md:82-84,122`
- Modify: `tests/conftest.py:212`, `tests/example_models.py:51-56,110-115`, `tests/test_models.py:108-110,165,198,209,233,246,260`, `tests/test_reparam.py:95,130`, `tests/contrib/test_blackjax.py:63-65`
- Modify (notebooks, in place): `forecasting_univariate`, `inference_methods_comparison`, `dynestyx_integration`, `hierarchical_forecasting_1`, `hierarchical_forecasting_2`, `fresh_retail_stockout`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `innovations(h: Horizon, name: str, prior: dist.Distribution, *, reparam: Reparam | None = None) -> Array`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
def test_innovations_shares_one_distribution_across_both_time_plates() -> None:
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, jnp.zeros((20, 1)))
    prior = dist.Normal(0.0, 0.3)

    def body() -> None:
        innovations(h, "drift", prior)

    tr = trace(seed(body, random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert prior.batch_shape == ()  # the plates expand a copy, never the instance
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_models.py::test_innovations_shares_one_distribution_across_both_time_plates -v`
Expected: FAIL with `KeyError: 'rng_key'`: a numpyro `Distribution` instance is callable (`Distribution.__call__` is the `rng_key`-taking sampler), so beartype's shallow `Callable` check passes and `dist_fn()` reaches that method with no key. Not a `TypeError`.

- [ ] **Step 3: Change the signature**

```python
def _sample_time_block(
    site: str,
    size: int,
    plate_name: str,
    prior: dist.Distribution,
    reparam: Reparam | None,
) -> Array:
    """Sample a single time block of ``size`` steps under a time plate at axis ``-2``."""
    with ExitStack() as stack:
        if reparam is not None:
            stack.enter_context(numpyro.handlers.reparam(config={site: reparam}))
        stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
        return cast(Array, numpyro.sample(site, prior))


def innovations(
    h: Horizon,
    name: str,
    prior: dist.Distribution,
    *,
    reparam: Reparam | None = None,
) -> Array:
```

Update the docstring parameter entry:

```text
    prior
        The per-step prior distribution, shared by the in-sample and forecast
        sites (each time plate expands a copy; the instance is never mutated).
        Its batch shape is the per-step shape, for example ``()`` for a scalar
        latent or ``(n_series,)`` under an enclosing series plate; the time axis
        comes from the plate.
```

and the two calls in the body (`prefix = _sample_time_block(name, h.t_obs, PlateName.TIME, prior, reparam)`, likewise for the suffix).

- [ ] **Step 4: Migrate the `.py` callers and the README**

Replace every `lambda: dist.Normal(...)` argument with the instance. Verbatim edits:

- `README.md:82-84`: `drift = innovations(h, "drift", dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0))` (one line, 94 characters, fits the 99 limit of the doctest).
- `README.md:122`: `` | `innovations(h, name, prior)` | ... ``.
- `tests/conftest.py:212`: `drift = innovations(h, "drift", dist.Normal(0.0, drift_scale))`.
- `tests/example_models.py:51-56` and `110-115`: replace the `lambda: dist.Normal(0.0, drift_scale),` line with `dist.Normal(0.0, drift_scale),`.
- `tests/test_models.py:108-110`: `drift = innovations(h, "drift", dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0))`; lines 165, 198, 209, 233, 246, 260: drop the `lambda: ` prefix.
- `tests/test_reparam.py:95,130` and `tests/contrib/test_blackjax.py:63-65`: drop the `lambda: ` prefix.

- [ ] **Step 5: Migrate the notebooks**

`rewrite` for `/tmp/nb_rewrite.py`, applied to the six notebooks listed above:

```python
import re

_LAZY = re.compile(r"lambda:\s*(dist\.Normal\([^()]*\))")


def rewrite(src: str) -> str:
    return _LAZY.sub(r"\1", src)
```

Expected: exactly one changed cell per notebook (the model cell). Then `uv run ruff check --fix docs/examples && uv run ruff format docs/examples`.

- [ ] **Step 6: Run the gates and tests**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest tests/test_models.py tests/test_reparam.py tests/contrib/test_blackjax.py README.md -v`
Expected: PASS (ty over the notebooks catches any site the regex missed as `too-many-positional-arguments`/`missing-argument`; `invalid-argument-type` is still active in notebooks at this point).

- [ ] **Step 7: Commit**

```bash
git add numpyro_forecast/models.py README.md tests docs/examples
git commit -m "feat!: innovations takes a Distribution instance instead of a thunk"
```

---

### Task 4: `markov_series()` takes `transition` and an optional `advance`

**Files:**
- Modify: `numpyro_forecast/models.py:191-332` (`Transition` alias, new `Advance` alias, `markov_series`)
- Modify: `numpyro_forecast/__init__.py:51-60,70-108` (export `Advance`), `great-docs.yml:76-84` (add `models.Advance`), `tests/test_package.py:14-16,44-46,78-80`
- Modify: `numpyro_forecast/var.py:65-76` (docstring example)
- Modify: `tests/test_markov.py:23-26,91-92,110-113,176-180`, `tests/test_reparam.py:145-148`
- Modify (notebook, in place): `docs/examples/fresh_retail_stockout.ipynb` (cell 45, `slope_transition`)
- Test: `tests/test_markov.py`

**Interfaces:**
- Produces:

```python
type Transition[Carry] = Callable[[Carry, PyTree[Array] | None], dist.Distribution]
type Advance[Carry] = Callable[[Carry, Array, PyTree[Array] | None], Carry]


def markov_series[Carry](
    h: Horizon,
    name: str,
    init_carry: Carry,
    transition: Transition[Carry],
    xs: PyTree[Array] | None = None,
    *,
    advance: Advance[Carry] | None = None,
    plates: Sequence[tuple[str, int]] = (),
    reparam_config: Mapping[str, Reparam] | None = None,
) -> Array: ...
```

`transition(carry, x_t)` returns the per-step distribution; the wrapper samples `z_t` and the next carry is `advance(carry, z_t, x_t)`, or `z_t` itself when `advance` is `None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_markov.py` replace `_ar1_transition` with the distribution-only form and add a window-carry test:

```python
def _ar1_transition(carry: Array, _: Array | None) -> dist.Distribution:
    return dist.Normal(PHI * carry, DRIFT)


def test_advance_builds_a_window_carry() -> None:
    """A two-lag carry: the sample enters the window and the oldest row leaves."""

    def transition(window: Array, _: Array | None) -> dist.Distribution:
        return dist.Normal(0.5 * window[0] + 0.25 * window[1], DRIFT)

    def advance(window: Array, z_t: Array, _: Array | None) -> Array:
        return jnp.concatenate([z_t[None], window[:1]], axis=0)

    covariates = empty_covariates(8)
    h = Horizon.from_data(covariates, jnp.zeros((5, 1)))

    def body() -> None:
        markov_series(h, "z", jnp.zeros((2, 1)), transition, advance=advance)

    tr = trace(seed(body, random.PRNGKey(0))).get_trace()
    assert tr["z"]["value"].shape == (5, 1)
    assert tr["z_future"]["value"].shape == (3, 1)
```

Also add the failure-mode test for a structured carry without `advance`:

```python
def test_missing_advance_with_structured_carry_is_rejected() -> None:
    def transition(window: Array, _: Array | None) -> dist.Distribution:
        return dist.Normal(window[0], DRIFT)

    covariates = empty_covariates(5)
    h = Horizon.from_data(covariates, jnp.zeros((5, 1)))

    def body() -> None:
        markov_series(h, "z", jnp.zeros((2, 1)), transition)

    with pytest.raises(ValueError, match="pass advance="):
        trace(seed(body, random.PRNGKey(0))).get_trace()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_markov.py -v`
Expected: `test_advance_builds_a_window_carry` FAILS with `TypeError: ... unexpected keyword argument 'advance'`; the existing AR(1) tests FAIL with `TypeError: cannot unpack non-iterable Normal object` (the wrapper still expects a tuple); `test_missing_advance_with_structured_carry_is_rejected` FAILS because the current code raises jax's scan error (`carry input and carry output must have equal types`) instead of a `ValueError` naming `advance=`.

- [ ] **Step 3: Implement**

Replace the alias block and the scan body:

```python
type Transition[Carry] = Callable[[Carry, PyTree[Array] | None], dist.Distribution]
"""``(carry, x_t) -> dist_t``: the distribution of the next latent given the carry.
The wrapper owns the sample statement; see `Advance` for the carry update.

``Carry`` is the user's carry type (any PyTree), bound per `markov_series()`
call; ``x_t`` is one row of the ``xs`` PyTree (``None`` for autonomous dynamics)."""

type Advance[Carry] = Callable[[Carry, Array, PyTree[Array] | None], Carry]
"""``(carry, z_t, x_t) -> carry``: the next carry from the current carry, the
*sampled* latent ``z_t`` and the exogenous row. Omit it when the carry is the
latent itself (the AR(1) case); a lag window keeps the last ``p`` samples."""
```

In `markov_series` add the keyword-only `advance: Advance[Carry] | None = None` (before `plates`) and change the body:

```python
def body(carry: Carry, x_t: PyTree[Array] | None) -> tuple[Carry, Array]:
    dist_t = transition(carry, x_t)
    _validate_markov_step_dist(dist_t)
    ctx = (
        numpyro.handlers.reparam(config=dict(reparam_config)) if reparam_config else nullcontext()
    )
    with ctx, _plate_stack(plates):
        z = cast(Array, numpyro.sample(site_name, dist_t))
    if advance is not None:
        return advance(carry, z, x_t), z
    return _validate_markov_default_carry(carry, z), z
```

with the new validator next to `_validate_markov_step_dist` (mirrors `_validate_ssoe_carry`, so a mismatch names the fix instead of surfacing jax's raw scan error `carry input and carry output must have equal types`):

```python
def _validate_markov_default_carry[Carry](carry: Carry, z: Array) -> Carry:
    """Require the sampled latent to match the carry when ``advance`` is omitted."""
    leaves = jax.tree.leaves(carry)
    if len(leaves) != 1 or leaves[0].shape != z.shape or leaves[0].dtype != z.dtype:
        msg = (
            "markov_series without advance= uses the sampled latent as the next carry, so "
            f"init_carry must be a single array shaped like one draw ({z.dtype}{z.shape}); got "
            f"{jax.tree.structure(carry)} with leaves "
            f"{[f'{leaf.dtype}{leaf.shape}' for leaf in leaves]}. Pass advance=(carry, z_t, x_t) "
            "-> carry to build a structured carry (for example a lag window) from the draw."
        )
        raise ValueError(msg)
    return cast(Carry, z)
```

(`jnp.asarray` the leaves before reading `.shape`/`.dtype` if a Python scalar `init_carry` must stay legal; the existing tests only pass arrays.)

Docstring updates in `markov_series`: the `transition` entry becomes "Per-step ``(carry, x_t) -> dist_t`` callable returning the distribution of the next latent (see `Transition`); the wrapper owns the ``numpyro.sample`` statement."; add an `advance` entry: "Optional ``(carry, z_t, x_t) -> carry`` (see `Advance`) that builds the next carry from the sampled latent; ``None`` means the carry *is* the latent, ``carry_{t+1} = z_t``, so ``init_carry`` must be a single array shaped like one draw. A vector autoregression with ``p`` lags keeps a ``(p, obs)`` window here."; extend the `Raises` entry with "or if ``advance`` is omitted and ``init_carry`` is not a single array with the shape and dtype of a draw".

Update `var.py:65-76` docstring example:

```python
    def transition(window, _):
        return dist.MultivariateNormal(var_mean(phi, window), scale_tril=scale_tril)


    def advance(window, z_t, _):
        return jnp.concatenate([window[..., 1:, :], z_t[..., None, :]], axis=-2)


    z = markov_series(h, "z", jnp.zeros((n_lags, n_obs)), transition, advance=advance)
```

Exports: add `Advance` to the `from numpyro_forecast.models import (...)` block and `__all__` in `numpyro_forecast/__init__.py`, to `models.Advance` in `great-docs.yml` (after `models.Transition`), and to the three import/`__all__` lists in `tests/test_package.py`.

- [ ] **Step 4: Migrate the remaining callers**

- `tests/test_markov.py:91-92` (`trans`), `110-113` (`bad_trans`, keep whatever makes it "bad"), `176-180` (`driven`): return the distribution only, drop the `, lambda z: z`; return annotation `-> dist.Distribution`. Remove the now-unused `Callable` import if nothing else uses it.
- `tests/test_reparam.py:145-148`: `return dist.Normal(carry, 1.0).to_event(1)`.
- `docs/examples/fresh_retail_stockout.ipynb`, cell 45, literal replacement in `rewrite`:

```python
OLD = """        def slope_transition(
            carry: Array, _: Array | None
        ) -> tuple[dist.Distribution, Callable[[Array], Array]]:
            return dist.Normal(phi_trend * carry, tau_trend), lambda value: value
"""
NEW = """        def slope_transition(carry: Array, _: Array | None) -> dist.Distribution:
            return dist.Normal(phi_trend * carry, tau_trend)
"""


def rewrite(src: str) -> str:
    return src.replace(OLD, NEW)
```

Expected: exactly one changed cell. If `Callable` becomes unused in that notebook, `ruff check --fix` removes the import.

- [ ] **Step 5: Run the gates and tests**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest tests/test_markov.py tests/test_reparam.py tests/test_package.py tests/test_docs_reference.py tests/test_var.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add numpyro_forecast great-docs.yml tests docs/examples/fresh_retail_stockout.ipynb
git commit -m "feat!: markov_series takes transition -> dist_t and an optional advance"
```

---

### Task 5: `ssoe()` takes `mean` and `update`

**Files:**
- Modify: `numpyro_forecast/models.py:335-345,413-454,510-709` (aliases, validators' messages and docstrings, `ssoe`; the prose at 432, 528-529, 549, 595, 597-602, 618, 627-636, 640-660 names `step`/`carry_fn`)
- Modify: `numpyro_forecast/var.py:23,28-30,82-180` (`Callable` import becomes unused; `var_step` returns the pair)
- Modify: `numpyro_forecast/__init__.py`, `great-docs.yml:81` (`models.SSOEStep` becomes `models.SSOEMean` and `models.SSOEUpdate`), `tests/test_package.py`
- Modify: `README.md:124,127,129`
- Modify: `tests/conftest.py:253-254,276-278`, `tests/example_models.py:9,167-178` (`Callable` import becomes unused), `tests/test_ssoe.py` (26 `ssoe(` calls, every step definition, the validation stubs at 326-482 and 628-631, the `CarryFn`/`identity_step` imports at 12 and 16), `tests/test_var.py:5,77-112,133-143` (`Any` import becomes unused; four direct `var_step(...)` callers plus the model)
- Modify (notebooks, in place; code cells and the markdown cells that explain the protocol): `arma`, `censored_demand`, `croston`, `tsb`, `availability_tsb`, `exponential_smoothing_state_space`, `var`
- Test: `tests/test_ssoe.py`, `tests/test_var.py`

**Interfaces:**
- Produces:

```python
type SSOEMean[Carry] = Callable[[Carry, PyTree[Array] | None], Float[Array, " *batch obs"]]
type SSOEUpdate[Carry] = Callable[[Carry, Array, Array, PyTree[Array] | None], Carry]


def ssoe[Carry](
    h: Horizon,
    name: str,
    y: Array | None,
    init_carry: Carry,
    mean: SSOEMean[Carry],
    update: SSOEUpdate[Carry],
    noise_dist: dist.Distribution,
    xs: PyTree[Array] | None = None,
) -> SSOEResult: ...


def var_step(
    phi: Float[Array, " *#batch lags obs obs"],
    intercept: Float[Array, " *#batch obs"] | None = None,
) -> tuple[
    SSOEMean[Float[Array, " *batch lags obs"]], SSOEUpdate[Float[Array, " *batch lags obs"]]
]: ...
```

`mean(carry, x_t)` returns the one-step-ahead mean row `(*batch, obs)`; `update(carry, y_t, eps_t, x_t)` returns the next carry from the row's value and error (in-sample `eps_t = y_t - mu_t`; over the horizon `y_t = mu_t + eps_t` with the drawn `eps_t`). An update that needs the mean calls `mean(carry, x_t)` again: the same expression on the same operands is bit-identical and XLA computes it once. It must not reconstruct the mean as `y_t - eps_t`: in float32 that round trip differs from `mu_t` in about 0.6% of in-sample rows and 2.5% of horizon rows (probe of 100_000 rows), which would change `r.mu`, the log density and every stored NUTS output of the censored notebook. `var_step` returns `(mean, update)`; callers unpack: `mean, update = var_step(phi, intercept)`. The one-line return annotation is 101 characters, so it is written in the wrapped form above (no new alias).

- [ ] **Step 1: Rewrite the test scaffolding to the new protocol (failing first)**

`tests/conftest.py`: delete `CarryFn` (lines 253-254) and replace `identity_step`:

```python
def identity_mean(carry: Array, _: object) -> Array:
    """An ``ssoe`` mean that emits the carry unchanged."""
    return carry


def keep_carry(carry: Array, y_t: Array, eps_t: Array, _: object) -> Array:
    """An ``ssoe`` update that never changes the carry."""
    return carry
```

`tests/test_ssoe.py`: the ARMA body becomes

```python
def _arma_ssoe_body(h: Horizon, covariates: Array) -> SSOEResult:
    y = covariates[..., : h.t_obs, :]
    mu, phi, theta, sigma = _arma_params()

    def mean(carry: tuple[Array, Array], _: object) -> Array:
        y_prev, eps_prev = carry
        return mu + phi * y_prev + theta * eps_prev

    def update(
        carry: tuple[Array, Array], y_t: Array, eps_t: Array, _: object
    ) -> tuple[Array, Array]:
        return y_t, eps_t

    r = ssoe(h, "eps", y, (mu[None], jnp.zeros((1,))), mean, update, dist.Normal(0.0, sigma))
    ...
```

the ETS body becomes

```python
def mean(carry: EtsCarry, _: object) -> Array:
    level, trend, season = carry
    return jnp.asarray(level + PHI_ETS * trend + season[0])[None]


def update(carry: EtsCarry, y_t: Array, eps_t: Array, _: object) -> EtsCarry:
    level, trend, season = carry
    e = eps_t[0]
    new_season = season[0] + 0.2 * e
    return (
        level + PHI_ETS * trend + 0.5 * e,
        PHI_ETS * trend + 0.1 * e,
        jnp.concatenate([season[1:], new_season[None]]),
    )


init: EtsCarry = (0.0, 0.0, jnp.zeros((4,)))
r = ssoe(h, "eps", y, init, mean, update, dist.Normal(0.0, sigma))
```

and every other `ssoe(h, name, y, init, step, noise, ...)` call passes `mean, update` in place of `step`; each inline `step` returning `(value, lambda y_t, eps_t: ...)` splits into the two functions with the lambda body moved into `update(carry, y_t, eps_t, x_t)`; a closed-over `carry`, `gate_t` or `x_t` becomes the corresponding parameter, and a closed-over mean (`pred` in the censored stub at 368-380) is obtained by calling `mean(carry, x_t)` inside `update`, never as `y_t - eps_t`. The validation stubs (lines 326-482, 628-631) keep their intent: a `mean` that returns `carry[0]` (scalar-mean rejection), an `update` that returns `(y_t, eps_t)` for a non-tuple carry (tree-structure rejection), `jnp.zeros((2,))` (shape rejection), `0.0` from `mean` (float-mean rejection), `carry.astype(jnp.float32)` (dtype rejection), and a `mean` that calls `numpyro.sample` (sample-inside rejection). Update the `match=` strings to the new messages in Step 3 (`"mean must return"`, `"update must return"`, `"mean and update must not call"`). Remove the `CarryFn` and `identity_step` names from the `conftest` import at lines 11-21 and import `identity_mean, keep_carry` instead.

The censored stub becomes:

```python
def mean(carry: tuple[Array, Array], x_t: tuple[Array, Array, Array] | None) -> Array:
    assert x_t is not None
    seasonal_t, _, _ = x_t
    lag_1, lag_2 = carry
    return 0.5 * lag_1 + 0.2 * lag_2 + seasonal_t


def update(
    carry: tuple[Array, Array], y_t: Array, _: Array, x_t: tuple[Array, Array, Array] | None
) -> tuple[Array, Array]:
    assert x_t is not None
    _, available_t, censored_t = x_t
    lag_1, _ = carry
    pred = mean(carry, x_t)  # same ops, same operands: bit-identical to the filter's mu_t
    on_shelf = jnp.where(censored_t == 1, jnp.maximum(y_t, pred), y_t)
    return jnp.clip(jnp.where(available_t == 1, on_shelf, pred), 0.0), lag_1
```

`tests/example_models.py:167-178` (Croston level channel; the function binds `result = ssoe(...)` and returns `result, noise`, keep that and the inline `xs=pad_future(gate, h.future)`; the `Callable` import at line 9 becomes unused and is deleted):

```python
def mean(level: Array, _: Array | None) -> Array:
    return level


def update(level: Array, y_t: Array, _: Array, gate_t: Array | None) -> Array:
    assert gate_t is not None  # xs is always passed here
    return jnp.where(gate_t, smoothing * y_t + (1.0 - smoothing) * level, level)


result = ssoe(
    h,
    name,
    values,
    init[None],
    mean,
    update,
    dist.Normal(0.0, noise),
    xs=pad_future(gate, h.future),
)
```

`tests/test_var.py`: four tests call the returned step directly (`var_step(phi, c)(lags, None)` at 79, `step = var_step(phi)` at 91-93, `base = var_step(phi)` at 99-103, `var_step(_phi())(jnp.zeros((P + 1, K)), None)` at 112) and the model at 141-142 passes it to `ssoe`; all become tuple unpacking (the `Any` import at line 5 becomes unused and is deleted):

```python
def test_var_step_mean_and_window_shift() -> None:
    phi, lags, c = _phi(), _lags(), jnp.array([0.1, 0.2])
    mean, update = var_step(phi, c)
    mu = mean(lags, None)
    assert jnp.allclose(mu, var_mean(phi, lags, c))
    y_t = jnp.array([1.0, 2.0])
    new = update(lags, y_t, y_t - mu, None)
    assert new.shape == lags.shape
    assert new.dtype == lags.dtype
    assert jnp.array_equal(new[:-1], lags[1:])
    assert jnp.array_equal(new[-1], y_t)


def test_var_step_ignores_exogenous_input() -> None:
    phi, lags = _phi(), _lags()
    mean, _ = var_step(phi)
    assert jnp.array_equal(mean(lags, None), mean(lags, jnp.ones((4,))))


def test_var_step_varx_by_wrapping() -> None:
    phi, lags, beta = _phi(), _lags(), jnp.array([[1.0, 0.0], [0.0, 2.0]])
    mean, _ = var_step(phi)

    def mean_x(carry: Array, x_t: Array) -> Array:
        return mean(carry, x_t) + beta @ x_t

    x_t = jnp.array([0.5, -1.0])
    assert jnp.allclose(mean_x(lags, x_t), var_mean(phi, lags) + beta @ x_t)


def test_var_step_rejects_wrong_lag_count_with_guidance() -> None:
    mean, _ = var_step(_phi())
    with pytest.raises(ValueError, match=r"lags=2.*init_carry=y\[\.\.\., :2, :\]"):
        mean(jnp.zeros((P + 1, K)), None)
```

(`x_t: Array`, not `Array | None`: ty rejects `beta @ x_t` on an optional.) The model at 141-142 becomes `mean, update = var_step(phi, intercept)` and `r = ssoe(h, "eps", y, y_init, mean, update, noise)`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ssoe.py tests/test_var.py -x -q`
Expected: FAIL at the first `ssoe(...)` call. The failure is jaxtyping's `TypeCheckError` from the beartype wrapper binding eight positional arguments to the current seven-parameter signature (not a plain `TypeError` message about argument counts), and the four `var_step` tests fail with `TypeError: cannot unpack non-iterable function object`.

- [ ] **Step 3: Implement `ssoe`**

Replace the `SSOEStep` alias with:

```python
type SSOEMean[Carry] = Callable[[Carry, PyTree[Array] | None], Float[Array, " *batch obs"]]
"""``(carry, x_t) -> mu_t``: the one-step-ahead mean of the current row, shape
``(*batch, obs)`` (a scalar state emits ``mu[None]``). `ssoe()` owns the error
site: ``mean`` must not call ``numpyro.sample`` (that is `markov_series()`).

``Carry`` is the user's carry type (any PyTree), bound per `ssoe()` call;
``x_t`` is one row of the ``xs`` PyTree (``None`` when ``xs`` is ``None``)."""

type SSOEUpdate[Carry] = Callable[[Carry, Array, Array, PyTree[Array] | None], Carry]
"""``(carry, y_t, eps_t, x_t) -> carry``: the next carry from the row's value and
error. In-sample ``eps_t = y_t - mu_t``; over the horizon ``eps_t`` is the drawn
error and ``y_t = mu_t + eps_t``. An update that needs the mean calls
``mean(carry, x_t)`` again (bit-identical, computed once by XLA) rather than
reconstructing it as ``y_t - eps_t``, which can differ by an ulp. Must preserve
the carry's tree structure, shapes and dtypes."""
```

Signature as in Interfaces. Scan bodies:

```python
def filter_body(carry: Carry, inputs: tuple[Array, PyTree[Array] | None]) -> tuple[Carry, Array]:
    y_t, x_t = inputs
    mu_t = jnp.asarray(mean(carry, x_t))
    _validate_ssoe_mean(mu_t, y_t)
    new_carry = _validate_ssoe_carry(carry, update(carry, y_t, y_t - mu_t, x_t))
    return new_carry, mu_t
```

```python
    def forecast_body(
        carry: Carry, inputs: tuple[Array, PyTree[Array] | None]
    ) -> tuple[Carry, tuple[Array, Array]]:
        eps_t, x_t = inputs
        mu_t = mean(carry, x_t)
        y_t = mu_t + eps_t
        return update(carry, y_t, eps_t, x_t), (mu_t, y_t)
```

Messages: in `_validate_ssoe_mean` replace "step must return a per-step mean" with "mean must return a per-step mean" and "step must return a floating per-step mean" with "mean must return a floating per-step mean"; in `_validate_ssoe_carry` replace both "carry_fn must return"/"carry_fn changed" with "update must return"/"update changed"; the trace guard becomes "mean and update must not call numpyro.sample or numpyro.deterministic (found sites: ...); a sampled transition is markov_series.".

Docstring: replace the `step` parameter entry with

```text
    mean
        ``(carry, x_t) -> mu_t`` (see `SSOEMean`): the mean for the current row
        (shape ``(*batch, obs)``, so a scalar state emits ``mu[None]``).
    update
        ``(carry, y_t, eps_t, x_t) -> carry`` (see `SSOEUpdate`): the next carry
        from the row's value and error. Over the horizon it receives the drawn
        ``eps_t`` (not a recomputed ``y_t - mu_t``, which can differ by an ulp);
        when the update needs the mean, call ``mean(carry, x_t)`` inside it.
```

rewrite the summary sentence "runs ``step`` in a raw ``jax.lax.scan``" as "runs ``mean`` and ``update`` in a raw ``jax.lax.scan``" and, in the same paragraph, "fed back through ``carry_fn``" (528-529) as "fed back through ``update``"; the "Frozen gates" paragraph's "``carry_fn`` is the identity" (549) becomes "``update`` returns the carry unchanged"; the `init_carry` entry's "through ``carry_fn``" (595) becomes "through ``update``"; the `xs` entry's "handed to ``step`` row by row" (618) becomes "handed to ``mean`` and ``update`` row by row"; the Examples intro "ARMA(1,1) with the lambda form of ``carry_fn``" (640) becomes "ARMA(1,1)"; `_validate_ssoe_carry`'s one-line docstring (432) becomes "Require ``update`` to preserve the carry's tree structure, shapes and dtypes."; the "Shapes" paragraph's "a tuple carry with scalar leaves reads ``eps_t[0]`` (the ETS idiom)" stays; and the two Examples become:

```text
    >>> def mean(carry, _):
    ...     y_prev, eps_prev = carry
    ...     return mu + phi * y_prev + theta * eps_prev
    >>> def update(carry, y_t, eps_t, _):
    ...     return y_t, eps_t
    >>> r = ssoe(
    ...     h, "eps", y, (mu[None], jnp.zeros((1,))), mean, update, dist.Normal(0.0, sigma)
    ... )
    >>> numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    >>> if h.future > 0:
    ...     numpyro.deterministic("forecast", r.y_future)

    A gated level (Croston, TSB) with the gate frozen over the horizon:

    >>> def mean(level, _):
    ...     return level
    >>> def update(level, y_t, _, gate_t):
    ...     return jnp.where(gate_t, alpha * y_t + (1 - alpha) * level, level)
    >>> gate_full = pad_future(gate, h.future)
    >>> r = ssoe(h, "eps", y, init[None], mean, update, dist.Normal(0.0, noise), xs=gate_full)
```

Also update the `Raises` entry ("if ``step`` returns a mean without ..." becomes "if ``mean`` returns a value without the observation axis or ``update`` a carry with a different tree structure, shape or dtype; if ``mean`` or ``update`` calls ``numpyro.sample``").

- [ ] **Step 4: Implement `var_step`**

```python
def var_step(
    phi: Float[Array, " *#batch lags obs obs"],
    intercept: Float[Array, " *#batch obs"] | None = None,
) -> tuple[
    SSOEMean[Float[Array, " *batch lags obs"]], SSOEUpdate[Float[Array, " *batch lags obs"]]
]:
    r"""Build the `~~numpyro_forecast.models.ssoe()` mean and update of a VAR."""
    n_lags, n_obs = phi.shape[-3], phi.shape[-1]

    def mean(
        carry: Float[Array, " *batch lags obs"], x_t: PyTree[Array] | None
    ) -> Float[Array, " *batch obs"]:
        if carry.shape[-2] != n_lags:
            msg = (
                f"var_step expects a carry of shape (*batch, lags={n_lags}, obs={n_obs}) holding "
                f"the last {n_lags} rows in natural time order; got {carry.shape}. Seed ssoe with "
                f"init_carry=y[..., :{n_lags}, :] and drive it with y[..., {n_lags}:, :]."
            )
            raise ValueError(msg)
        return var_mean(phi, carry, intercept)

    def update(
        carry: Float[Array, " *batch lags obs"],
        y_t: Float[Array, " *batch obs"],
        eps_t: Float[Array, " *batch obs"],
        x_t: PyTree[Array] | None,
    ) -> Float[Array, " *batch lags obs"]:
        return jnp.concatenate([carry[..., 1:, :], y_t[..., None, :]], axis=-2)

    return mean, update
```

(The one-line return annotation is 101 characters; the wrapped form above is what `ruff format` produces, and no module-level alias is introduced because a public `type Window` would need a docstring and a `reference:` entry. The docstring keeps today's full text; only the summary line and the sections quoted below change.) The `Callable` import at `var.py:23` is now unused: delete it.

Docstring: "Build the `ssoe()` mean and update of a VAR from its coefficients." with the VARX example

```python
    mean, update = var_step(phi, intercept)


    def mean_x(carry, x_t):
        return mean(carry, x_t) + beta @ x_t
```

the Returns section "A ``(mean, update)`` pair for `~~numpyro_forecast.models.ssoe()`: ``mean(carry, x_t)`` is `var_mean()` of the window and ``update(carry, y_t, eps_t, x_t)`` drops the oldest row and appends ``y_t``.", and the `var_model` example calling `mean, update = var_step(phi, intercept)` then `r = ssoe(h, "eps", y, y_init, mean, update, noise)`. Change the import at `var.py:29` to `from numpyro_forecast.models import SSOEMean, SSOEUpdate`.

Exports: replace `SSOEStep` with `SSOEMean` and `SSOEUpdate` in `numpyro_forecast/__init__.py` (import block and `__all__`, keep `__all__` sorted), in `great-docs.yml` (`models.SSOEMean`, `models.SSOEUpdate` replace `models.SSOEStep`) and in `tests/test_package.py` (three lists).

README: line 124 becomes `` | `ssoe(h, name, y, init_carry, mean, update, noise_dist)` | ... ``; line 129 keeps "the step factory (`var_step`)" (it still builds the VAR step, now as a `(mean, update)` pair); no other README text names `step`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_ssoe.py tests/test_var.py tests/test_package.py tests/test_docs_reference.py tests/test_docstring_markup.py README.md -v`
Expected: PASS.

- [ ] **Step 6: Migrate the seven notebooks (code cells)**

Literal replacements per notebook. Every `OLD` below is the verbatim committed cell text (checked with `json.load`; note the trailing comment on `arma`'s `init_carry` line, the two wordings of the gated-level comment, and the comment block inside `censored_demand`'s `carry_fn`); each `rewrite` chains the listed `str.replace` calls and is applied with `expect=1`, so a mismatch stops the script instead of silently skipping a notebook.

`arma.ipynb` (code cell 13):

```python
OLD = """    def step(carry, _):
        y_prev, error_prev = carry
        pred = mu + phi * y_prev + theta * error_prev
        return pred, lambda y_t, eps_t: (y_t, eps_t)

    init_carry = (mu[None], jnp.zeros((1,)))  # y_{-1} = mu and eps_{-1} = 0 seed the recursion
    r = ssoe(h, "eps", y, init_carry, step, dist.Normal(loc=0, scale=sigma))
"""
NEW = """    def mean(carry, _):
        y_prev, error_prev = carry
        return mu + phi * y_prev + theta * error_prev

    def update(carry, y_t, eps_t, _):
        return y_t, eps_t

    init_carry = (mu[None], jnp.zeros((1,)))  # y_{-1} = mu and eps_{-1} = 0 seed the recursion
    r = ssoe(h, "eps", y, init_carry, mean, update, dist.Normal(loc=0, scale=sigma))
"""
```

`croston.ipynb` (code cell 12), `tsb.ipynb` (code cell 12), `availability_tsb.ipynb` (code cell 15). The step text differs only in the comment: croston says `update only at events.`, tsb and availability_tsb say `update only where gated.`; `availability_tsb` passes `init` rather than `init[None]` to `ssoe`, which is untouched by these replacements:

```python
def gated_rewrite(src: str, where: str) -> str:
    old_step = f"""    def step(level, gate_t):
        # Emit the pre-update level (the one-step-ahead mean); update only {where}.
        return level, lambda y_t, _: jnp.where(
            gate_t, smoothing * y_t + (1 - smoothing) * level, level
        )
"""
    new_step = f"""    def mean(level, _):
        # Emit the pre-update level (the one-step-ahead mean).
        return level

    def update(level, y_t, _, gate_t):
        # Update only {where}; the gate is frozen over the horizon.
        return jnp.where(gate_t, smoothing * y_t + (1 - smoothing) * level, level)
"""
    old_call = "        step,\n        dist.Normal(loc=0, scale=noise),\n"
    new_call = "        mean,\n        update,\n        dist.Normal(loc=0, scale=noise),\n"
    return src.replace(old_step, new_step).replace(old_call, new_call)


# croston: partial(gated_rewrite, where="at events"); tsb and availability_tsb: where="where gated"
```

`exponential_smoothing_state_space.ipynb` (code cell 11):

```python
OLD = """    def step(carry, _):
        level, trend, seasonality = carry
        mu = level + phi * trend + seasonality[0]
        # Rows carry the observation axis: emit a (1,) mean, read the scalar error back.
        return mu[None], lambda y_t, eps_t: advance(carry, eps_t[0])

    init_state = (level_init, trend_init, seasonality_init)
    r = ssoe(h, "eps", y, init_state, step, dist.Normal(0, noise))
"""
NEW = """    def mean(carry, _):
        level, trend, seasonality = carry
        # Rows carry the observation axis: emit a (1,) mean.
        return (level + phi * trend + seasonality[0])[None]

    def update(carry, y_t, eps_t, _):
        # Read the scalar error back from the (1,) row.
        return advance(carry, eps_t[0])

    init_state = (level_init, trend_init, seasonality_init)
    r = ssoe(h, "eps", y, init_state, mean, update, dist.Normal(0, noise))
"""
```

`censored_demand.ipynb` (code cell 10). `update` obtains the prediction by calling `mean` again (bit-identical to the filter's `mu_t`; see Interfaces), never as `y_t - eps_t`:

```python
OLD = """    def step(carry, x_t):
        seasonal_t, available_t, censored_t = x_t
        lag_1, lag_2 = carry
        pred = mu + phi_1 * lag_1 + phi_2 * lag_2 + seasonal_t

        def carry_fn(y_t, _):
            # The filtered lag: pass clean observations through, floor capped days at the
            # prediction, and substitute the prediction on stockout days.
            on_shelf = jnp.where(censored_t == 1, jnp.maximum(y_t, pred), y_t)
            y_filtered = jnp.where(available_t == 1, on_shelf, pred)
            return jnp.clip(y_filtered, min=0.0), lag_1

        return pred, carry_fn
"""
NEW = """    def mean(carry, x_t):
        seasonal_t, _, _ = x_t
        lag_1, lag_2 = carry
        return mu + phi_1 * lag_1 + phi_2 * lag_2 + seasonal_t

    def update(carry, y_t, eps_t, x_t):
        _, available_t, censored_t = x_t
        lag_1, _ = carry
        # The filtered lag: pass clean observations through, floor capped days at the
        # prediction, and substitute the prediction on stockout days. Calling mean again
        # reproduces the filter's prediction exactly (same expression, same inputs).
        pred = mean(carry, x_t)
        on_shelf = jnp.where(censored_t == 1, jnp.maximum(y_t, pred), y_t)
        y_filtered = jnp.where(available_t == 1, on_shelf, pred)
        return jnp.clip(y_filtered, min=0.0), lag_1
"""
OLD_CALL = 'r = ssoe(h, "eps", y, init_carry, step, dist.Normal(loc=0, scale=sigma), xs=xs)'
NEW_CALL = (
    'r = ssoe(h, "eps", y, init_carry, mean, update, dist.Normal(loc=0, scale=sigma), xs=xs)'
)
```

`var.ipynb` (code cell 10): `r = ssoe(h, "eps", y, y_init, var_step(phi, intercept), noise)` becomes two lines, `mean, update = var_step(phi, intercept)` and `r = ssoe(h, "eps", y, y_init, mean, update, noise)` (same indentation as the original line).

Then `uv run ruff check --fix docs/examples && uv run ruff format docs/examples && uv run ty check`.

- [ ] **Step 7: Migrate the seven notebooks (markdown cells)**

The prose that explains the protocol must match the code; apply these sentence replacements with `cell_type="markdown"` and `expect=1` per notebook (the LaTeX fragments contain backslashes, so write the literals as raw strings). Verbatim old fragments were extracted from the committed cells:

- `arma.ipynb` markdown cell 12: "a `step` function, and the innovation distribution; `step(carry, x_t)` returns the one-step-ahead mean and a `carry_fn(y_t, eps_t)` that builds the next carry, here simply the day's value and error." becomes "a `mean` function, an `update` function, and the innovation distribution; `mean(carry, x_t)` returns the one-step-ahead mean and `update(carry, y_t, eps_t, x_t)` builds the next carry, here simply the day's value and error."
- `censored_demand.ipynb` markdown cell 9, five fragments: "an initial carry, a `step` function, and the innovation distribution." becomes "an initial carry, a `mean` function, an `update` function, and the innovation distribution."; r"`step(carry, x_t)` returns the one-step-ahead mean $\hat{y}_t$ and a `carry_fn(y_t, eps_t)` that builds the next carry from the day's value and its error; closing `carry_fn` over the prediction is what lets the lag filter above floor capped days at $\hat{y}_t$." becomes r"`mean(carry, x_t)` returns the one-step-ahead mean $\hat{y}_t$ and `update(carry, y_t, eps_t, x_t)` builds the next carry from the day's value and its error; calling `mean` again inside `update` (the same expression on the same inputs, so exactly the filter's prediction) is what lets the lag filter above floor capped days at $\hat{y}_t$."; "runs `step` over the observed history" becomes "runs `mean` over the observed history"; the two occurrences of "through `carry_fn`" become "through `update`"; "the same `carry_fn` serves both scans" becomes "the same `update` serves both scans".
- `croston.ipynb` markdown cell 11, two fragments: "a `step` function returning the one-step-ahead mean and the carry update, and the innovation distribution" becomes "a `mean` function returning the one-step-ahead mean, an `update` function returning the next carry, and the innovation distribution"; "hands `ssoe` a `step` that emits the *pre-update* level (the one-step-ahead mean) and a `carry_fn` that applies the gated update above." becomes "hands `ssoe` a `mean` that emits the *pre-update* level (the one-step-ahead mean) and an `update` that applies the gated update above."
- `exponential_smoothing_state_space.ipynb` markdown cell 10, three fragments: "the initial state, a `step` function, and the innovation distribution, and it owns the two scans" becomes "the initial state, a `mean` function, an `update` function, and the innovation distribution, and it owns the two scans"; r"at each step `step(carry, x_t)` returns the one-step-ahead mean $\mu_t$ and a `carry_fn(y_t, eps_t)` that advances the state" becomes r"at each step `mean(carry, x_t)` returns the one-step-ahead mean $\mu_t$ and `update(carry, y_t, eps_t, x_t)` advances the state"; "feeding those innovations back through `carry_fn`" becomes "feeding those innovations back through `update`".
- `tsb.ipynb` markdown cell 11: "building block a `step` that emits the *pre-update* level (the one-step-ahead mean) and a `carry_fn` that applies the gated update;" becomes "building block a `mean` that emits the *pre-update* level (the one-step-ahead mean) and an `update` that applies the gated update;".
- `var.ipynb` markdown cell 0: "turns sampled coefficients into a step for the" becomes "turns sampled coefficients into the `mean` and `update` functions for the"; markdown cell 8: "`var_step(phi, intercept)` builds the step function from the sampled coefficients:" becomes "`var_step(phi, intercept)` builds the `mean` and `update` functions from the sampled coefficients:" (two cells, so `expect=2` for `var.ipynb`).
- `availability_tsb.ipynb` has no markdown mention of the protocol (checked by grepping its markdown cells for `step`/`carry_fn`); `fresh_retail_stockout.ipynb`'s markdown does not describe the `markov_series` transition protocol either.

After the markdown pass, grep every notebook's markdown and code cells for `carry_fn`, `` `step` `` and `step(carry` and expect zero hits outside prose that talks about time steps.

- [ ] **Step 8: Smoke the rewritten notebooks**

For each of the eight notebooks touched in Tasks 4 and 5 (`arma`, `censored_demand`, `croston`, `tsb`, `availability_tsb`, `exponential_smoothing_state_space`, `var`, `fresh_retail_stockout`), run `/tmp/nb_smoke.py` with the model cell index and a smoke cell built from that notebook's names (see the procedure section).
Expected: "smoke OK" for all eight; each trace contains `eps_future` (or `slope_future` for `fresh_retail_stockout`) and a `forecast` site with the notebook's horizon at axis `-2`.

- [ ] **Step 9: Commit**

```bash
git add numpyro_forecast great-docs.yml README.md tests docs/examples
git commit -m "feat!: ssoe takes mean and update functions; var_step returns the pair"
```

---

### Task 6: Remove the type-narrowing wrappers from the notebooks

**Files:**
- Modify: `pyproject.toml` (new `[[tool.ty.overrides]]` block after `[tool.ty.environment]`)
- Modify (notebooks, in place): `arma`, `availability_tsb`, `censored_demand`, `croston`, `dynestyx_integration`, `fresh_retail_stockout`, `hierarchical_forecasting_1`, `hierarchical_forecasting_2`, `tsb`, `var`
- Modify: `AGENTS.md` (document the override and the `.py` convention)

**Interfaces:**
- Produces: notebooks free of `cast("Array", ...)` and `jnp.asarray(numpyro.sample(...))`; `.py` files unchanged (they keep `jnp.asarray(...)` narrowing until Task 8's upstream fix lands).

- [ ] **Step 1: Add the override**

```toml
[tool.ty.environment]
python-version = "3.12"

# numpyro 0.22 annotates `numpyro.sample`/`numpyro.deterministic` as returning
# `jax.typing.ArrayLike` (a union with Python scalars), so every indexed, attribute-accessed
# or Array-typed use of a sampled value is a diagnostic. The package and tests narrow with
# `jnp.asarray(...)`; the example notebooks are prose-first, so the four rules that union
# triggers are ignored there instead (nothing else is). Remove once numpyro narrows the
# primitives' return types (tracked in docs/dev/issue_116_review_plan.md, Task 8).
[[tool.ty.overrides]]
include = ["docs/examples/**"]

[tool.ty.overrides.rules]
not-subscriptable = "ignore"
invalid-argument-type = "ignore"
invalid-return-type = "ignore"
unresolved-attribute = "ignore"
```

- [ ] **Step 2: Strip the wrappers**

`rewrite` for `/tmp/nb_rewrite.py`, applied to the ten notebooks with `expect=None` (measured changed-cell counts on the committed files: 1 each, except `dynestyx_integration` 4 and `fresh_retail_stockout` 2; ty and the counter in Step 3 are the acceptance check):

```python
import re

_WRAPPERS = (
    re.compile(r'cast\(\s*"?Array"?\s*,\s*'),  # cast("Array", <expr>)
    re.compile(r"jnp\.asarray\(\s*(?=numpyro\.sample\()"),  # jnp.asarray(numpyro.sample(...))
)
_COMMENTS = (
    "# jnp.asarray only narrows numpyro's union return type for the type checker.\n",
    "# The sum of two raw sample results needs a cast for the type checker.\n",
    "# asarray narrows numpyro's union return type for the type checker.\n",
)


def _unwrap(src: str, opener: re.Pattern[str]) -> str:
    while (m := opener.search(src)) is not None:
        start = m.start()
        depth, k = 0, src.index("(", start)
        while True:  # find the wrapper's closing paren
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        src = src[:start] + src[m.end() : k].strip() + src[k + 1 :]
    return src


def rewrite(src: str) -> str:
    for comment in _COMMENTS:
        src = "\n".join(line for line in src.split("\n") if line.strip() != comment.strip())
    for opener in _WRAPPERS:
        src = _unwrap(src, opener)
    return src
```

Leave `cast("xr.DataArray", ...)` and `cast("float", ...)` alone (the regex only matches `Array`). Then `uv run ruff check --fix docs/examples && uv run ruff format docs/examples` (removes `from typing import cast` where unused, rewraps the shortened lines). The override also makes the three existing `# ty: ignore[invalid-argument-type]` directives in `fresh_retail_stockout.ipynb` (cell 45: the two `assert isinstance(..., Float[Array, ...])` lines; cell 88: `contributions.to_dataset(name="contribution")`) redundant; ty reports them as `unused-ignore-comment` warnings, so delete those three trailing comments in the same `rewrite` (`src.replace("  # ty: ignore[invalid-argument-type]", "")` is exact: no other notebook carries a `ty: ignore`).

- [ ] **Step 3: Verify**

Run: `uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. Then confirm the override is scoped: temporarily add `x: str = 1` to a notebook cell, run `uv run ty check`, expect `invalid-assignment` reported for that notebook, revert.

Run: `python3 -c "import json,glob,re; print(sum(len(re.findall(r'cast\(\s*\"?Array', ''.join(''.join(c['source']) for c in json.load(open(f))['cells'] if c['cell_type']=='code'))) for f in glob.glob('docs/examples/*.ipynb')))"`
Expected: `0`. Likewise for `jnp.asarray(numpyro.sample`.

- [ ] **Step 4: Document**

In `AGENTS.md`, "Hard requirements", add: "**Sampled values and `ty`:** numpyro annotates `numpyro.sample` as returning `ArrayLike`. In `numpyro_forecast/`, `tests/` and `scripts/` narrow with `jnp.asarray(numpyro.sample(...))` when the value is indexed, attribute-accessed or passed to an `Array`-typed parameter (never `typing.cast`). In the example notebooks write the plain `numpyro.sample(...)`: `pyproject.toml` ignores `not-subscriptable`, `invalid-argument-type`, `invalid-return-type` and `unresolved-attribute` under `docs/examples/**` and nothing else. Those four are therefore blind in notebooks (a wrong argument type, a misspelled attribute or a stale return annotation is not reported there), which is why a notebook's model cell must be smoke-executed after any package API change; every other rule still applies, so keep the notebooks clean otherwise."

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml docs/examples AGENTS.md
git commit -m "chore: drop sample-site type narrowing from notebooks via a scoped ty override"
```

---

### Task 7: Module docstring, README prose and durations

**Files:**
- Modify: `numpyro_forecast/models.py:1-11` (module docstring mentions the protocols)
- Modify: `README.md:117-131` (prose around the table)
- Modify: `.test_durations` (optional refresh)

- [ ] **Step 1: Module docstring**

Append to the `models.py` module docstring: "The recursive blocks take plain per-step functions: `markov_series()` a `Transition` (and an optional `Advance`), `ssoe()` an `SSOEMean` and an `SSOEUpdate`; none of them returns a closure, and the wrapper owns every sample site."

- [ ] **Step 2: README prose**

In the paragraph after the table (line 127) nothing names `step`; in line 131 replace "`ssoe` is an iid error plate plus a deterministic scan that consumes those errors" with "`ssoe` is an iid error plate plus a deterministic scan driven by your `mean` and `update` functions". Confirm the doctest still runs: `uv run pytest README.md -v`.

- [ ] **Step 3: Full suite and durations**

Run: `make tests`
Expected: PASS. If the new tests are notably slow or the CI groups drift, run `make store-durations` and commit `.test_durations`.

- [ ] **Step 4: Commit**

```bash
git add numpyro_forecast/models.py README.md .test_durations
git commit -m "docs: describe the two-function step protocols"
```

---

### Task 8: Upstream numpyro track (separate repository, separate PR)

**Files (pyro-ppl/numpyro):**
- Modify: `numpyro/primitives.py` (`sample`, `deterministic` annotations), `numpyro/distributions/distribution.py:115-119` (`DistributionMeta.__call__` return type)
- Test: `test/test_typing.py` (new; a `reveal_type`-style assertion via `typing.assert_type` under the checker numpyro runs in CI)

**Interfaces:**
- Produces, once released: `numpyro.sample("x", dist.Normal(...))` typed `Array`; `numpyro.deterministic("d", value)` typed as its argument.

- [ ] **Step 1: Open the issue**

Title: "Narrow the return types of `numpyro.sample` and `numpyro.deterministic`". Body: PR #2206 made concrete distributions return `Array` from `sample`, but the primitive still returns `ArrayLike`, and `DistributionMeta.__call__ -> Any` hides the concrete class, so typed consumers must wrap every site (link this plan's evidence: the four ty rules and the counts). Propose the shape below and note the deliberate exception (base and wrapper classes keep `ArrayLike`, per the #2206 review).

- [ ] **Step 2: PR shape**

```python
# numpyro/distributions/distribution.py
_D = TypeVar("_D", bound="Distribution")


class DistributionMeta(type):
    def __call__(cls: type[_D], *args: Any, **kwargs: Any) -> _D: ...
```

```python
# numpyro/primitives.py
class _SamplesArray(Protocol):
    def sample(self, key: Optional[jax.Array], sample_shape: tuple[int, ...] = ()) -> Array: ...


@overload
def sample(
    name: str,
    fn: _SamplesArray,
    obs: Optional[Array] = None,
    rng_key: Optional[ArrayLike] = None,
    sample_shape: tuple[int, ...] = (),
    infer: Optional[dict] = None,
    obs_mask: Optional[ArrayLike] = None,
) -> Array: ...
@overload
def sample(
    name: str,
    fn: Distribution,
    obs: Optional[ArrayLike] = None,
    rng_key: Optional[ArrayLike] = None,
    sample_shape: tuple[int, ...] = (),
    infer: Optional[dict] = None,
    obs_mask: Optional[ArrayLike] = None,
) -> ArrayLike: ...


_T = TypeVar("_T", bound=ArrayLike)


def deterministic(name: str, value: _T) -> _T: ...
```

The first overload matches every distribution whose `sample` returns `Array` (all concrete families after #2206); `Distribution`, `ExpandedDistribution`, `Independent`, `MaskedDistribution`, `TransformedDistribution` and `Delta` keep `ArrayLike` and fall through to the second, so numpy-valued handler use cases stay typed as today. `.expand(...)`/`.to_event(...)` return the wrappers and therefore still yield `ArrayLike`; that is the honest residual and is stated in the PR.

- [ ] **Step 3: Follow-up in this repository (after the numpyro release)**

Bump `numpyro>=<release>` in `pyproject.toml`, delete the `[[tool.ty.overrides]]` block from Task 6, delete every `jnp.asarray(numpyro.sample(...))` whose only purpose is narrowing (`tests/test_ssoe.py:62-65,128,366,628,689-690`, `tests/test_var.py:135-140`, `tests/example_models.py:163-165`, `numpyro_forecast/var.py:136-143` docstring, `README.md` if any) and the three `cast(Array, numpyro.sample(...))` in `numpyro_forecast/models.py` (`_sample_time_block`, the `markov_series` body, the `ssoe` future-error site), then re-run `uv run ty check`; sites that draw from `.expand(...).to_event(...)` wrappers (for example `tests/test_ssoe.py:85`, and the `var.py` docstring's `intercept`/`sigma`/`phi` draws) keep `jnp.asarray` because those wrappers stay typed `ArrayLike` by design. Update the `AGENTS.md` paragraph from Task 6 accordingly.

---

## Self-review

- Spec coverage: item 1 (Tasks 1, 2), item 2 (Task 3), item 3 (Tasks 4, 5), item 4 (Tasks 6, 8), item 5 (decision recorded, no task), tooling bump and notebook normalization (Task 0), docs and AGENTS.md updates folded into the tasks that change the contract.
- Type consistency: `Transition[Carry] -> dist.Distribution` and `Advance[Carry]` (Task 4) are what `var.py`'s docstring and `tests/test_markov.py` use; `SSOEMean`/`SSOEUpdate` (Task 5) are imported by `var.py`, exported from `__init__.py`, listed in `great-docs.yml` and asserted by `tests/test_package.py`; `ssoe(h, name, y, init_carry, mean, update, noise_dist, xs=None)` is the order used in README, docstrings, tests and notebooks; `innovations(h, name, prior, *, reparam=None)` matches README line 122; `update(carry, y_t, eps_t, x_t)` is the argument order in every snippet (the gated notebooks bind it as `update(level, y_t, _, gate_t)`).
- Notebook outputs: Tasks 3, 4, 5 change only how the same distributions are constructed and how the same per-step arithmetic is expressed (op-for-op identical; the one place that needs the mean inside `update` recomputes it through `mean(carry, x_t)`), so the PRNG stream and every stored output are unchanged; Task 6 removes no-op wrappers. No re-execution is required; the smoke runs in Task 5 Step 8 prove the rewritten model cells execute.
- Adversarial review (two independent reviewers plus probes, 2026-09-29) found and this revision fixed: four non-matching `OLD` literals, the missing markdown-cell rewrites, the `y_t - eps_t` mean recovery (not bit-exact), four unlisted `var_step` callers in `tests/test_var.py`, unused imports left behind (`Callable` in `var.py` and `example_models.py`, `Any` in `test_var.py`), the `x_t: Array | None` VARX snippet ty rejects, the raw jax scan error when `advance` is omitted with a structured carry, the `# asarray narrows` comment variant, the non-byte-stable notebook round trip, wrong expected-failure messages (Tasks 2, 3, 5), the non-discriminating leaf assertion, the `dynestyx_integration.ipynb` copy of `StateSpaceResult`, and the 101-character `var_step` annotation.
