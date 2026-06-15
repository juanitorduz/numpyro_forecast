# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`numpyro_forecast` is a JAX/NumPyro port of Pyro's `pyro.contrib.forecast`
module. The design context lives in the module docstrings (`forecaster.py`,
`evaluate.py`, `util.py`) and the example notebooks under `examples/` — read the
relevant ones before making non-trivial changes.

## Conventions

- **Array layout:** time at axis `-2`, observation/event dim at `-1`, batch dims
  to the left (matches Pyro).
- **Train vs forecast:** a single model handles both. In-sample time latents use
  a fixed site name (`drift`); the forecast horizon uses a separate `_future`
  site so `AutoNormal` never resizes and `Predictive` draws the suffix from the
  prior. The horizon is derived from shapes (`covariates` longer than `data`).
- **Functional style:** pure model functions, explicit `PRNGKey` threading,
  `jax.lax.scan` for latent levels, no global parameter store.

## Hard requirements

- Every function (public and private) has complete input and return type hints,
  checked with `ty`.
- Every public function/class has a NumPy-style docstring (ruff `D`/`DOC`).
- **jaxtyping:** annotate array shapes as `Float[Array, " time obs"]` with a
  **leading space** in the shape string (per the jaxtyping FAQ this turns ruff's
  `F821` into `F722`, which we ignore globally; `F821` stays active otherwise).
  Do **not** use `from __future__ import annotations` (incompatible with runtime
  type checking).

## Notebooks & ArviZ (>= 1.0)

The examples target the new ArviZ (>= 1.0). Key differences from legacy arviz:

- `InferenceData` is gone; ArviZ now wraps `xarray.DataTree`. `az.from_dict(...)`
  still works for assembling `posterior` / `posterior_predictive` groups.
- ArviZ is modular: `arviz-base` (data), `arviz-stats` (stats, `.azstats`
  accessor / `az.hdi`), `arviz-plots` (plots) — all under the `arviz` namespace.
- **Use `az.plot_lm` to plot forecasts with HDI bands against observed data —
  not the legacy `az.plot_hdi`.**
- Defaults changed: credible interval `0.94 → 0.89`, interval kind `hdi → eti`
  (pass explicit arguments where the published tutorials assumed the old values).

## Commands

```bash
uv sync --all-extras
prek run --all-files
uv run ruff check . && uv run ruff format --check .
uv run ty check numpyro_forecast/
uv run pytest
```
