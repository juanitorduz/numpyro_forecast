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
- **Notebooks:** `uv run pytest --nbmake examples/` (executes the example notebooks)
- **All hooks:** `prek run --all-files`

## Guidelines

- Every function (public and private) must have complete input and return type
  hints. Type checking is enforced with `ty`.
- Every public function and class must have a NumPy-style docstring.
- Array shapes are annotated with `jaxtyping` (e.g. `Float[Array, "time obs"]`).
- Follow the array convention: time at axis `-2`, the observation dim at `-1`,
  batch dims to the left.
- Add tests for new functionality. Keep one logical change per pull request.
