"""Tests for the runtime type-checker configuration of the jaxtyping import hook."""

import ast
from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation, BeartypeCallHintReturnViolation

import numpyro_forecast.typecheck
from numpyro_forecast.typecheck import typechecker


@typechecker
def _real(x: float) -> float:
    return x


@typechecker
def _complex(z: complex) -> complex:
    return z


@typechecker
def _count(n: int) -> int:
    return n


@typechecker
def _int_as_real(n: int) -> float:
    return n


@pytest.mark.parametrize("value", [7, True, 7.0, np.float64(7.0)], ids=repr)
def test_float_hint_follows_the_pep484_numeric_tower(value: float) -> None:
    # A Python int (and so a bool) satisfies a float hint, as it does for ty (issue #165).
    # np.float64 is a subclass of float and passes with or without the tower.
    assert _real(value) == value


@pytest.mark.parametrize("value", [7, 7.0, 7.0 + 1.0j], ids=repr)
def test_complex_hint_follows_the_pep484_numeric_tower(value: complex) -> None:
    assert _complex(value) == value


def test_return_hint_follows_the_pep484_numeric_tower() -> None:
    assert _int_as_real(3) == 3


@pytest.mark.parametrize("value", ["7", None, np.int64(7), np.float32(7.0)], ids=repr)
def test_float_hint_rejects_values_outside_the_tower(value: object) -> None:
    # NumPy scalars are not part of the tower (np.int64 is no int subclass, np.float32 no
    # float subclass): a parameter that takes them says so with IntScalar or RealScalar.
    with pytest.raises(BeartypeCallHintParamViolation):
        _real(value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [7.0, 7.5, np.float64(7.0), np.int64(7)], ids=repr)
def test_int_hint_is_not_widened(value: object) -> None:
    # The tower is one-way: a float never satisfies an int hint, so no count is truncated.
    with pytest.raises(BeartypeCallHintParamViolation):
        _count(value)  # ty: ignore[invalid-argument-type]


def test_return_hint_still_rejects_values_outside_the_tower() -> None:
    @typechecker
    def _text_as_real() -> float:
        return "7"  # ty: ignore[invalid-return-type]

    with pytest.raises(BeartypeCallHintReturnViolation):
        _text_as_real()


def test_typecheck_module_imports_only_beartype() -> None:
    """``typecheck.py`` is imported before the hook, so the hook never wraps it.

    A package module that it imported would be imported before the hook too and
    would lose its runtime checks. `test_every_package_function_is_runtime_type_checked`
    in ``tests/test_package.py`` catches a function defined here, but not a module
    without module-level functions (``typing.py``), so this test pins the imports.
    """
    tree = ast.parse(Path(numpyro_forecast.typecheck.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "typecheck.py must not use relative imports"
            imported.add((node.module or "").split(".")[0])
    assert imported == {"beartype"}
