"""Runtime type-checker configuration for the package's jaxtyping import hook.

``numpyro_forecast/__init__.py`` installs the jaxtyping import hook with
`typechecker`, so every package function and class is checked by beartype
with `BEARTYPE_CONF` at call time. The configuration turns on the PEP 484
implicit numeric tower: a Python ``int`` satisfies a ``float`` hint (and an
``int`` or ``float`` a ``complex`` hint), as it does for the static checker
``ty``. Without it, beartype rejects ``fourier_features(10, 7, 1)`` because
``7`` is not a ``float`` (issue #165). The tower is one-way: a ``float`` still
fails an ``int`` hint, so a count is never truncated silently.

NumPy scalars are not part of the tower: ``np.int64`` is not a subclass of
``int``. A parameter that accepts them says so in its hint with
`~~numpyro_forecast.typing.IntScalar` or
`~~numpyro_forecast.typing.RealScalar` and normalizes the value at the
function boundary (``operator.index`` or ``float``).

The package imports this module before it installs the hook, because the hook
resolves the dotted name ``numpyro_forecast.typecheck.typechecker`` by
attribute access on the package. The hook therefore does not wrap this module:
keep it free of functions and of package imports, so that no package code
escapes the runtime checks.
"""

from beartype import BeartypeConf, beartype

BEARTYPE_CONF = BeartypeConf(is_pep484_tower=True)
"""The beartype configuration of the package's runtime type checks."""

typechecker = beartype(conf=BEARTYPE_CONF)
"""The decorator that the jaxtyping import hook applies to every package function."""
