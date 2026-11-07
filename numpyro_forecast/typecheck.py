"""Runtime type-checker configuration for the package's jaxtyping import hook.

``numpyro_forecast/__init__.py`` installs the jaxtyping import hook with
`typechecker`, so every package function and class is checked by beartype
with `BEARTYPE_CONF` at call time. Nested functions are the exception: they
carry ``@no_type_check``, because the hook would otherwise decorate them again
on every call of the enclosing function, and each decoration clears Python's
warning registries (issue #235). The configuration turns on the PEP 484
implicit numeric tower, which the static checker ``ty`` applies as well: a
Python ``int`` satisfies a ``float`` hint (and an ``int`` or ``float`` a
``complex`` hint), so a call that type-checks never fails at runtime because
an integer was passed where a ``float`` is hinted (``fit_pathfinder(...,
ftol=1)``, a ``BacktestResult`` built by hand with ``walltime=0``), and a
``-> float`` return may carry an ``int`` (issue #165). The tower is one-way:
a ``float`` still fails an ``int`` hint, so a count is never truncated
silently.

NumPy scalars are not part of the tower: ``np.int64`` is not a subclass of
``int``. A parameter that accepts them says so in its hint with
`~~numpyro_forecast.typing.IntScalar` or
`~~numpyro_forecast.typing.RealScalar` and normalizes the value at the
function boundary (``operator.index``, or ``float`` for a NumPy float); the
seasonal builders and ``pad_future`` take an integer period or fill through
``RealScalar``, which spells ``int``, not through the tower.

The package imports this module first inside the ``install_import_hook`` block:
the hook resolves the dotted name ``numpyro_forecast.typecheck.typechecker``
lazily, by attribute access on the package, when the first decorated function
is defined. Keep this module free of function definitions: the hook would try
to decorate one with the typechecker it is still defining, and fail at import.
"""

from beartype import BeartypeConf, beartype

BEARTYPE_CONF = BeartypeConf(is_pep484_tower=True)
"""The beartype configuration of the package's runtime type checks."""

typechecker = beartype(conf=BEARTYPE_CONF)
"""The decorator the jaxtyping import hook applies to every package function but nested ones."""
