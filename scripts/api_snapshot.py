"""Generate a great-docs API snapshot keyed by the curated reference entries.

``great-docs api-snapshot`` records only a package's top-level exports under bare names
(``Forecaster``), but the ``reference:`` section of ``great-docs.yml`` documents symbols by
their ``module.name`` path (``forecaster.Forecaster``), which is also how the reference page
files are named. The multi-version build prunes a release's API pages by matching page stems
against snapshot keys, so the snapshot must be keyed the same way for that pruning to work.

This script resolves every ``reference:`` entry of the repo's ``great-docs.yml`` against a
package tree (the working tree by default, or a release checkout via ``--package-root``) and
writes the resolvable ones to a snapshot JSON in the schema great-docs reads. Entries that do
not resolve are dropped with a notice: those symbols (or their module paths) do not exist in
that package tree, which is exactly what the versioned build needs to prune the corresponding
reference pages from that release's docs.

It is normally invoked by ``scripts/build_docs.py`` at build time, against a temporary
worktree of the newest release tag, with the output cached in the gitignored
``.great-docs-cache/``. Manual runs are only for debugging.

Usage:

```bash
uv run python scripts/api_snapshot.py                          # snapshot the working tree
uv run python scripts/api_snapshot.py --package-root <path>    # e.g. a release-tag worktree
```
"""

import argparse
import importlib
import inspect
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "great-docs.yml"
# Default output location: the gitignored build-time snapshot cache.
SNAPSHOT_DIR = REPO_ROOT / ".great-docs-cache" / "api-snapshots"
PACKAGE = "numpyro_forecast"

_MISSING = object()


def reference_entries(config_path: Path) -> list[str]:
    """Collect the dotted ``module.name`` entries of the curated API reference.

    Parameters
    ----------
    config_path
        Path to ``great-docs.yml``.

    Returns
    -------
    list[str]
        The ``contents:`` entries of every ``reference:`` section, in order.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entries: list[str] = []
    for section in config.get("reference", []):
        for item in section.get("contents", []):
            entries.append(item if isinstance(item, str) else item["name"])
    return entries


def resolve_entry(entry: str) -> object:
    """Resolve a dotted reference entry inside the imported package.

    Parameters
    ----------
    entry
        A reference entry such as ``"forecaster.Forecaster"``: everything up to
        the last dot is a module path below `PACKAGE`, the last component
        is an attribute of that module.

    Returns
    -------
    object
        The resolved symbol, or the `_MISSING` sentinel when the module
        path or the attribute does not exist in this package tree. An
        ``ImportError`` from outside the package (e.g. a missing third-party
        dependency) propagates so a broken environment cannot silently produce
        an emptier snapshot.
    """
    modname, _, attr = entry.rpartition(".")
    try:
        module = importlib.import_module(f"{PACKAGE}.{modname}" if modname else PACKAGE)
    except ModuleNotFoundError as err:
        if err.name is not None and err.name.startswith(PACKAGE):
            return _MISSING
        raise
    return getattr(module, attr, _MISSING)


def symbol_kind(obj: object) -> str:
    """Classify a symbol the way great-docs snapshots do.

    Parameters
    ----------
    obj
        The resolved symbol.

    Returns
    -------
    str
        ``"class"``, ``"function"``, or ``"attribute"``.
    """
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "attribute"


def build_snapshot(entries: list[str], version: str) -> tuple[dict[str, Any], list[str]]:
    """Build the snapshot document for the entries that resolve.

    Parameters
    ----------
    entries
        Dotted reference entries from `reference_entries()`.
    version
        Version label recorded in the snapshot.

    Returns
    -------
    tuple[dict[str, Any], list[str]]
        The snapshot document (great-docs ``ApiSnapshot`` schema, with symbols
        keyed by the dotted entry) and the entries that did not resolve.
    """
    symbols: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for entry in entries:
        obj = resolve_entry(entry)
        if obj is _MISSING:
            skipped.append(entry)
            continue
        info: dict[str, Any] = {"name": entry, "kind": symbol_kind(obj)}
        if inspect.isclass(obj):
            bases = [base.__name__ for base in obj.__bases__ if base is not object]
            if bases:
                info["bases"] = bases
        symbols[entry] = info
    return {"version": version, "package_name": PACKAGE, "symbols": symbols}, skipped


def main() -> int:
    """Resolve the reference entries and write the snapshot JSON.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Generate a great-docs API snapshot keyed by the curated reference entries."
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=REPO_ROOT,
        help="directory whose numpyro_forecast/ package tree to resolve against "
        "(default: the repo root, i.e. the working tree)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output file (default: .great-docs-cache/api-snapshots/<version>.json)",
    )
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    # Shadow any installed copy so the snapshot reflects exactly this tree.
    sys.path.insert(0, str(package_root))

    pyproject = tomllib.loads((package_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    snapshot, skipped = build_snapshot(reference_entries(CONFIG_PATH), version)
    for entry in skipped:
        print(f"  - {entry}: not in this package tree, pruned from the {version} docs")

    output = args.output or SNAPSHOT_DIR / f"{version}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(snapshot['symbols'])} symbols ({version}) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
